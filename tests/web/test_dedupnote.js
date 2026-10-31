// jsdom test for the convergence banner (dedupnote.js): a payload with
// converged === false shows the red "Fit did not converge" banner under the
// constraint banner; a converged (or flag-less legacy) payload hides it again.
//
// The module listens on the "pyroorb:render" event app.js's render() dispatches,
// so the test drives REAL fits through a stubbed fetch (runFit()) rather than
// simulating the hook -- if render() ever stops dispatching, this fails.
//
//     npm i jsdom && node tests/web/test_dedupnote.js
//
// Skips (exit 0) when jsdom is not installed.
const fs = require("fs");
const path = require("path");
let JSDOM;
try { ({ JSDOM } = require("jsdom")); }
catch { console.log("SKIP: jsdom not installed (npm i jsdom to run this test)"); process.exit(0); }

const ROOT = path.join(__dirname, "..", "..", "src", "pyroorb", "web", "static");
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8");
const appjs = // fitjob.js first: app.js's runFit drives the fit through FitJob (the
// queued /api/fit_start + /api/fit_result path), so the browser global it
// defines has to exist before any fit runs.
fs.readFileSync(path.join(ROOT, "fitjob.js"), "utf8") + "\n" +
  fs.readFileSync(path.join(ROOT, "app.js"), "utf8");
const ddjs = fs.readFileSync(path.join(ROOT, "dedupnote.js"), "utf8");

// Minimal fit payload shaped like serialise_fit's output (see dom_smoke.test.js).
const PAYLOAD = {
  ok: true, object: "TEST 1", combined: false, designations: ["TEST 1"], mp_names: [],
  other_designations: [], force_model: "assist", iod_method: "gauss",
  summary: { rms_residual: 0.5, rms_normalized: 0.7, n_obs: 8, n_obs_total: 10,
             n_excluded: 2, n_oppositions: 1, arc_days: 30, obs_codes: 1, systematic: null },
  epoch: { jd: 2461000.5, iso: "2026-01-01T00:00:00" },
  nongrav: { model: null, params: {}, sigmas: {} },
  constraints: {}, asteroids: false,
  physical: { class: "MBA", pha: false, moid_earth_au: 1.0, tisserand_jupiter: 3.5,
              U: 1, H: 20, H_sigma: 0.1, G: 0.15, n_mags: 5, frame: "J2000" },
  elements: [], planetocentric: null,
  residuals: Array.from({ length: 10 }, (_, i) => ({
    i, jd_utc: 2460990.5 + i, obscode: "F51", designation: "TEST 1",
    ra_deg: 10 + i, dec_deg: 5, mag: 20, band: "G", computed_mag: 20, dmag: 0,
    r_au: 2.5, delta_au: 1.6, d_ra_arcsec: 0.1, d_dec_arcsec: -0.1,
    sigma_ra: 0.5, sigma_dec: 0.5, resid_sigmas: 0.3, included: i < 8,
    obs_id: null, trksub: null, ades_rms_ra: null, ades_rms_dec: null,
    mag_sigma: null, log_snr: null, obs80: " ".repeat(80),
  })),
  observatories: [{ obscode: "F51", name: "Pan-STARRS", n: 10, n_used: 8, median_resid: 0.14 }],
  history: [], orbit3d: { a: 2.5, e: 0.1, i: 5, node: 10, peri: 20, M: 30, epoch_jd: 2461000.5,
    n_deg_per_day: 0.25, q: 2.25, Q: 2.75, path: null, position: null },
  stage_residuals: [], session: "s1", meta: {},
};

const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                      { url: "http://localhost/", runScripts: "dangerously",
                        pretendToBeVisual: true });
const w = dom.window;
w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
w.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
w.requestAnimationFrame = (f) => setTimeout(f, 0);
w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
  get: (t, k) => (k === "measureText" ? () => ({ width: 1 }) : () => {}) });

let dropped;                         // meta.duplicates_dropped the next fit reports
w.fetch = async (url) => {
  if (String(url).startsWith("/api/export/")) return { text: async () => "" };
  const p = JSON.parse(JSON.stringify(PAYLOAD));
  if (dropped !== undefined) p.meta = { duplicates_dropped: dropped };
  return { json: async () => p };
};

// Queued fitting: runFit goes through FitJob, which POSTs /api/fit_start and reads
// the payload off the finished job.  Wrap the stub rather than short-circuit it, so
// whatever this test's own stub does to the payload still applies.
w.fetch = ((f) => async (url, opts) => {
  const u = String(url);
  if (u.startsWith("/api/session")) {
    const body = { ok: true, session: "s1" };
    return { json: async () => body, text: async () => JSON.stringify(body) };
  }
  if (!u.startsWith("/api/fit_start")) return f(url, opts);
  const inner = await f("/api/fit", opts);
  const job = { ok: true, job: "job1", state: "done", done: true,
                result: await inner.json() };
  return { json: async () => job, text: async () => JSON.stringify(job) };
})(w.fetch);


// app.js reads every response body as TEXT and parses it (so a proxy answering
// a request it gave up on with a non-JSON page yields a readable message
// instead of "Unexpected token 'u'"). Give these mocks a text() for free.
w.fetch = ((f) => async (...a) => {
  const r = await f(...a);
  return r.text ? r : { ...r, text: async () => JSON.stringify(await r.json()) };
})(w.fetch);

for (const src of [appjs, ddjs]) {
  const scr = w.document.createElement("script");
  scr.textContent = src;
  w.document.body.appendChild(scr);
}

let failures = 0;
function check(name, cond) {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}`);
  if (!cond) failures++;
}

async function fit(rec) {
  dropped = rec;
  w.document.getElementById("desig").value = "TEST 1";
  await w.eval("runFit()");
  await new Promise((r) => setTimeout(r, 20));
}

(async () => {
  await new Promise((r) => setTimeout(r, 0));    // let jsdom settle
  check("no note before any payload", !w.document.getElementById("dedupNote"));

  await fit([{ designation: "P12pwHF", kept_designation: "2013 PY26", obscode: "O18", jd_utc: 2461265.4 },
             { designation: "P12pwHF", kept_designation: "2013 PY26", obscode: "O18", jd_utc: 2461265.41 }]);
  const box = w.document.getElementById("dedupNote");
  if (!box) {
    console.log('  FAIL  no note after a fit -- did render() stop dispatching "pyroorb:render"?');
    process.exit(1);
  }
  check("note created next to constraintBanner",
        box.previousElementSibling === w.document.getElementById("constraintBanner"));
  check("shown when duplicates were dropped", !box.hidden);
  check("counts the dropped copies", /2 cross-listed observations/.test(box.textContent));
  check("names both designations", /P12pwHF/.test(box.textContent) && /2013 PY26/.test(box.textContent));
  check("is informational, not severe", !box.classList.contains("severe"));

  await fit([]);
  check("hidden again for an empty record", box.hidden);

  await fit(undefined);
  check("hidden for a payload without the record (legacy)", box.hidden);

  console.log(failures ? `${failures} FAILED` : "all passed");
  process.exit(failures ? 1 : 0);
})();
