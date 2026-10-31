// jsdom test for the degrees-of-freedom banner (dof.js): a payload whose fit has no
// spare residual rows shows a banner saying the sigmas are not measured uncertainties;
// a comfortably over-determined payload hides it again.
//
// Like the convergence banner it listens on the "pyroorb:render" event app.js's render()
// dispatches, so the test drives REAL fits through a stubbed fetch (runFit()).
//
//     npm i jsdom && node tests/web/test_dof.js
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
const dofjs = fs.readFileSync(path.join(ROOT, "dof.js"), "utf8");

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

let dofBlock;                        // what the next served fit reports as p.dof
let constraintBlock = null;          // ... and as p.constraint (the unconstrained banner)
w.fetch = async (url) => {
  if (String(url).startsWith("/api/export/")) return { text: async () => "" };
  const p = JSON.parse(JSON.stringify(PAYLOAD));
  p.dof = dofBlock === undefined ? null : dofBlock;
  p.constraint = constraintBlock;
  if (constraintBlock) p.orbit_suppressed = true;
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

for (const src of [appjs, dofjs]) {
  const scr = w.document.createElement("script");
  scr.textContent = src;
  w.document.body.appendChild(scr);
}

let failures = 0;
function check(name, cond) {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}`);
  if (!cond) failures++;
}

// dof = 2*n_obs - n_params, exactly as serialise_fit computes it.
function block(nObs, nParams = 6) {
  return { n_obs: nObs, n_resids: 2 * nObs, n_params: nParams,
           dof: 2 * nObs - nParams };
}

// A real fit through app.js: fetch is stubbed, everything else (render and the
// "pyroorb:render" dispatch it ends on) is the shipping code path.
async function fit(d) {
  dofBlock = d;
  w.document.getElementById("desig").value = "TEST 1";
  await w.eval("runFit()");
  await new Promise((r) => setTimeout(r, 20));
}

(async () => {
  await new Promise((r) => setTimeout(r, 0));    // let jsdom settle
  check("no banner before any payload", !w.document.getElementById("dofBanner"));

  await fit(block(3));                            // 6 rows, 6 params -> dof 0
  const box = w.document.getElementById("dofBanner");
  if (!box) {
    console.log('  FAIL  no banner after a fit -- did render() stop dispatching ' +
                '"pyroorb:render"?');
    process.exit(1);
  }
  check("shown at dof 0", !box.hidden);
  check("severe at dof 0", box.classList.contains("severe"));
  check("says there are no degrees of freedom", /no degrees of freedom/i.test(box.textContent));
  check("says the sigmas are not measured", /not measured uncertainties/i.test(box.textContent));
  check("shows the arithmetic", /3 observations give 6 residuals for 6 fitted parameters/
        .test(box.textContent));

  await fit(block(4));                            // 8 rows, 6 params -> dof 2
  check("shown at dof 2", !box.hidden);
  check("not severe at dof 2", !box.classList.contains("severe"));
  check("says barely over-determined", /barely over-determined/i.test(box.textContent));

  await fit(block(5));                            // dof 4 -- past the thin threshold
  check("hidden at dof 4", box.hidden);

  await fit(block(4, 9));                         // a 9-param comet fit on 4 obs -> dof -1
  check("shown for a negative dof", !box.hidden);
  check("severe for a negative dof", box.classList.contains("severe"));

  await fit(undefined);                           // legacy payload, no dof block
  check("hidden for a payload without the block (legacy)", box.hidden);

  // Both apply: ONE banner. The dof count rides on the unconstrained
  // banner as a sentence; the standalone banner and its sigma caveat stay away.
  constraintBlock = { level: "unconstrained", headline: "Essentially unconstrained — 0.0-day, 1-opposition arc",
                      detail: "The arc is too short to determine the orbit." };
  await fit(block(3));
  const cb = w.document.getElementById("constraintBanner");
  check("no separate dof banner under the unconstrained one", box.hidden);
  check("the unconstrained banner carries the dof sentence",
        !cb.hidden && /no degrees of freedom \(dof 0\)/i.test(cb.textContent));
  check("...after its own detail", /too short to determine the orbit\. With 3 observations/.test(cb.textContent));
  check("...without the sigma caveat", !/measured uncertainties/i.test(cb.textContent));
  await fit(block(4));
  check("thin dof folds in too", box.hidden && /leave only 2 degrees of freedom/.test(cb.textContent));
  await fit(block(8));
  check("a comfortable dof adds nothing", !/degrees? of freedom/.test(cb.textContent));
  constraintBlock = null;
  await fit(block(3));
  check("alone again, the dof banner comes back", !box.hidden);

  console.log(failures ? `\n${failures} failed` : "\nall dof-banner tests passed");
  process.exit(failures ? 1 : 0);
})();
