// jsdom test for the tracklet-by-tracklet digest2 column (d2tracklets.js). On a q < 1.8
// payload (d2_tracklets.eligible) a "D2 per tracklet" button appears beside "＋ addl info";
// turning it on POSTs /api/d2_tracklets and adds a column right after the station code holding
// each tracklet's NEO score on its FIRST row only; clicking the column header flips raw <->
// no-ID, the same shared setting as the digest2 note's switch (d2note.js); an ineligible
// payload hides the button. Drives real fits through a stubbed fetch (runFit()).
//
//     npm i jsdom && node tests/web/test_d2tracklets.js
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
const d2js = fs.readFileSync(path.join(ROOT, "d2note.js"), "utf8");
const trkjs = fs.readFileSync(path.join(ROOT, "d2tracklets.js"), "utf8");

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

let eligible = true;
const posted = [];
const TRK = [
  { first: 0, n: 3, station: "F51", rms: 0.1, scores: { NEO: 88 }, noid_scores: { NEO: 41 },
    shown: [["NEO", "NEO(q < 1.3)", 88]], shown_noid: [["NEO", "NEO(q < 1.3)", 41]] },
  { first: 5, n: 2, station: "F51", rms: 0.1, scores: { NEO: 12 }, noid_scores: { NEO: 70 },
    shown: [["NEO", "NEO(q < 1.3)", 12]], shown_noid: [["NEO", "NEO(q < 1.3)", 70]] },
];
w.fetch = async (url, opts) => {
  const u = String(url);
  if (u.startsWith("/api/export/")) return { text: async () => "" };
  if (u.startsWith("/api/d2_tracklets")) {
    posted.push(JSON.parse(opts.body));
    return { json: async () => ({ ok: true, truncated: false, tracklets: TRK }) };
  }
  const p = JSON.parse(JSON.stringify(PAYLOAD));
  p.d2_tracklets = { eligible };
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

for (const src of [appjs, d2js, trkjs]) {
  const scr = w.document.createElement("script");
  scr.textContent = src;
  w.document.body.appendChild(scr);
}

let failures = 0;
function check(name, cond) {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}`);
  if (!cond) failures++;
}
const tick = (ms = 30) => new Promise((r) => setTimeout(r, ms));
const $ = (id) => w.document.getElementById(id);
async function fit() {
  $("desig").value = "TEST 1";
  await w.eval("runFit()");
  await tick();
}
const headCells = () => [...w.document.querySelectorAll("#obsScroll table.obs thead th")];
const rowCells = (i) => [...w.document.querySelectorAll(`#obsScroll table.obs tbody tr[data-i="${i}"] td`)];

(async () => {
  await tick(0);
  await fit();
  const btn = $("d2TrkBtn");
  check("the button sits beside '＋ addl info'", !!btn && btn.previousElementSibling === $("addlBtn"));
  check("...shown for q < 1.8", !btn.hidden);
  check("no column until asked", !w.document.querySelector("th.d2h") && posted.length === 0);

  btn.click();
  await tick(60);
  check("turning it on fetches the scores once", posted.length === 1 && posted[0].session === "s1");
  const hs = headCells();
  check("the column header follows the station code", hs[2].textContent === "code" &&
        hs[3].classList.contains("d2h") && /D2 raw/.test(hs[3].textContent));
  check("the first row of a tracklet carries its NEO score", rowCells(0)[3].textContent === "88");
  check("...a later row of it is blank", rowCells(1)[3].textContent === "");
  check("the next tracklet's first row carries its own", rowCells(5)[3].textContent === "12");
  check("every row has the extra cell", rowCells(1).length === hs.length);
  check("the cell's tooltip names the tracklet", /3 obs from F51/.test(rowCells(0)[3].title));

  headCells()[3].click();
  await tick(60);
  check("clicking the header flips to no-ID", /D2 no-ID/.test(headCells()[3].textContent) &&
        rowCells(0)[3].textContent === "41" && rowCells(5)[3].textContent === "70");
  check("a score at the NEOCP bar is picked out", rowCells(5)[3].classList.contains("d2hot") &&
        !rowCells(0)[3].classList.contains("d2hot"));
  check("...the same shared setting the note uses", w.D2Flavour.get() === "noid");
  check("no refetch for a flavour flip", posted.length === 1);
  headCells()[3].click();
  await tick(60);
  check("and back to raw", /D2 raw/.test(headCells()[3].textContent) && rowCells(0)[3].textContent === "88");

  await fit();
  await tick(60);
  check("a new payload re-fetches, the column stays on", posted.length === 2 && !!w.document.querySelector("th.d2h"));

  btn.click();
  await tick(30);
  check("turning it off removes the column", !w.document.querySelector("th.d2h") &&
        headCells().length === rowCells(0).length && !btn.classList.contains("active"));

  eligible = false;
  btn.click();                       // on again, but the next payload is not eligible
  await fit();
  check("q >= 1.8: no button, no column", btn.hidden && !w.document.querySelector("th.d2h"));

  console.log(failures ? `${failures} FAILED` : "all passed");
  process.exit(failures ? 1 : 0);
})();
