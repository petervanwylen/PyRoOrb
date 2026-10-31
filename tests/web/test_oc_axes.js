// jsdom test for the O−C plot's axis switch: the colour key (● RA·cosδ / ● Dec) IS
// the control, so a single-axis view costs no header room. Three states cycle --
// both, Dec only, RA only -- one axis always stays on (an empty plot is not a view),
// and the y-scale follows the SHOWN axes (a hidden big Dec miss must stop squashing
// the RA points the user asked to look at).
//
//     npm i jsdom && node tests/web/test_oc_axes.js
//
// Skips (exit 0) when jsdom is not installed.
const fs = require("fs");
const path = require("path");
let JSDOM;
try { ({ JSDOM } = require("jsdom")); }
catch { console.log("SKIP: jsdom not installed (npm i jsdom to run this test)"); process.exit(0); }

const ROOT = path.join(__dirname, "..", "..", "src", "pyroorb", "web", "static");
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8");
const appjs = fs.readFileSync(path.join(ROOT, "fitjob.js"), "utf8") + "\n" +
  fs.readFileSync(path.join(ROOT, "app.js"), "utf8");

// RA residuals are small, Dec residuals are huge: the axis that is hidden must not
// set the scale, which is what makes the two amax reads below different.
const PAYLOAD = {
  ok: true, object: "TEST 1", combined: false, designations: ["TEST 1"], mp_names: [],
  other_designations: [], force_model: "assist", iod_method: "gauss",
  summary: { rms_residual: 0.5, rms_normalized: 0.7, n_obs: 6, n_obs_total: 6,
             n_excluded: 0, n_oppositions: 1, arc_days: 6, obs_codes: 1, systematic: null },
  epoch: { jd: 2461000.5, iso: "2026-01-01T00:00:00" },
  nongrav: { model: null, params: {}, sigmas: {} },
  constraints: {}, asteroids: false,
  physical: { class: "MBA", pha: false, moid_earth_au: 1.0, tisserand_jupiter: 3.5,
              U: 1, H: 20, H_sigma: 0.1, G: 0.15, n_mags: 5, frame: "J2000" },
  elements: [], planetocentric: null,
  residuals: Array.from({ length: 6 }, (_, i) => ({
    i, jd_utc: 2460990.5 + i, obscode: "F51", designation: "TEST 1",
    ra_deg: 10 + i, dec_deg: 5, mag: 20, band: "G", computed_mag: 20, dmag: 0,
    r_au: 2.5, delta_au: 1.6, d_ra_arcsec: 0.2, d_dec_arcsec: 4.0,
    sigma_ra: 0.5, sigma_dec: 0.5, resid_sigmas: 0.3, included: true,
    obs_id: null, trksub: null, ades_rms_ra: null, ades_rms_dec: null,
    mag_sigma: null, log_snr: null, obs80: " ".repeat(80),
  })),
  observatories: [{ obscode: "F51", name: "Pan-STARRS", n: 6, n_used: 6, median_resid: 0.2 }],
  history: [], orbit3d: { a: 2.5, e: 0.1, i: 5, node: 10, peri: 20, M: 30, epoch_jd: 2461000.5,
    n_deg_per_day: 0.25, q: 2.25, Q: 2.75, path: null, position: null },
  stage_residuals: [], session: "s1", meta: {},
};
const EPHEM = { ok: true, designation: "TEST 1", site: "500", rows: [], hidden: {},
                hidden_note: "", site_has_horizon: true };
const IMPACT = { ok: true, assessed: false, message: "no impact" };

const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                      { url: "http://localhost/", runScripts: "dangerously",
                        pretendToBeVisual: true });
const w = dom.window;
w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
w.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
w.requestAnimationFrame = (f) => setTimeout(f, 0);
w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
  get: (t, k) => (k === "measureText" ? () => ({ width: 1 }) : () => {}) });

w.fetch = async (url) => {
  const u = String(url);
  const body = u.startsWith("/api/ephem") ? EPHEM : u.startsWith("/api/impact") ? IMPACT : PAYLOAD;
  const json = JSON.parse(JSON.stringify(body));
  return { json: async () => json, text: async () => JSON.stringify(json) };
};
w.fetch = ((f) => async (url, opts) => {
  const u = String(url);
  if (u.startsWith("/api/session")) {
    const body = { ok: true, session: "s1" };
    return { json: async () => body, text: async () => JSON.stringify(body) };
  }
  if (!u.startsWith("/api/fit_start")) return f(url, opts);
  const inner = await f("/api/fit", opts);
  const job = { ok: true, job: "job1", state: "done", done: true, result: await inner.json() };
  return { json: async () => job, text: async () => JSON.stringify(job) };
})(w.fetch);

const scr = w.document.createElement("script");
scr.textContent = appjs;
w.document.body.appendChild(scr);

const tick = (ms = 20) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const $ = (id) => w.document.getElementById(id);
  const assert = (c, m) => { if (!c) { console.error("FAIL:", m); process.exit(1); }
                             else console.log("  PASS ", m); };
  // jsdom lays nothing out, so ocBody has no size: pin one so the SVG is drawn.
  const box = $("ocBody");
  Object.defineProperty(box, "clientWidth", { value: 400 });
  Object.defineProperty(box, "clientHeight", { value: 200 });

  $("desig").value = "TEST 1";
  await w.eval("runFit()");
  await tick();

  const key = (axis) => $("ocAxisToggle").querySelector(`button[data-axis="${axis}"]`);
  const dots = () => ({
    ra: (box.innerHTML.match(/fill="var\(--ra\)"/g) || []).length,
    dec: (box.innerHTML.match(/fill="var\(--dec\)"/g) || []).length,
  });
  // the topmost y-axis label is amax in the active unit
  const amax = () => Math.abs(parseFloat((box.innerHTML.match(/>([-\d.]+)″</) || [])[1]));

  let d = dots();
  assert(d.ra === 6 && d.dec === 6, "both axes drawn to start");
  const bothMax = amax();
  assert(bothMax >= 4, `scale covers the big Dec residuals (${bothMax}″)`);

  key("dec").click(); await tick();
  d = dots();
  assert(d.ra === 6 && d.dec === 0, "clicking the Dec key leaves RA only");
  assert(key("dec").classList.contains("off") && !key("ra").classList.contains("off"),
         "the hidden key dims, the shown one does not");
  assert(amax() < bothMax, `the scale follows the shown axis (${amax()}″ < ${bothMax}″)`);

  // clicking the ONLY visible axis shows the other one alone rather than nothing
  key("ra").click(); await tick();
  d = dots();
  assert(d.ra === 0 && d.dec === 6, "clicking the last visible axis swaps to the other");

  key("ra").click(); await tick();
  d = dots();
  assert(d.ra === 6 && d.dec === 6, "clicking the hidden axis brings both back");

  // the unit switch still works with one axis hidden
  key("ra").click(); await tick();
  $("ocUnitToggle").querySelector('button[data-unit="sigma"]').click(); await tick();
  assert(/σ</.test(box.innerHTML), "σ units still apply to a single-axis plot");
  d = dots();
  assert(d.ra === 0 && d.dec === 6, "and the hidden axis stays hidden across the unit switch");

  console.log("\nPASS: O−C axis switch (RA only / Dec only / both)");
})();
