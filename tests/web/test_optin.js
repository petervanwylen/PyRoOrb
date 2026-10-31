// jsdom test for "show an orbit anyway" (orbitsuppress.js's opt-in bar) and the faint orbit
// family (orbitfan.js). A suppressed payload with a session offers the button; clicking it
// POSTs /api/nominal and renders the asked-for orbit -- values marked "≈", sigma column
// "undetermined", the bar naming the most probable orbit and its distance, the orbit exports
// enabled; "Back to the ranges" POSTs {on:false} and the page is suppressed again. The fan
// module must stroke every family orbit through the shared _drawPlanets step.
//
// Drives real fits through a stubbed fetch (runFit()) so an app.js that stopped calling the
// hooks fails here.
//
//     npm i jsdom && node tests/web/test_optin.js
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
const osjs = fs.readFileSync(path.join(ROOT, "orbitsuppress.js"), "utf8");
const fanjs = fs.readFileSync(path.join(ROOT, "orbitfan.js"), "utf8");

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
  elements: [
    { name: "a", label: "semi-major axis", unit: "au", value: 2.5, sigma: 0.001 },
    { name: "e", label: "eccentricity", unit: "", value: 0.1, sigma: 0.0002 },
    { name: "asc_node", label: "ascending node", unit: "deg", value: 10, sigma: 0.01 },
  ],
  planetocentric: null, orbit_suppressed: false, constraint: null,
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


const BANNER = {
  level: "unconstrained", method: "range",
  headline: "Essentially unconstrained — 0.1-day, 1-opposition arc",
  detail: "The arc is too short to determine the orbit.",
  ranges: { a: [0.56, 3.65], e: [0.40, 0.91], asc_node: [0, 360] },
  ranges_method: "mc", ranges_pct: [5, 95],
};
const FAMILY = { source: "range", orbits: [
  { path: [[1, 0, 0], [0, 1, 0], [-1, 0, 0]], closed: true, weight: 1.0 },
  { path: [[2, 0, 0], [0, 2, 0], [-2, 0, 0]], closed: true, weight: 0.2 },
  { path: [[3, 0, 0], [0, 3, 0], [-3, 0, 0]], closed: true, weight: 0.0 } ] };

function suppressed(p) {
  p.orbit_suppressed = true;
  p.constraint = JSON.parse(JSON.stringify(BANNER));
  p.physical = { ...p.physical, class: null, pha: null, moid_earth_au: null,
                 tisserand_jupiter: null, U: null, H: null, H_sigma: null };
  p.orbit3d = null;
  p.orbit_family = FAMILY;
  p.elements.forEach((e) => { e.range = BANNER.ranges[e.name] || null; e.value = null; e.sigma = null; });
  return p;
}
function optedIn(p) {
  p.orbit_suppressed = false;
  p.constraint = JSON.parse(JSON.stringify(BANNER));
  p.nominal_optin = { source: "population", rho_med: 0.178,
                      rho_pct: { "5": 0.0119, "50": 0.178, "95": 0.438 }, vmag: 21.6 };
  p.physical = { ...p.physical, pha: null, moid_earth_au: null, tisserand_jupiter: null, U: null };
  p.orbit_family = FAMILY;
  p.elements.forEach((e) => { e.range = BANNER.ranges[e.name] || null; });
  return p;
}

const posted = [];
w.fetch = async (url, opts) => {
  const u = String(url);
  if (u.startsWith("/api/export/")) return { text: async () => "" };
  const p = JSON.parse(JSON.stringify(PAYLOAD));
  if (u.startsWith("/api/nominal")) {
    const b = JSON.parse(opts.body);
    posted.push(b);
    return { json: async () => (b.on ? optedIn(p) : suppressed(p)) };
  }
  return { json: async () => suppressed(p) };
};
// Queued fitting: runFit goes through FitJob (POST /api/fit_start, read the finished job).
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

// app.js reads every response body as TEXT and parses it.
w.fetch = ((f) => async (...a) => {
  const r = await f(...a);
  return r.text ? r : { ...r, text: async () => JSON.stringify(await r.json()) };
})(w.fetch);

for (const src of [appjs, osjs, fanjs]) {
  const scr = w.document.createElement("script");
  scr.textContent = src;
  w.document.body.appendChild(scr);
}

let failures = 0;
function check(name, cond) {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}`);
  if (!cond) failures++;
}
const tick = (ms = 20) => new Promise((r) => setTimeout(r, ms));
const $ = (id) => w.document.getElementById(id);
const exportBtn = (kind) => w.document.querySelector(`#exportMenu button[data-export="${kind}"]`);

(async () => {
  await tick(0);
  $("desig").value = "TEST 1";
  await w.eval("runFit()");
  await tick();

  const bar = $("optinBar");
  check("the bar exists on a suppressed orbit", !!bar && !bar.hidden);
  const on = $("optinOn");
  check("...offering 'Show an orbit anyway'", !!on && /Show an orbit anyway/.test(on.textContent));
  check("orbit exports start disabled", exportBtn("mpcorb") && exportBtn("mpcorb").disabled);
  check("the 3D note names the faint family",
        /faint lines are 3 that fit/.test($("orbitSuppressNote").textContent));
  check("...and moves off the drawing", $("orbitSuppressNote").classList.contains("with-fan"));

  on.click();
  await tick(60);
  check("clicking POSTs /api/nominal with the session, on:true",
        posted.length === 1 && posted[0].session === "s1" && posted[0].on === true);
  const txt = $("elemBody").textContent;
  check("values come back marked ≈", /≈ 2\.5/.test(txt));
  check("the sigma column says undetermined", /undetermined/.test(txt));
  check("the range rides on as a tooltip",
        /0\.56 … 3\.65/.test($("elemBody").querySelector("td.os-optin").getAttribute("title")));
  check("the bar names the most probable orbit and its distance",
        /most probable orbit/.test($("optinBar").textContent) &&
        /0\.178 au/.test($("optinBar").textContent));
  check("the banner stays up", !$("constraintBanner").hidden);
  check("orbit exports are enabled once asked", !exportBtn("mpcorb").disabled);
  check("the 3D 'no orbit' note is gone", $("orbitSuppressNote").hidden);

  $("optinOff").click();
  await tick(60);
  check("'Back to the ranges' POSTs on:false", posted.length === 2 && posted[1].on === false);
  check("...and the page is suppressed again", exportBtn("mpcorb").disabled &&
        !/≈/.test($("elemBody").textContent) && !!$("optinOn"));

  // the fan: every family orbit is stroked through the shared planets step, fainter for a
  // lower weight, and nothing at all without a family
  const strokes = w.eval(`(() => {
    const alphas = [];
    const ctx = { save() {}, restore() {}, set globalAlpha(v) { this._a = v; }, get globalAlpha() { return this._a; } };
    const f = { ctx, stroke: () => alphas.push(ctx._a), S: (p) => p };
    OrthoView.prototype._drawPlanets.call(Object.create(OrthoView.prototype), f, 2461000.5, { planets: [] });
    return alphas;
  })()`);
  check("one stroke per family orbit", strokes.length === 3);
  check("opacity follows the weight", strokes[0] > strokes[1] && strokes[1] > strokes[2] && strokes[2] > 0);

  console.log(failures ? `${failures} FAILED` : "all passed");
  process.exit(failures ? 1 : 0);
})();
