// jsdom test for orbitsuppress.js: when the payload says the arc does not determine an orbit
// (`orbit_suppressed`, from src/pyroorb/constraint.py) the elements panel shows an interval per element
// and NO value or sigma, and the orbit exports are disabled -- while the ephemeris export, which
// is a sky prediction rather than an orbit, stays available. A determined payload must get its
// ordinary value +/- sigma rows and its exports back.
//
// Drives real fits through a stubbed fetch (runFit()) rather than calling the module directly,
// so an app.js that stopped calling the hooks fails here.
//
//     npm i jsdom && node tests/web/test_orbitsuppress.js
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

let suppress;                        // whether the next fit reports a suppressed orbit
w.fetch = async (url) => {
  if (String(url).startsWith("/api/export/")) return { text: async () => "" };
  const p = JSON.parse(JSON.stringify(PAYLOAD));
  if (suppress) {
    p.orbit_suppressed = true;
    p.constraint = {
      level: "unconstrained", method: "cov",
      headline: "Essentially unconstrained \u2014 0.1-day, 1-opposition arc",
      detail: "The arc is too short to determine the orbit.",
      s_max: 1.6, bound_fraction: 0.05, n_modes: null, U: 11.2,
      ranges: { a: [0.56, 3.65], e: [0.40, 0.91], asc_node: [0, 360] },
      ranges_method: "mc", ranges_pct: [5, 95],
    };
    p.physical = { ...p.physical, class: null, pha: null, moid_earth_au: null,
                   tisserand_jupiter: null, U: null, H: null, H_sigma: null };
    p.orbit3d = null;
    p.elements.forEach((e) => {
      e.range = p.constraint.ranges[e.name] || null;
      e.value = null; e.sigma = null;
    });
  }
  return { json: async () => p };
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

for (const src of [appjs, osjs]) {
  const scr = w.document.createElement("script");
  scr.textContent = src;
  w.document.body.appendChild(scr);
}

let failures = 0;
function check(name, cond) {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}`);
  if (!cond) failures++;
}

async function fit(sup) {
  suppress = sup;
  w.document.getElementById("desig").value = "TEST 1";
  await w.eval("runFit()");
  await new Promise((r) => setTimeout(r, 20));
}

const exportBtn = (kind) =>
  w.document.querySelector(`#exportMenu button[data-export="${kind}"]`);

(async () => {
  await new Promise((r) => setTimeout(r, 0));    // let jsdom settle

  await fit(true);
  const body = w.document.getElementById("elemBody");
  const txt = body.textContent;
  check("shows the interval, not a value", /0\.56\s*…\s*3\.65/.test(txt));
  check("carries the element's unit once", /3\.65\s*au/.test(txt));
  check("no nominal value leaks through", !/2\.5\d*\s*au/.test(txt));
  check("no ± sigma anywhere", !/±/.test(txt));
  check("a free angle reads 0 … 360", /0\s*…\s*360\s*deg/.test(txt));
  check("labels the percentile band", /5–95%/.test(txt));
  check("orbit exports disabled", ["mpec", "elements", "mpcorb", "sof", "state", "covar"]
        .every((k) => exportBtn(k).disabled));
  check("ephemeris export still available", !exportBtn("ephem").disabled);
  check("the banner still explains why", !w.document.getElementById("constraintBanner").hidden);
  const note = w.document.getElementById("orbitSuppressNote");
  check("the 3D pane says why it is empty", note && !note.hidden && /No orbit to draw/.test(note.textContent));
  const phys = w.document.getElementById("physBody").textContent;
  check("no class / MOID / U / H derived from the hidden orbit",
        !/MBA/.test(phys) && !/\bPHA\b/.test(w.document.getElementById("status").textContent));
  check("...they read as blanks, not \"null\"", !/null/.test(phys));

  await fit(false);
  const txt2 = w.document.getElementById("elemBody").textContent;
  check("a determined orbit shows its value again", /2\.5\d*\s*au/.test(txt2));
  check("...and its sigma", /±/.test(txt2));
  check("...and no interval", !/…/.test(txt2));
  check("...and its exports are back", ["mpec", "elements", "mpcorb", "sof", "state", "covar"]
        .every((k) => !exportBtn(k).disabled));
  check("...and the 3D pane note is gone",
        w.document.getElementById("orbitSuppressNote").hidden);
  check("...and its class is shown again", /MBA/.test(w.document.getElementById("physBody").textContent));

  console.log(failures ? `${failures} FAILED` : "all passed");
  process.exit(failures ? 1 : 0);
})();
