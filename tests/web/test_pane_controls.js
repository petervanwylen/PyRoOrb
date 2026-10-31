// jsdom test for the central-pane framework: each pane in #paneSelect owns exactly
// two things -- a control strip (.panectl[data-pane]) in the panel header and a body
// (.panebody[data-pane]) -- and switching panes shows that pair and NOTHING else.
//
// The bug this locks down: #obsControls carried an inline `display:flex`, which
// outranks the stylesheet's [hidden] rule, so the observation buttons (⤓ worst, Full
// step, MPC 80, ＋ addl info, ⧉ copy, worst first ...) sat over the ephemeris, the
// orbarium and the impact pane. The structural half is asserted too: no pane element
// may carry an inline display, and every <option> must have a strip and a body.
//
//     npm i jsdom && node tests/web/test_pane_controls.js
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
const css = fs.readFileSync(path.join(ROOT, "style.css"), "utf8");

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


// the real stylesheet, so [hidden] rules (and any inline display fighting them) count
const style = w.document.createElement("style");
style.textContent = css;
w.document.head.appendChild(style);

const scr = w.document.createElement("script");
scr.textContent = appjs;
w.document.body.appendChild(scr);

const tick = (ms = 20) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const $ = (id) => w.document.getElementById(id);
  const D = w.document;
  const assert = (c, m) => { if (!c) { console.error("FAIL:", m); process.exit(1); }
                             else console.log("  PASS ", m); };
  // jsdom's getComputedStyle is not layout-aware, so display:none does not inherit:
  // walk the ancestors ourselves, which is what "on screen" actually means here.
  const shown = (el) => {
    for (let n = el; n && n.nodeType === 1; n = n.parentElement)
      if (n.hidden || w.getComputedStyle(n).display === "none") return false;
    return true;
  };
  const panes = [...D.querySelectorAll("#paneSelect option")].map((o) => o.value);

  // ---- structure: every pane has a strip and a body, and no inline display ----
  assert(panes.length >= 4, "panes: " + panes.join(", "));
  for (const p of panes) {
    assert(D.querySelector(`.panectl[data-pane="${p}"]`), `${p} has a control strip`);
    assert(D.querySelector(`.panebody[data-pane="${p}"]`), `${p} has a body`);
  }
  for (const el of D.querySelectorAll(".panectl, .panebody, .paneowned")) {
    assert(!(el.style && el.style.display),
           `${el.id || el.className} carries no inline display (it would outrank [hidden])`);
    assert(panes.includes(el.dataset.pane),
           `${el.id || el.className} is tagged with a real pane: ${el.dataset.pane}`);
  }

  $("desig").value = "TEST 1";
  await w.eval("runFit()");
  await tick();

  // ---- switching: exactly one strip and one body visible, always ----
  for (const p of panes) {
    await w.eval(`setCentralPane('${p}')`);
    await tick();
    for (const el of D.querySelectorAll(".panectl")) {
      const own = el.dataset.pane === p;
      assert(shown(el) === own,
             `${p}: strip ${el.id} ${own ? "shows" : "hidden"}`);
    }
    for (const el of D.querySelectorAll(".panebody")) {
      const own = el.dataset.pane === p;
      assert(shown(el) === own, `${p}: body ${el.id} ${own ? "shows" : "hidden"}`);
    }
    // the named offenders, by button, not by container. On their own pane they must
    // still be reachable -- a strip that hides everywhere is not the fix either.
    for (const id of ["worstBtn", "mpc80Btn", "addlBtn", "worstFirst"]) {
      assert(shown($(id)) === (p === "obs"),
             `${p}: ${id} ${p === "obs" ? "is on screen" : "is not on screen"}`);
    }
    for (const id of ["copyObsBtn", "fullStepBtn", "multiStepBtn", "reAutoBtn"]) {
      // these three also hide on app state, so only the off-pane half is a rule
      assert(p === "obs" || !shown($(id)), `${p}: ${id} is not on screen`);
    }
    assert(shown($("ephemParamsBtn")) === (p === "ephem"),
           `${p}: ephemeris settings button ${p === "ephem" ? "is" : "is not"} on screen`);
    assert($("skyview").hidden === (p !== "ephem"),
           `${p}: on-sky panel follows the ephemeris pane`);
  }

  // ---- a state-driven pane-owned element can't strand itself either ----
  await w.eval("setCentralPane('obs')");
  await w.eval("paneStateShow('obsBanner', true)");
  assert(shown($("obsBanner")), "replay banner shows on its own pane");
  await w.eval("setCentralPane('ephem')");
  await tick();
  assert(!shown($("obsBanner")), "replay banner hides when the pane is switched away");
  await w.eval("setCentralPane('obs')");
  assert(shown($("obsBanner")), "and comes back with its pane, state intact");

  console.log("\nPASS: central-pane framework (one strip + one body per pane)");
})();
