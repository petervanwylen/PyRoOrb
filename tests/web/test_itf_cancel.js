// jsdom test for stopping a running ITF search (itfcancel.js + the runItfSearch
// hooks in app.js). While a search runs, "stop search" sits next to "see trace"
// and the trace viewer shows a "Stop search" button; either POSTs
// /api/itf_cancel with the running job's handle, and the stopped search ends as
// "stopped" -- an info banner, never "failed".
//
//     npm i jsdom && node tests/web/test_itf_cancel.js
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
const canceljs = fs.readFileSync(path.join(ROOT, "itfcancel.js"), "utf8");

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

// server model: the search runs until /api/itf_cancel arrives, then answers
// "stopping" once and "cancelled" after that
let job = null, cancelBodies = [], stopPolls = 0;
const respond = (body) => ({ json: async () => body,
                             text: async () => JSON.stringify(body) });
w.fetch = async (url, opts) => {
  const u = String(url);
  if (u.startsWith("/api/session")) return respond({ ok: true, session: "s1" });
  if (u.startsWith("/api/fit_start")) {
    return respond({ ok: true, job: "job1", state: "done", done: true,
                     result: JSON.parse(JSON.stringify(PAYLOAD)) });
  }
  if (u.startsWith("/api/itf_search")) {
    job = JSON.parse(opts.body).options.trace_id;
    return respond({ ok: true, job });
  }
  if (u.startsWith("/api/itf_cancel")) {
    cancelBodies.push(JSON.parse(opts.body));
    return respond({ ok: true, done: false, cancelling: true });
  }
  if (u.startsWith("/api/itf_result")) {
    if (!cancelBodies.length) return respond({ ok: true, done: false, elapsed: 1 });
    if (stopPolls++ < 1) {
      return respond({ ok: true, done: false, elapsed: 2, cancelling: true });
    }
    return respond({ ok: true, done: true, elapsed: 2,
                     result: { ok: false, cancelled: true,
                               error: "the search was stopped at your request" } });
  }
  if (u.startsWith("/api/itf_trace")) {
    return respond({ ok: true, lines: [], next: 0, done: false });
  }
  if (u.startsWith("/api/itf_defaults")) {
    return respond({ ok: true, max_uncert_arcsec: 144000, uncert_mult: 2.5,
                     mag_limit: 23.5, rank_mode: "ml", candidate_source: "index",
                     trace_level: 2 });
  }
  return respond(JSON.parse(JSON.stringify(PAYLOAD)));
};

const banners = [];
for (const src of [appjs, canceljs]) {
  const scr = w.document.createElement("script");
  scr.textContent = src;
  w.document.body.appendChild(scr);
}
const realFlash = w.flashBanner;
w.flashBanner = (msg, kind) => { banners.push({ msg, kind }); };
w.eval("flashBanner = window.flashBanner");

(async () => {
  const $ = (id) => w.document.getElementById(id);
  const assert = (c, m) => { if (!c) { console.error("FAIL:", m); process.exit(1); } };
  const tick = (ms = 20) => new Promise((r) => setTimeout(r, ms));

  $("desig").value = "TEST 1";
  await w.eval("runFit()");
  await tick();
  assert(realFlash, "app.js defines flashBanner");

  const run = w.eval("runItfSearch()");
  let finished = false;
  run.then(() => { finished = true; });
  await tick(300);

  assert(job && job.startsWith("tr"), "the search started with a trace-id job");
  assert($("itfStopLink"), "a 'stop search' link sits beside 'see trace'");
  assert($("itfBtn").disabled, "the search button stays busy while running");

  // the trace viewer offers Stop only for the RUNNING search's feed (opened the
  // way the "see trace" link opens it)
  w.eval(`openItfTrace(${JSON.stringify(job)})`);
  await tick();
  assert(!$("itfTraceStop").hidden, "trace viewer shows 'Stop search' while running");
  w.eval(`openItfTrace("trSomeOldSearch")`);
  await tick();
  assert($("itfTraceStop").hidden, "no Stop on an older search's feed");
  w.eval(`openItfTrace(${JSON.stringify(job)})`);
  await tick();

  $("itfStopLink").dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await tick();
  assert(cancelBodies.length === 1 && cancelBodies[0].job === job,
         "Stop POSTs /api/itf_cancel with the running job: " + JSON.stringify(cancelBodies));
  assert($("itfTraceStop").disabled, "the viewer's button is disabled while stopping");
  // a second click (link hidden; the viewer button disabled) sends nothing more
  $("itfTraceStop").dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await tick();
  assert(cancelBodies.length === 1, "a double stop sends one cancel");

  for (let i = 0; i < 100 && !finished; i++) await tick(50);
  assert(finished, "the search ends once the server says it stopped");
  assert(/stopped/i.test($("msg").textContent), "msg says stopped: " + $("msg").textContent);
  assert(!banners.some((b) => b.kind === "alert"),
         "a user's stop is not reported as a failure: " + JSON.stringify(banners));
  assert(banners.some((b) => b.kind === "info" && /stopped/i.test(b.msg)),
         "an info banner says it stopped: " + JSON.stringify(banners));
  assert(!$("itfBtn").disabled, "the search button is usable again");
  assert($("itfTraceStop").hidden, "no Stop button once nothing is running");
  console.log("PASS: test_itf_cancel.js");
  process.exit(0);
})().catch((e) => { console.error("FAIL:", e); process.exit(1); });
