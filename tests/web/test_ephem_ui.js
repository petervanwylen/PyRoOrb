// jsdom test for the ephemeris display cuts in the web UI: the options modal carries a
// faint-limit box (placeholder from the ENGINE's default, not an HTML literal) and a
// "suppress unobservables" checkbox that disables itself for a site with no horizon;
// both ride out on /api/ephem and on the text export, and what they hid is shown above
// the table rather than silently leaving it short.
//
//     npm i jsdom && node tests/web/test_ephem_ui.js
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

const EPHEM_ROW = {
  jd_utc: 2461000.5, date_iso: "2026-01-01T00:00:00", ra_deg: 10, dec_deg: 5,
  ra_hms: "00 40 00.00", dec_dms: "+05 00 00.0", delta_au: 1.5, r_au: 2.4,
  elong_deg: 150, phase_deg: 10, v_mag: 19, motion_arcsec_min: 0.5, motion_pa_deg: 90,
  unc_smaj_arcsec: 1, unc_smin_arcsec: 0.5, unc_pa_deg: 45,
  alt_deg: 60, az_deg: 180, sun_alt_deg: -30,
};
let ephemResponse = { ok: true, designation: "TEST 1", site: "500", rows: [EPHEM_ROW],
                      hidden: {}, hidden_note: "", site_has_horizon: true };

const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                      { url: "http://localhost/", runScripts: "dangerously",
                        pretendToBeVisual: true });
const w = dom.window;
w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
w.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
w.requestAnimationFrame = (f) => setTimeout(f, 0);
w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
  get: (t, k) => (k === "measureText" ? () => ({ width: 1 }) : () => {}) });

const seen = [];                      // every URL the app fetched, in order
w.fetch = async (url, opts) => {
  const u = String(url);
  seen.push(u);
  w.__lastFetch = { url: u, body: opts && opts.body ? JSON.parse(opts.body) : null };
  if (u.startsWith("/api/ephem_site")) {
    return { json: async () => ({ ok: true, site_has_horizon: !/site=500/.test(u) }) };
  }
  if (u.startsWith("/api/ephem")) {
    return { json: async () => JSON.parse(JSON.stringify(ephemResponse)) };
  }
  if (u.startsWith("/api/export/ephem")) {
    return { text: async () => "Ephemeris  (observatory F51)" };
  }
  return { json: async () => JSON.parse(JSON.stringify(PAYLOAD)) };
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

const scr = w.document.createElement("script");
scr.textContent = appjs;
w.document.body.appendChild(scr);

const tick = (ms = 20) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const $ = (id) => w.document.getElementById(id);
  const assert = (c, m) => { if (!c) { console.error("FAIL:", m); process.exit(1); } };
  const lastEphemUrl = () => [...seen].reverse().find((u) => u.startsWith("/api/ephem?"));

  assert($("ephFaint"), "faint-limit input exists");
  assert($("ephUnobs"), "suppress-unobservables checkbox exists");

  $("desig").value = "TEST 1";
  await w.eval("runFit()");
  await tick();

  // ---- pane defaults: no cuts, so the fetch carries neither param -------------
  await w.eval("setCentralPane('ephem')");
  await tick();
  let u = lastEphemUrl();
  assert(u && !/faint=/.test(u) && !/unobs=/.test(u),
         "no cuts by default -> plain ephemeris request: " + u);

  // ---- the modal: engine-owned faint default, site-gated checkbox -------------
  $("ephemParamsBtn").click();
  await tick();
  assert(!$("ephemOverlay").hidden, "options modal opens");
  assert($("ephFaint").value === "", "faint limit starts blank (= no limit)");
  // the box quotes no default at all -- no find_orb number to go stale in the UI
  assert(!/22|find_orb/.test($("ephFaint").placeholder + $("ephFaint").title),
         "faint box carries no default reference: " + $("ephFaint").placeholder);
  assert(!seen.some((s) => s.startsWith("/api/ephem_defaults")),
         "no defaults endpoint is consulted");
  assert($("ephUnobs").disabled, "suppression disabled for the geocentre (no horizon)");
  assert(/no horizon/.test($("ephUnobsHint").textContent),
         "hint says why: " + $("ephUnobsHint").textContent);

  // a real ground station re-enables it
  $("ephSite").value = "F51";
  $("ephSite").dispatchEvent(new w.Event("change", { bubbles: true }));
  await tick();
  assert(!$("ephUnobs").disabled, "suppression enabled for a site with a horizon");

  // ---- both cuts ride out on the pane fetch -----------------------------------
  $("ephFaint").value = "21.5";
  $("ephUnobs").checked = true;
  ephemResponse = { ...ephemResponse, site: "F51",
                    rows: [EPHEM_ROW,
                           { ...EPHEM_ROW, date_iso: "2026-01-03T00:00:00",
                             hidden_before: { n: 2, why: "2 too faint" } },
                           { ...EPHEM_ROW, date_iso: "2026-01-05T00:00:00",
                             hidden_before: { n: 3, why: "3 below the horizon" },
                             hidden_after: { n: 1, why: "1 in daylight" } }],
                    hidden: { faint: 2, horizon: 3, daylight: 1 },
                    hidden_note: "6 epoch(s) hidden: 2 too faint, 3 below the horizon, 1 in daylight" };
  $("ephGo").click();
  await tick();
  u = lastEphemUrl();
  assert(/faint=21\.5/.test(u) && /unobs=1/.test(u) && /site=F51/.test(u),
         "cuts sent to /api/ephem: " + u);
  assert(/V≤21\.5/.test($("ephemRangeLabel").textContent)
         && /observable only/.test($("ephemRangeLabel").textContent),
         "range label states the active cuts: " + $("ephemRangeLabel").textContent);
  assert(/6 epoch\(s\) hidden/.test($("ephemWrap").textContent),
         "what was hidden is shown above the table");
  assert($("ephemWrap").querySelector("table"), "the surviving rows still render");
  assert(/>alt</.test($("ephemWrap").innerHTML),
         "altitude columns appear once suppression supplies them (a kept row IS up)");

  // ---- a visual break wherever a run of epochs was cut ------------------------
  const breaks = [...$("ephemWrap").querySelectorAll("tr.eph-break")];
  assert(breaks.length === 3, "one break per suppressed run (2 interior + 1 trailing), got "
         + breaks.length);
  assert(/2 hidden — 2 too faint/.test(breaks[0].textContent), breaks[0].textContent);
  assert(/3 hidden — 3 below the horizon/.test(breaks[1].textContent), breaks[1].textContent);
  assert(/1 hidden — 1 in daylight/.test(breaks[2].textContent), breaks[2].textContent);
  const dataCols = $("ephemWrap").querySelector("tr[data-i]").children.length;
  assert(Number(breaks[0].firstElementChild.getAttribute("colspan")) === dataCols,
         "break spans the whole table width: " + dataCols);
  const order = [...$("ephemWrap").querySelectorAll("tbody tr")]
    .map((tr) => (tr.classList.contains("eph-break") ? "B" : "R")).join("");
  assert(order === "RBRBRB", "breaks sit between the rows they separate: " + order);

  // break rows must not disturb row selection (it keys on data-i, not position)
  $("ephemWrap").querySelectorAll("tr[data-i]")[2]
    .dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await tick();
  const sel = [...$("ephemWrap").querySelectorAll("tr.sel")];
  assert(sel.length === 1 && sel[0].dataset.i === "2",
         "clicking the 3rd data row selects THAT row across the breaks: "
         + sel.map((t) => t.dataset.i));

  // ---- a cut that hides EVERYTHING explains itself rather than reading as failure
  ephemResponse = { ...ephemResponse, rows: [],
                    hidden_note: "8 epoch(s) hidden: 8 below the horizon" };
  await w.eval("loadEphem()");
  await tick();
  assert(/8 epoch\(s\) hidden/.test($("ephemWrap").textContent),
         "empty table says why it is empty: " + $("ephemWrap").textContent);

  // ---- a site with no horizon + suppression on: say so, don't pretend ---------
  ephemResponse = { ...ephemResponse, rows: [EPHEM_ROW], hidden: {}, hidden_note: "",
                    site_has_horizon: false };
  await w.eval("loadEphem()");
  await tick();
  assert(/no horizon/.test($("ephemWrap").textContent),
         "no-horizon site notes that nothing could be suppressed: " + $("ephemWrap").textContent);

  // ---- the text EXPORT carries the same two cuts ------------------------------
  $("exportBtn").click();
  w.document.querySelector('#exportMenu button[data-export="ephem"]')
    .dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await tick();
  $("ephFaint").value = "20";
  $("ephUnobs").checked = true;
  $("ephGo").click();
  await tick();
  const ex = [...seen].reverse().find((s) => s.startsWith("/api/export/ephem"));
  assert(ex && /faint=20/.test(ex) && /unobs=1/.test(ex),
         "export request carries the same cuts: " + ex);

  console.log("PASS: ephemeris faint limit + suppress-unobservables UI");
})();
