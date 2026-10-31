// jsdom smoke test for the web UI's step controls: loads index.html + app.js with a
// stubbed fetch, runs a fit, and checks the three solve buttons (labels, visibility
// transitions incl. the obsAdded flag, exact POST payloads) AND the f/m keyboard
// shortcuts -- the f-key handler once broke silently when its target function was
// renamed (a ReferenceError inside a DOM listener that no python test can see).
//
//     npm i jsdom && node tests/web/dom_smoke.test.js
//
// Skips (exit 0) when jsdom is not installed -- it is a dev dependency, not shipped.
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

// Minimal fit payload shaped like serialise_fit's output.
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

// Strip the page's own <script src> tags (jsdom won't fetch them; we inject app.js
// ourselves AFTER stubbing) and run with scripts enabled so the injection executes.
const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                      { url: "http://localhost/", runScripts: "dangerously",
                        pretendToBeVisual: true });
const w = dom.window;
w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
w.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
w.requestAnimationFrame = (f) => setTimeout(f, 0);
w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
  get: (t, k) => (k === "measureText" ? () => ({ width: 1 }) : () => {}) });
w.fetch = async (url, opts) => {
  w.__lastFetch = { url: String(url), body: opts && opts.body ? JSON.parse(opts.body) : null };
  if (String(url).startsWith("/api/export/")) {         // text products -> .text()
    return { text: async () => "EXPORTED-TEXT for " + String(url) };
  }
  const p = JSON.parse(JSON.stringify(PAYLOAD));
  if (String(url).startsWith("/api/reauto")) p.session = "s2";
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

const scr = w.document.createElement("script");
scr.textContent = appjs;
w.document.body.appendChild(scr);

(async () => {
  const $ = (id) => w.document.getElementById(id);
  const assert = (c, m) => { if (!c) { console.error("FAIL:", m); process.exit(1); } };

  // three buttons exist with the right glyphs
  assert($("fullStepBtn") && $("fullStepBtn").textContent.includes("→"), "Full step has →");
  assert($("multiStepBtn") && $("multiStepBtn").textContent.includes("↻"), "Multi-step has ↻");
  assert($("reAutoBtn") && $("reAutoBtn").textContent.includes("🪄"), "Re-auto-solve has 🪄");

  // run a fit -> step buttons visible, reAuto hidden (nothing added)
  $("desig").value = "TEST 1";
  await w.eval("runFit()");
  await new Promise((r) => setTimeout(r, 20));
  assert(!$("fullStepBtn").hidden, "Full step visible after fit");
  assert(!$("multiStepBtn").hidden, "Multi-step visible after fit");
  assert($("reAutoBtn").hidden, "Re-auto-solve hidden before any paste-add");

  // ---- the ADES photometric band is a DEFAULT column (no "addl info" toggle needed) ----
  // It sits between mag and Δmag, so the mag a row shows can be read in its own filter.
  const ths = [...$("obsScroll").querySelectorAll("table.obs thead th")].map((t) => t.textContent);
  assert(ths.includes("band"), "band is a default obs column, got " + JSON.stringify(ths));
  const bandCol = ths.indexOf("band");
  assert(ths[bandCol - 1] === "mag" && ths[bandCol + 1] === "Δmag",
         "band sits between mag and Δmag, got " + JSON.stringify(ths));
  const cell = (row, k) => row.querySelectorAll("td")[k].textContent.trim();
  const trs = [...$("obsScroll").querySelectorAll("table.obs tbody tr")];
  assert(trs.length === 10 && cell(trs[0], bandCol) === "G",
         "band cell shows the reported band, got " + cell(trs[0], bandCol));
  // An OBS80 obs with no band carries the single-space sentinel (obs80 col 71 / ades.py),
  // which must read as an em-dash, not a blank cell.
  w.eval('CURRENT.residuals[0].band = " "; CURRENT.residuals[1].band = null; renderObs(CURRENT)');
  const trs2 = [...$("obsScroll").querySelectorAll("table.obs tbody tr")];
  assert(cell(trs2[0], bandCol) === "—" && cell(trs2[1], bandCol) === "—",
         "empty/null band renders as an em-dash, got " + cell(trs2[0], bandCol));
  w.eval('CURRENT.residuals[0].band = "G"; CURRENT.residuals[1].band = "G"; renderObs(CURRENT)');

  // Full step click posts single:true; Multi-step posts single:false
  await w.eval("runStep(true)");
  await new Promise((r) => setTimeout(r, 20));
  assert(w.__lastFetch.url === "/api/fullstep" && w.__lastFetch.body.single === true,
         "Full step posts single:true, got " + JSON.stringify(w.__lastFetch));
  await w.eval("runStep(false)");
  await new Promise((r) => setTimeout(r, 20));
  assert(w.__lastFetch.body.single === false, "Multi-step posts single:false");

  // simulate a successful add-to-fit paste -> reAuto becomes visible
  w.eval("obsAdded = true; renderObs(CURRENT)");
  assert(!$("reAutoBtn").hidden, "Re-auto-solve visible after obs added");

  // clicking it posts to /api/reauto with the session, and stays visible after render
  await w.eval("runReAuto()");
  await new Promise((r) => setTimeout(r, 20));
  assert(w.__lastFetch.url === "/api/reauto" && w.__lastFetch.body.session === "s1",
         "Re-auto posts session");
  assert(!$("reAutoBtn").hidden, "Re-auto-solve still visible after re-solve");

  // a fresh fetch resets the flag -> hidden again
  await w.eval("runFit()");
  await new Promise((r) => setTimeout(r, 20));
  assert($("reAutoBtn").hidden, "Re-auto-solve hidden again after a fresh fetch");

  // keyboard shortcuts: f = ONE Full step (single:true), m = Multi-step (single:false).
  // Uncaught ReferenceErrors in jsdom listeners surface as jsdomError -- trap them.
  let vcErr = null;
  dom.virtualConsole.on("jsdomError", (e) => { vcErr = e; });
  const press = (key) => w.document.dispatchEvent(
    new w.KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }));
  w.__lastFetch = null;
  press("f");
  await new Promise((r) => setTimeout(r, 20));
  assert(!vcErr, "f-key handler threw: " + (vcErr && vcErr.detail));
  assert(w.__lastFetch && w.__lastFetch.url === "/api/fullstep"
         && w.__lastFetch.body.single === true,
         "f key runs a single Full step, got " + JSON.stringify(w.__lastFetch));
  w.__lastFetch = null;
  press("m");
  await new Promise((r) => setTimeout(r, 20));
  assert(!vcErr, "m-key handler threw: " + (vcErr && vcErr.detail));
  assert(w.__lastFetch && w.__lastFetch.body.single === false,
         "m key runs a Multi-step, got " + JSON.stringify(w.__lastFetch));
  // a modifier chord is the BROWSER's (Ctrl-F = find in page): no step, not consumed
  for (const mod of ["ctrlKey", "metaKey", "altKey"]) {
    w.__lastFetch = null;
    const ev = new w.KeyboardEvent("keydown", { key: "f", bubbles: true, cancelable: true, [mod]: true });
    w.document.dispatchEvent(ev);
    await new Promise((r) => setTimeout(r, 20));
    assert(w.__lastFetch === null, mod + "+f must not run a Full step");
    assert(!ev.defaultPrevented, mod + "+f must be left to the browser");
  }

  // ---- Export runs off the CURRENT interactive fit (session handle), copy/paste box ----
  // After the steps above CURRENT.session is "s1": exportUrl must carry it so the server
  // returns the on-screen orbit, NOT re-run a fresh autofit via desig=.
  assert(w.eval('exportUrl("mpcorb")') === "/api/export/mpcorb?session=s1",
         "exportUrl uses the live session, got " + w.eval('exportUrl("mpcorb")'));

  // clicking a text export (mpcorb) fetches that URL and fills the copy/paste overlay
  w.__lastFetch = null;
  $("exportMenu").querySelector('button[data-export="mpcorb"]').click();
  await new Promise((r) => setTimeout(r, 20));
  assert(w.__lastFetch && w.__lastFetch.url === "/api/export/mpcorb?session=s1",
         "mpcorb export fetches the session URL, got " + JSON.stringify(w.__lastFetch));
  assert(!$("exportResultOverlay").hidden, "export result overlay is shown");
  assert($("exportResultText").value.includes("EXPORTED-TEXT"),
         "export box holds the fetched text, got " + $("exportResultText").value);
  assert($("exportResultTitle").textContent === "MPCORB line", "export title set");

  // the state-vector export is a JSON text product: same copy/paste box, same handle
  w.__lastFetch = null;
  $("exportMenu").querySelector('button[data-export="state"]').click();
  await new Promise((r) => setTimeout(r, 20));
  assert(w.__lastFetch && w.__lastFetch.url === "/api/export/state?session=s1",
         "state export fetches the session URL, got " + JSON.stringify(w.__lastFetch));
  assert($("exportResultTitle").textContent === "State vector + covariance (JSON)",
         "state export title set, got " + $("exportResultTitle").textContent);

  // every menu entry that lands in the copy/paste box must have a title (else "Export")
  for (const b of $("exportMenu").querySelectorAll("button[data-export]")) {
    const k = b.dataset.export;
    if (k === "mpec") continue;                    // opens a tab, no box title
    assert(w.eval("EXPORT_TITLES[" + JSON.stringify(k) + "]"),
           "export kind '" + k + "' has no EXPORT_TITLES entry");
  }

  // pseudo-MPEC stays a rendered-HTML new tab (window.open), still session-scoped
  let opened = null;
  w.open = (u) => { opened = u; return null; };
  $("exportMenu").querySelector('button[data-export="mpec"]').click();
  await new Promise((r) => setTimeout(r, 20));
  assert(opened === "/api/export/mpec?session=s1",
         "mpec opens a session-scoped tab, got " + opened);

  // Radar (delay/Doppler) surfaces in the status bar when the fit carries it -- the web
  // equivalent of the CLI's "radar : N/M delay/Doppler measurements in the fit" line. A
  // radar-less fit shows no radar stat (CURRENT has none from the stub payload).
  assert(!$("status").innerHTML.includes("radar"), "no radar stat for a radar-less fit");
  w.eval("renderStatus(Object.assign({}, CURRENT, { radar: "
       + "{ applied: true, n: 19, n_used: 19, post_median_z: 0.4, note: null, rows: [] } }))");
  assert($("status").innerHTML.includes("radar") && $("status").innerHTML.includes("19"),
         "radar stat shows in status bar, got " + $("status").innerHTML);

  // ---- a one-off fit route (re-auto) that outlives its inline deadline must be
  // POLLED, not rendered as its pending job stub ----
  // src/pyroorb/web/fitqueue.py:run_bounded answers past _SYNC_FIT_DEADLINE with
  // {ok:true, pending:true, job:...} instead of a result. This used to be handed
  // straight to render(), showing the caller an empty stub as if it were the
  // re-solved fit.
  {
    let resultCalls = 0;
    w.fetch = ((f) => async (url, opts) => {
      const u = String(url);
      if (u.startsWith("/api/reauto")) {
        const body = { ok: true, pending: true, job: "pendingjob1", state: "running" };
        return { json: async () => body, text: async () => JSON.stringify(body) };
      }
      if (u.startsWith("/api/fit_result")) {
        resultCalls++;
        if (resultCalls < 2) {
          const body = { ok: true, job: "pendingjob1", state: "running", done: false };
          return { json: async () => body, text: async () => JSON.stringify(body) };
        }
        const done = { ok: true, job: "pendingjob1", state: "done", done: true,
                       result: Object.assign(JSON.parse(JSON.stringify(PAYLOAD)),
                                             { session: "s3" }) };
        return { json: async () => done, text: async () => JSON.stringify(done) };
      }
      return f(url, opts);
    })(w.fetch);

    await w.eval("runReAuto()");
    // let the poll loop's fast-path timers (FitJob POLL_FAST_MS = 600ms) run out
    await new Promise((r) => setTimeout(r, 1500));
    assert(w.eval("CURRENT.session") === "s3",
           "a pending re-auto job stub was polled to its real result, got session "
           + w.eval("CURRENT.session"));
    assert(resultCalls >= 2, "re-auto polled /api/fit_result at least twice, got "
           + resultCalls);
  }

  console.log("DOM smoke test: all assertions passed");
})().catch((e) => { console.error("FAIL:", e); process.exit(1); });
