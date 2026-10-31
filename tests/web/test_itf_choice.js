// jsdom test for the two-result choice (itfchoice.js + the runItfSearch hook in
// app.js). An ITF search can answer with a submittable linkage AND a fuller
// unconfirmed one (`itf.alternative`); the page must ask which to show, show
// exactly the chosen one, and never ask when there is only one result.
//
//     npm i jsdom && node tests/web/test_itf_choice.js
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
const choicejs = fs.readFileSync(path.join(ROOT, "itfchoice.js"), "utf8");

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

let itfResponse = null;              // set per scenario
let submitResponse = null;
w.confirm = () => w.__confirmReturn !== false;   // default: user accepts
w.fetch = async (url, opts) => {
  // result polls don't count as "the last request the UI made" -- the
  // assertions below are about the POST that started the search
  if (!String(url).startsWith("/api/itf_result")) {
    w.__lastFetch = { url: String(url),
                      body: opts && opts.body ? JSON.parse(opts.body) : null };
  }
  if (String(url).startsWith("/api/itf_submit")) {
    return { json: async () => JSON.parse(JSON.stringify(submitResponse)) };
  }
  // The search runs on a worker thread and the client polls for it: the POST
  // answers with a job handle at once (nothing waits on a request long enough
  // for a hosted proxy to kill it) and /api/itf_result hands over the payload.
  if (String(url).startsWith("/api/itf_search")) {
    w.__itfJobPolls = 0;
    return { json: async () => ({ ok: true, job: "trTEST" }) };
  }
  if (String(url).startsWith("/api/itf_result")) {
    // one "still running" answer first, so the poll loop itself is exercised
    const n = ++w.__itfJobPolls;
    return { json: async () => (n < 2
      ? { ok: true, done: false, elapsed: 3.0 }
      : { ok: true, done: true, elapsed: 7.0,
          result: JSON.parse(JSON.stringify(itfResponse)) }) };
  }
  if (String(url).startsWith("/api/itf_trace")) {
    (w.__traceFetches = w.__traceFetches || []).push(String(url));
    return { json: async () => ({ ok: true, lines: [], next: 0, done: true }) };
  }
  if (String(url).startsWith("/api/itf_defaults")) {
    // deliberately UNLIKE the index.html literals: the engine's value must win
    return { json: async () => ({ ok: true, max_uncert_arcsec: 54321,
                                  uncert_mult: 3.7, mag_limit: 21.5,
                                  mag_limit_deep: 24.0, rank_mode: "ml",
                                  candidate_source: "scan", trace_level: 2 }) };
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

for (const src of [appjs, choicejs]) {
  const scr = w.document.createElement("script");
  scr.textContent = src;
  w.document.body.appendChild(scr);
}

(async () => {
  const $ = (id) => w.document.getElementById(id);
  const assert = (c, m) => { if (!c) { console.error("FAIL:", m); process.exit(1); } };
  const tick = () => new Promise((r) => setTimeout(r, 20));
  // the search polls for its result on a timer: wait until either the dialog
  // is up or the search has finished (node's own timer keeps the loop alive)
  const dialogOrDone = async (run) => {
    let finished = false;
    run.then(() => { finished = true; });
    for (let i = 0; i < 500; i++) {
      const ov = $("itfChoiceOverlay");
      if (ov && !ov.hidden) return true;
      if (finished) return false;
      await tick();
    }
    throw new Error("the search neither finished nor asked");
  };
  const click = (id) => $(id).dispatchEvent(new w.MouseEvent("click", { bubbles: true }));

  $("desig").value = "TEST 1";
  await w.eval("runFit()");
  await tick();

  const found = (session, submittable, trks) => {
    const p = JSON.parse(JSON.stringify(PAYLOAD));
    p.session = session;
    p.itf = { found: true, possible: !submittable, submittable, status: submittable ? 1 : 0.6,
              reason: "dry-run", added_obs: 4 * trks.length, itf_obs80: "",
              linkage: submittable ? { links: {} } : null, warnings: [],
              tracklets: trks.map((t) => ({ trksub: t, date: "2016 01 02", stn: "F51", n_obs: 4 })) };
    return p;
  };
  const pair = (completeLeads) => {
    const sub = found("sLinked", true, ["PR00001"]);
    const full = found("sFull", false, ["PR00001", "PR00002", "PR00003"]);
    const [p, alt] = completeLeads ? [full, sub] : [sub, full];
    p.itf.alternative = alt;
    return p;
  };
  const search = async (resp, pick) => {
    itfResponse = resp;
    const run = w.eval("runItfSearch()");
    const shown = await dialogOrDone(run);
    if (shown && pick === "escape") {
      w.document.dispatchEvent(new w.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    } else if (shown && pick) {
      click(pick === "complete" ? "itfChoiceComplete" : "itfChoiceSubmittable");
    }
    await run;
    await tick();
    return shown;
  };

  // two results, the complete one leading: the user is asked, and gets what they pick
  let asked = await search(pair(true), "complete");
  assert(asked, "two different results open the choice dialog");
  assert(w.eval("CURRENT.session") === "sFull", "picking complete shows the fuller result");
  assert(w.eval("CURRENT.itf.possible") === true, "the fuller result is the unconfirmed one");
  assert(w.eval("CURRENT.itf.alternative") === undefined, "the page holds one result, not the pair");
  assert(/POSSIBLE/.test($("flashHost").textContent), "and it is announced as possible");

  asked = await search(pair(true), "submittable");
  assert(asked && w.eval("CURRENT.session") === "sLinked",
         "picking submittable shows the linkage, whichever result the server led with");
  assert(w.eval("CURRENT.itf.submittable") === true, "the shown result is submittable");

  asked = await search(pair(false), "complete");
  assert(asked && w.eval("CURRENT.session") === "sFull",
         "the alternative is reachable when the linkage leads too");

  asked = await search(pair(true), "escape");
  assert(asked && w.eval("CURRENT.session") === "sLinked",
         "dismissing the dialog shows the submittable result");

  // the dialog names what each result links
  itfResponse = pair(true);
  const run = w.eval("runItfSearch()");
  assert(await dialogOrDone(run), "the dialog opens again");
  const txt = $("itfChoiceBody").textContent;
  assert(/PR00003/.test(txt) && /Most complete/.test(txt) && /Submittable/.test(txt),
         "both results are described: " + txt);
  click("itfChoiceComplete");
  await run;
  await tick();

  // one result: no dialog
  asked = await search(found("sOnly", true, ["PR00001"]), "complete");
  assert(!asked, "a single result never asks");
  assert(w.eval("CURRENT.session") === "sOnly", "the single result is shown");

  console.log("ITF choice test: all assertions passed");
})().catch((e) => { console.error("FAIL:", e); process.exit(1); });
