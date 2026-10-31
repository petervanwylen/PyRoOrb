// jsdom test for the ITF-search UI: the toolbar button enables with a fit, the
// options modal opens/closes, "Search ITF" POSTs the session + options to
// /api/itf_search, a found result re-renders the workspace from the fresh
// payload, and a no-find shows the transient "no additional found" banner.
//
//     npm i jsdom && node tests/web/test_itf_ui.js
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
const confjs = fs.readFileSync(path.join(ROOT, "submitconfirm.js"), "utf8");

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

for (const src of [appjs, confjs]) {
  const scr = w.document.createElement("script");
  scr.textContent = src;
  w.document.body.appendChild(scr);
}

(async () => {
  const $ = (id) => w.document.getElementById(id);
  const assert = (c, m) => { if (!c) { console.error("FAIL:", m); process.exit(1); } };
  // The final confirmation is the submitconfirm.js dialog (not window.confirm):
  // run the submit, let it open, then click Submit / Cancel. See
  // tests/test_submit_confirm.js for what the dialog itself must show.
  // the remembered submitter identity (askIdentity reads it; nothing prompts)
  w.localStorage.setItem("itfLinkName", "J. Smith");
  w.localStorage.setItem("itfLinkEmail", "js@example.org");
  const submitVia = async (choice) => {
    const p = w.eval("runItfSubmit()");
    await new Promise((r) => setTimeout(r, 20));
    const btn = $(choice === "go" ? "submitConfirmGo" : "submitConfirmCancel");
    if (btn) btn.dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
    await p;
    await new Promise((r) => setTimeout(r, 20));
  };

  assert($("itfBtn"), "ITF button exists");
  assert($("itfBtn").disabled, "ITF button disabled before any fit");
  assert($("itfOverlay").hidden, "options modal starts hidden");

  $("desig").value = "TEST 1";
  await w.eval("runFit()");
  await new Promise((r) => setTimeout(r, 20));
  assert(!$("itfBtn").disabled, "ITF button enabled once an orbit is loaded");

  // open the modal, tweak an option, run a NOT-FOUND search
  $("itfBtn").click();
  assert(!$("itfOverlay").hidden, "modal opens on click");

  // The modal's numbers come from the ENGINE (GET /api/itf_defaults ->
  // ItfConfig), not from the index.html literals -- an engine default that
  // moves (uncert_mult 1.1 -> 2.5) must not leave the UI searching on the old
  // one, which is how the web search silently missed tracklets the CLI found.
  await new Promise((r) => setTimeout(r, 20));
  assert($("itfUncMult").value === "3.7",
         "corridor multiplier taken from the engine, not the HTML literal: "
         + $("itfUncMult").value);
  assert($("itfMaxUnc").value === "54321",
         "max uncertainty taken from the engine: " + $("itfMaxUnc").value);
  // the corridor's faint cut is an engine default too -- same trap, same rule --
  // but SHOWN, never filled in: an untyped field lets the engine deepen the
  // default per object, and a typed value (even the default) is used exactly
  assert($("itfMagLimit").placeholder === "default is 21.5",
         "faint-limit default shown from the engine, not the HTML literal: "
         + $("itfMagLimit").placeholder);
  assert($("itfMagLimit").value === "",
         "faint limit left empty so the per-object default applies: "
         + $("itfMagLimit").value);
  assert($("itfTraceLevel").value === "2",
         "trace detail taken from the engine default: " + $("itfTraceLevel").value);
  assert($("itfCandidateSource").value === "scan",
         "candidate source taken from the engine, not the HTML literal: "
         + $("itfCandidateSource").value);
  // The trace level is a LAUNCH option (the search reads it once, when it
  // starts), so it must live in this modal. It used to sit only in the trace
  // VIEWER, where picking it looked like it did nothing.
  assert($("itfOverlay").contains($("itfTraceLevel")),
         "the starting trace level is picked in the launch modal");
  $("itfRankMode").value = "certainty";
  $("itfMaxUnc").value = "12000";
  $("itfMagLimit").value = "22.8";
  $("itfTraceLevel").value = "3";
  itfResponse = { ok: true, itf: { found: false, status: 0, reason: "no likely matches" } };
  await w.eval("runItfSearch()");
  await new Promise((r) => setTimeout(r, 20));
  assert($("itfOverlay").hidden, "modal closes when the search starts");
  const b = w.__lastFetch;
  assert(b.url === "/api/itf_search", "posted to /api/itf_search");
  assert(b.body.session === "s1", "session posted");
  assert(Array.isArray(b.body.mask) && b.body.mask.length === 10, "mask posted");
  assert(b.body.options.rank_mode === "certainty" && b.body.options.max_uncert_arcsec === 12000
         && b.body.options.candidate_source === "scan" && b.body.options.jitter === false,
         "options posted: " + JSON.stringify(b.body.options));
  assert(b.body.options.mag_limit === 22.8,
         "faint-limit override posted: " + JSON.stringify(b.body.options));
  assert(b.body.options.trace_level === 3,
         "the chosen trace level reaches the search: " + JSON.stringify(b.body.options));
  assert($("itfTraceLevelLive").value === "3",
         "the viewer's live control starts where the search started: "
         + $("itfTraceLevelLive").value);
  // the search must be BACKGROUNDED and polled: a request held open for the
  // length of a real search is what a hosted proxy kills, answering the client
  // with a non-JSON body ("upstream error")
  assert(b.body.async === true, "the search is started async, not awaited");
  assert(w.__itfJobPolls >= 2, "the result was polled out of /api/itf_result: "
         + w.__itfJobPolls);
  const host = $("flashHost");
  assert(host && /no additional observations found/.test(host.textContent),
         "transient 'no additional found' banner shown");

  // A DISK-CACHE HIT carries no session and no restore token, only `refit` (the
  // designation). The search must claim a session for it (ensureLive re-runs the
  // fit) and go -- it used to return silently, so the button "did nothing"
  // (2020 BB167 on pyroorb.net).
  {
    const cached = JSON.parse(JSON.stringify(PAYLOAD));
    delete cached.session;
    cached.refit = "TEST 1";
    await w.eval(`render(${JSON.stringify(cached)}); CURRENT = ${JSON.stringify(cached)};`);
    assert(!w.eval("CURRENT.session"), "the cached payload starts without a session");
    w.__lastFetch = null;
    itfResponse = { ok: true, itf: { found: false, status: 0, reason: "no likely matches" } };
    await w.eval("runItfSearch()");
    await new Promise((r) => setTimeout(r, 20));
    const c = w.__lastFetch;
    assert(c && c.url === "/api/itf_search",
           "a refit-only (disk-cached) fit still starts the search: " + JSON.stringify(c));
    assert(c.body.session === "s1", "the claimed session is posted: " + c.body.session);
    assert(!$("itfBtn").disabled, "the button is re-enabled after the search");
  }

  // WITHHELD: a tracklet linked and the search declined to show it. That is
  // not "nothing found" -- it gets its own, cautionary banner carrying the
  // clause that refused (engine_api sets itf.suppressed).
  itfResponse = { ok: true, itf: { found: false, suppressed: true, status: 0.6,
    reason: "nothing shown: the tracklet that linked is almost certainly one "
      + "of many that would fit this orbit just as well" } };
  await w.eval("runItfSearch()");
  await new Promise((r) => setTimeout(r, 20));
  const withheld = $("flashHost").textContent;
  assert(/linked but is not being shown/.test(withheld),
         "withheld candidate gets its own banner: " + withheld);
  assert(/one of many/.test(withheld), "the refusing clause is shown: " + withheld);
  assert(!/nothing shown:/.test(withheld),
         "the engine's prefix is dropped, the banner says it already: " + withheld);

  // FOUND: fresh payload replaces the view; banner names the tracklets
  itfResponse = JSON.parse(JSON.stringify(PAYLOAD));
  itfResponse.session = "s2";
  itfResponse.object = "TEST 1 extended";
  itfResponse.itf = { found: true, status: 1, reason: "dry-run", added_obs: 4,
                      tracklets: [{ trksub: "PR00001", date: "2016 01 02",
                                    stn: "F51", n_obs: 4 }],
                      itf_obs80: "", linkage: { links: {} }, submittable: true,
                      submit_enabled: true };
  await w.eval("runItfSearch()");
  await new Promise((r) => setTimeout(r, 20));
  assert(w.eval("CURRENT.session") === "s2", "found result replaced the session");
  assert(/PR00001/.test($("flashHost").textContent), "banner names the linked tracklet");
  assert(!$("workspace").hidden, "workspace visible after extended fit render");

  // Submit-linkage button: visible after a find, hidden otherwise
  assert($("itfSubmitBtn"), "submit button element exists");
  assert(!$("itfSubmitWrap").hidden, "Submit linkage visible after a find");
  assert($("itfDirectBtn").hidden,
         "no direct-to-MPC choice unless the server sets DIRECT_SUBMIT_ALLOWED");
  assert(!$("itfDownloadBtn").hidden && !$("itfIdentityBtn").hidden,
         "the menu always offers the file and the identity");
  {
    const allowed = JSON.parse(JSON.stringify(w.eval("CURRENT")));
    allowed.itf.direct_submit_allowed = true;
    await w.eval(`render(${JSON.stringify(allowed)})`);
    assert(!$("itfDirectBtn").hidden, "DIRECT_SUBMIT_ALLOWED adds the direct-to-MPC choice");
    allowed.itf.direct_submit_allowed = false;
    await w.eval(`render(${JSON.stringify(allowed)})`);
  }

  // The gate: toggling observations makes the displayed orbit stale, so
  // "Submit linkage" must go unclickable (with a tooltip saying why) until a
  // Full step re-solves on the current set -- you cannot submit a linkage the
  // orbit on screen has not been fitted to.
  assert(!$("itfSubmitBtn").disabled, "submit clickable on a freshly solved fit");
  w.eval("workingMask[0] = !workingMask[0]; renderObs(CURRENT);");
  assert($("itfSubmitBtn").disabled,
         "submit disabled once an observation is toggled");
  assert(/Full step/.test($("itfSubmitBtn").title),
         "tooltip tells the user to run a Full step: " + $("itfSubmitBtn").title);
  // ...and a re-render (what a Full step returns) clears it again
  await w.eval("render(CURRENT)");
  assert(!$("itfSubmitBtn").disabled, "a solve re-enables submit");
  assert(!/Full step/.test($("itfSubmitBtn").title),
         "...and drops the stale-solve tooltip (back to the button's own)");

  // decline the confirmation -> no POST
  const before = w.__lastFetch.url;
  await submitVia("cancel");
  assert(w.__lastFetch.url === before, "declining confirm does not POST");

  // accept -> POSTs the session + the remembered identity to the QUEUE route;
  // a taken submission hides the button (one-shot)
  submitResponse = { ok: true, submitted: true, accepted: true, status: "queued",
                     route: "queue", queued: { document: 17, token: "t",
                     status_url: "https://www.pyroorb.net/identifications/?document=17&token=t",
                     dashboard: "https://www.pyroorb.net/identifications/dashboard" } };
  await submitVia("go");
  assert(w.__lastFetch.url === "/api/itf_submit" && w.__lastFetch.body.session === "s2",
         "submit POSTs the session to /api/itf_submit");
  assert(w.__lastFetch.body.route === "queue", "the main button submits to the queue");
  assert(w.__lastFetch.body.name === "J. Smith" && w.__lastFetch.body.email === "js@example.org",
         "the header is the remembered name / e-mail, never a built-in one");
  assert($("itfSubmitWrap").hidden, "submit button hidden after a 200 (one-shot)");
  assert(/queued as submission 17/.test($("flashHost").textContent), "success banner shown");

  // A multi-designation fit (no ITF search anywhere) offers its identification:
  // the payload's ident linkage block must raise the same Submit button, say why,
  // and the confirm must name every id being claimed as one object.
  const identPayload = JSON.parse(JSON.stringify(PAYLOAD));
  identPayload.session = "s3";
  identPayload.object = "1999 TV43";
  identPayload.designations = ["1999 TV43", "OOn0024"];
  identPayload.itf = {
    found: true, kind: "ident", pending: true, status: 1.0, clean: true,
    reason: "clean identification", neocp: true, added_obs: 0,
    designations: ["J99T43V"],
    tracklets: [{ trksub: "OOn0024", date: "2026 07 19", stn: "O18", n_obs: 3 }],
    linkage: { links: { link_0: { designations: ["J99T43V"],
                                  trksubs: [["OOn0024", "2026 07 19.770914", "O18"]],
                                  identification_type: "neocp" } } },
    submittable: true, submit_enabled: true, warnings: [], vetting: {},
  };
  await w.eval(`render(${JSON.stringify(identPayload)})`);
  w.eval(`CURRENT = ${JSON.stringify(identPayload)}`);
  assert(!$("itfSubmitWrap").hidden,
         "Submit linkage offered for a multi-designation fit, with no ITF search");
  assert(/one object|identification/i.test($("flashHost").textContent),
         "a banner explains the offered identification: " + $("flashHost").textContent);

  const p = w.eval("runItfSubmit()");                        // open, read, then decline
  await new Promise((r) => setTimeout(r, 20));
  const asked = $("submitConfirmBody").textContent;
  assert(/J99T43V/.test(asked) && /OOn0024/.test(asked),
         "the confirmation names every id being claimed: " + asked);
  assert(/NEOCP/.test(asked), "an NEOCP identification says so: " + asked);
  $("submitConfirmCancel").dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await p;

  // A 'y2' -- an identification whose join requires EXCEPTING a tracklet -- is shown for review
  // but must NOT raise a submit button: the identification document cannot carry the exception,
  // so sending it would claim the excepted tracklet as this object. It goes to the MPC as a Jira
  // ticket, and the UI says so in ONE banner (no headline plus a caution repeating it, and no
  // "review before submitting" pointing at a button that isn't there).
  w.eval('_identFlashKey = null; $("flashHost").innerHTML = "";');
  const y2Payload = JSON.parse(JSON.stringify(identPayload));
  y2Payload.session = "s4";
  y2Payload.itf = Object.assign({}, identPayload.itf, {
    clean: false, submittable: false, warnings: [],
    reason: "an identification WITH EXCEPTIONS -- the join holds only if the 6-obs 2021-05-09 "
      + "F51 tracklet of 2018 VL78 is not this object. Check that tracklet in the record before "
      + "claiming anything. NOT submittable here: an identification document has no way to say "
      + "'except these observations', so an identification with exceptions goes to the MPC as a "
      + "Jira ticket.",
  });
  await w.eval(`render(${JSON.stringify(y2Payload)})`);
  const y2Text = $("flashHost").textContent;
  assert($("itfSubmitWrap").hidden, "a y2 raises no Submit linkage button");
  assert($("itfSubmitWrap").hidden, "...nor the download-the-document alternative");
  assert(/Jira ticket/.test(y2Text), "the banner says where it DOES go: " + y2Text);
  assert(!/Review before submitting/.test(y2Text),
         "...and does not point at a button that is not there: " + y2Text);
  assert($("flashHost").children.length === 1,
         `ONE banner, not a headline plus a caution repeating it (got ${$("flashHost").children.length})`);

  // A 'y-examine' -- ambiguous but real, and the ONE non-clean verdict that stays
  // submittable, so it is the one that gets the " Review before submitting..." tail.
  // Its reason is written as a PHRASE to be read after a lead-in and carries no full
  // stop of its own, so the two sentences ran together in every such banner:
  // "...retain >=95% of their obs Review before submitting the identification."
  w.eval('_identFlashKey = null; $("flashHost").innerHTML = "";');
  const yxPayload = JSON.parse(JSON.stringify(identPayload));
  yxPayload.session = "s5";
  yxPayload.itf = Object.assign({}, identPayload.itf, {
    clean: false, submittable: true, warnings: [],
    reason: "identification candidate: both arcs retain >=95% of their obs",
  });
  await w.eval(`render(${JSON.stringify(yxPayload)})`);
  const yxText = $("flashHost").textContent;
  assert(/Review before submitting/.test(yxText),
         "a y-examine IS submittable, so it keeps the review tail: " + yxText);
  assert(/obs\. Review before submitting/.test(yxText),
         "the reason ends as a sentence before the next one starts: " + yxText);
  assert(!/obs Review/.test(yxText), "no run-on: " + yxText);
  // ...and a reason that already punctuates itself must not collect a second stop
  w.eval('_identFlashKey = null; $("flashHost").innerHTML = "";');
  const yxDot = JSON.parse(JSON.stringify(yxPayload));
  yxDot.session = "s6";
  yxDot.itf = Object.assign({}, yxPayload.itf,
                            { reason: "an ambiguous link worth a human look." });
  await w.eval(`render(${JSON.stringify(yxDot)})`);
  assert(!/\.\./.test($("flashHost").textContent),
         "no doubled full stop: " + $("flashHost").textContent);

  // Escape closes the modal
  $("itfBtn").click();
  assert(!$("itfOverlay").hidden, "modal reopens");
  w.document.dispatchEvent(new w.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  assert($("itfOverlay").hidden, "Escape closes the ITF modal");

  // The viewer's "Detail" control re-levels the RUNNING search: it rides out on
  // the feed poll (&level=), which is what makes it something other than a
  // decoration -- before, it was read only at launch and moving it here did
  // nothing at all.
  w.__traceFetches = [];
  w.eval('openItfTrace("trTEST")');
  await new Promise((r) => setTimeout(r, 20));
  assert((w.__traceFetches[0] || "").includes("level=3"),
         "the feed poll carries the level: " + w.__traceFetches[0]);
  $("itfTraceLevelLive").value = "4";
  $("itfTraceLevelLive").dispatchEvent(new w.Event("change", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 20));
  assert(w.__traceFetches.some((u) => u.includes("level=4")),
         "changing Detail pushes the new level at once: "
         + JSON.stringify(w.__traceFetches));

  console.log("ITF UI test: all assertions passed");
})().catch((e) => { console.error("FAIL:", e); process.exit(1); });
