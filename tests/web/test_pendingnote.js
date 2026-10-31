// jsdom test for the pending-linkage verdict note (pendingnote.js): after an ITF
// search filed its find as a POSSIBLE, a Full step whose observation set has GROWN
// and now passes the search's gate raises the "Submit linkage" button -- and this
// module says why, once per verdict.  A step on an unchanged set says nothing (the
// search's own banner already did), and neither does an ident-kind block.
//
// The module listens on the "pyroorb:render" event app.js's render() dispatches,
// so the test drives REAL fits through a stubbed fetch (runFit()) rather than
// simulating the hook -- if render() ever stops dispatching, this fails.
//
//     npm i jsdom && node tests/web/test_pendingnote.js
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
const pnjs = fs.readFileSync(path.join(ROOT, "pendingnote.js"), "utf8");

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

let itfBlock;                        // p.itf the next fit reports
w.fetch = async (url) => {
  if (String(url).startsWith("/api/export/")) return { text: async () => "" };
  const p = JSON.parse(JSON.stringify(PAYLOAD));
  if (itfBlock !== undefined) p.itf = itfBlock;
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

for (const src of [appjs, pnjs]) {
  const scr = w.document.createElement("script");
  scr.textContent = src;
  w.document.body.appendChild(scr);
}

let failures = 0;
function check(name, cond) {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}`);
  if (!cond) failures++;
}

function banners() {
  const host = w.document.getElementById("flashHost");
  return host ? [...host.querySelectorAll(".flash-msg")].map((e) => e.textContent) : [];
}
function clearBanners() {
  const host = w.document.getElementById("flashHost");
  if (host) host.innerHTML = "";
}

async function fit(block) {
  itfBlock = block;
  w.document.getElementById("desig").value = "TEST 1";
  await w.eval("runFit()");
  await new Promise((r) => setTimeout(r, 20));
}

const GROWN_OK = { found: true, pending: true, submittable: true, possible: false,
  submit_enabled: true, outcome: "linked", tracklets: [], added_obs: 2, linkage: {},
  warnings: [], diagnostics: [],
  reason: "pending linkage: 1 observation(s) added or switched on since the search, and the "
        + "grown set passes the search's own submittability gate" };
const GROWN_NO = { ...GROWN_OK, submittable: false, possible: true, linkage: null,
  reason: "1 observation(s) added or switched on since the search, but "
        + "the set still fails the search's submittability gate: opposition profile "
        + "not submittable: 1.5 opposition(s), 4 night(s)" };
const SAME = { ...GROWN_NO,
  reason: "the search filed this as a possible (one lone new night), and "
        + "re-solving the same observations does not change that" };
// a SWAP: an observation the search judged switched off while another was switched on, which
// is a CHANGED set and not a grown one
const SWAPPED = { ...GROWN_NO,
  reason: "the search filed this as a possible (one lone new night); 1 observation(s) it "
        + "judged are no longer in the fit, and a set it has not judged is not one it can "
        + "be overruled on" };
// the profile-only 1.75 possible, after the user re-solved it (engine_api._pending_verdict)
const PROMOTED = { ...GROWN_OK,
  reason: "the search filed this as a possible on its opposition profile alone (two new oppositions of one "
        + "night each); re-solved after a step, it is offered for submission at your judgement" };

(async () => {
  await new Promise((r) => setTimeout(r, 0));    // let jsdom settle

  await fit(SAME);
  check("a step on the unchanged set: no button", w.document.getElementById("itfSubmitWrap").hidden);
  check("...and no banner from this module", !banners().some((t) => /Linkage (now|still)/.test(t)));

  clearBanners();
  await fit(GROWN_NO);
  check("grown but still refused: no button", w.document.getElementById("itfSubmitWrap").hidden);
  check("...says it is still not submittable and why",
        banners().some((t) => /Linkage still not submittable/.test(t) && /1\.5 opposition/.test(t)));

  clearBanners();
  await fit(GROWN_NO);
  check("the same verdict again: not repeated", !banners().some((t) => /Linkage still/.test(t)));

  clearBanners();
  await fit(SWAPPED);
  check("a swapped set: no button", w.document.getElementById("itfSubmitWrap").hidden);
  check("...and the refusal the user's own edit provoked names itself",
        banners().some((t) => /Linkage still not submittable/.test(t)
                              && /no longer in the fit/.test(t)));

  clearBanners();
  await fit(GROWN_OK);
  check("grown and passing: the button appears", !w.document.getElementById("itfSubmitWrap").hidden);
  check("...with a banner saying why", banners().some((t) => /Linkage now submittable/.test(t)));

  clearBanners();
  await fit(PROMOTED);
  check("a profile-only possible re-solved after a step: the button appears",
        !w.document.getElementById("itfSubmitWrap").hidden);
  check("...with a banner saying it is offered at the user's judgement",
        banners().some((t) => /Linkage now submittable/.test(t) && /your judgement/.test(t)));

  clearBanners();
  await fit({ ...GROWN_OK, kind: "ident" });
  check("an ident-kind block is not this module's", !banners().some((t) => /Linkage now/.test(t)));

  console.log(failures ? `${failures} FAILED` : "all passed");
  process.exit(failures ? 1 : 0);
})();
