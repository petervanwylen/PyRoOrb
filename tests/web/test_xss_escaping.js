// jsdom regression test: nothing the SERVER or the URL supplies may reach the
// DOM as markup.
//
// The bug this pins: `?object=` is a deliberately shareable deep link
// (app.js pushObjectUrl / fitFromUrl), and its value was interpolated straight
// into `$("msg").innerHTML`, so a crafted link ran script on the app's origin
// -- which, since a session handle is the API's only credential, meant reading
// and driving whatever the victim had open. The same pattern applied to every
// server-sent `error` / `hint` / `banner` string.
//
// The assertion is deliberately DOM-shaped rather than "did script run": a
// payload that is correctly escaped exists only as TEXT, so the element it
// tried to create must not be in the tree at all. That holds regardless of
// whether jsdom would have executed the handler.
//
//     npm i jsdom && node tests/web/test_xss_escaping.js
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

// Two shapes, because they exercise different escapes: an element with an event
// handler, and one that closes an attribute first.
const PAYLOAD = '<img src=x onerror="window.__pwned=1">';
const ATTR_PAYLOAD = '"><img src=x onerror="window.__pwned=1">';

// A successful fit, so the panes that need a live CURRENT (the nongrav pane
// bails without one) can be exercised. Same shape as dom_smoke.test.js.
const GOOD = {
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
  stage_residuals: [], session: "s1-testhandle", meta: {},
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

let nextBody = {};                    // what the stubbed API answers with
w.fetch = async (url) => {
  if (String(url).startsWith("/api/export/")) return { text: async () => "" };
  const body = typeof nextBody === "function" ? nextBody(String(url)) : nextBody;
  return { json: async () => body, text: async () => JSON.stringify(body) };
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


const scr = w.document.createElement("script");
scr.textContent = appjs;
w.document.body.appendChild(scr);

let failures = 0;
function check(name, cond) {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}`);
  if (!cond) failures++;
}

// The core assertion: the payload survived as TEXT and created no element.
function assertInert(where, name, expectText = true) {
  const el = typeof where === "string" ? w.document.getElementById(where) : where;
  if (!el) { check(`${name} (element present)`, false); return; }
  const injected = el.querySelectorAll("img, script, svg, iframe").length;
  check(`${name}: no element injected`, injected === 0);
  if (expectText) {
    check(`${name}: payload rendered as literal text`,
          el.textContent.includes("<img") || el.textContent.includes("img src=x"));
  }
}

(async () => {
  await new Promise((r) => setTimeout(r, 0));

  // 1. The shareable deep link. fitFromUrl() copies ?object= into #desig and
  //    fits it; runFit writes the designation into #msg before the fetch even
  //    resolves, so the pending-state line is the one that mattered.
  nextBody = { ok: false, error: "no such object" };
  w.document.getElementById("desig").value = PAYLOAD;
  await w.eval("runFit()");
  await new Promise((r) => setTimeout(r, 20));
  assertInert("msg", "?object= designation echoed by runFit");

  // 2. A server-sent error string (+ its hint sibling).
  nextBody = { ok: false, error: PAYLOAD, hint: ATTR_PAYLOAD };
  w.document.getElementById("desig").value = "TEST 1";
  await w.eval("runFit()");
  await new Promise((r) => setTimeout(r, 20));
  assertInert("msg", "server error + hint on a failed fit");

  // 3. The pluto banner takes a different branch of the same handler.
  nextBody = { ok: false, pluto_banned: true, banner: PAYLOAD, error: "" };
  await w.eval("runFit()");
  await new Promise((r) => setTimeout(r, 20));
  assertInert("msg", "pluto banner");

  // 4. The nongrav pane keeps its own error surface -- but it bails without a
  //    live CURRENT, so land a real fit first (which also proves the escaping
  //    did not break the ordinary success path).
  nextBody = GOOD;
  w.document.getElementById("desig").value = "TEST 1";
  await w.eval("runFit()");
  await new Promise((r) => setTimeout(r, 20));
  check("a normal fit still renders (escaping broke nothing)",
        w.eval("CURRENT && CURRENT.session") === "s1-testhandle");
  check("normal fit clears the message line", w.document.getElementById("msg").textContent === "");

  nextBody = { ok: false, error: PAYLOAD };
  await w.eval('runNongrav("a1a2")');
  await new Promise((r) => setTimeout(r, 20));
  assertInert("ngBody", "nongrav pane error");

  // 5. Nothing anywhere in the document may have been created by a payload,
  //    and no handler may have fired.
  check("no payload element anywhere in the document",
        w.document.querySelectorAll('img[src="x"]').length === 0);
  check("no injected handler ran (window.__pwned unset)", w.__pwned === undefined);

  console.log(failures ? `\n${failures} FAILURE(S)` : "\nall XSS escaping checks passed");
  process.exit(failures ? 1 : 0);
})();
