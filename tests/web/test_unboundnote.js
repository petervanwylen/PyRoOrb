// jsdom test for the unbound-orbit notes (unboundnote.js): a genuinely unbound orbit (kept
// "unbound") shows NOTHING; a bound orbit shown although an unbound one fits says why (barely
// changes the residuals / too few nights); a planetary-reach reminder shows the engine's text;
// a payload without the block hides the note.
//
// The module listens on the "pyroorb:render" event app.js's render() dispatches, so the test
// drives REAL fits through a stubbed fetch (runFit()) rather than simulating the hook.
//
//     npm i jsdom && node tests/web/test_unboundnote.js
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
const ubjs = fs.readFileSync(path.join(ROOT, "unboundnote.js"), "utf8");

// Minimal fit payload shaped like serialise_fit's output (see test_dedupnote.js).
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

let unbound;                          // payload.unbound the next fit reports (undefined = absent)
w.fetch = async (url) => {
  if (String(url).startsWith("/api/export/")) return { text: async () => "" };
  const p = JSON.parse(JSON.stringify(PAYLOAD));
  if (unbound !== undefined) p.unbound = unbound;
  return { json: async () => p };
};

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

w.fetch = ((f) => async (...a) => {
  const r = await f(...a);
  return r.text ? r : { ...r, text: async () => JSON.stringify(await r.json()) };
})(w.fetch);

for (const src of [appjs, ubjs]) {
  const scr = w.document.createElement("script");
  scr.textContent = src;
  w.document.body.appendChild(scr);
}

let failures = 0;
function check(name, cond) {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}`);
  if (!cond) failures++;
}

async function fit(rec) {
  unbound = rec;
  w.document.getElementById("desig").value = "TEST 1";
  await w.eval("runFit()");
  await new Promise((r) => setTimeout(r, 20));
}

const CHECK = { e_free: 1.00023, rms_free: 0.235, e_bound: 0.9999, rms_bound: 0.252,
                nights: 5, arc_days: 12.0, arc_ok: true, damaged: false, kept: "bound" };

(async () => {
  await new Promise((r) => setTimeout(r, 0));
  check("no note before any payload", !w.document.getElementById("unboundNote"));

  await fit({ check: { ...CHECK, kept: "unbound", damaged: true }, reminder: null });
  const box = w.document.getElementById("unboundNote");
  if (!box) {
    console.log('  FAIL  no note element after a fit -- did render() stop dispatching "pyroorb:render"?');
    process.exit(1);
  }
  check("an orbit kept unbound (1I/2I/3I) says nothing", box.hidden);

  await fit({ check: CHECK, reminder: null });
  check("bound shown although unbound fits: note shown", !box.hidden);
  check("...says it fits about as well", /fits about as well/.test(box.textContent));
  check("...carries both rms values", /0\.235/.test(box.textContent) && /0\.252/.test(box.textContent));
  check("is informational, not severe", !box.classList.contains("severe"));
  check("...points at Multi-step for the lowest-rms orbit",
        /Multi-step/.test(box.textContent) && /lowest-rms/.test(box.textContent));
  check("...warns lowest rms can overfit", /fit noise/.test(box.textContent));
  check("...says a systematic that vanishes favours unbound",
        /systematic trend/.test(box.textContent));

  // The σ and its bar are BOTH the engine's (check.sigma / check.sigma_bar), and the pair
  // is comparable by construction: the engine converts the evidence into the scale its own
  // rule is written in, so "N σ, short of the 3σ" can never contradict itself. It could:
  // while the note wrote the bar as a literal 3 and the engine reported the raw scaled Δχ²
  // it decided on an F-quantile of, a two-tracklet arc read "only 9.0σ, short of the 3σ it
  // takes". Hence a case where the sentence would have been absurd, and no literal bar.
  await fit({ check: { ...CHECK, sigma: 1.7, sigma_bar: 3.0, dchi2_raw: 2.89, station_nights: 7 },
              reminder: null });
  check("with a measured significance: says how many sigma, against the engine's bar",
        /1\.7σ/.test(box.textContent) && /3σ it takes/.test(box.textContent));
  check("...and the sigma it reports is below the bar it names",
        !/9\.0σ/.test(box.textContent));
  check("...names the independent looks the evidence rests on",
        /on 7 independent station-nights of astrometry/.test(box.textContent));

  // A bar the engine moved is rendered as the engine moved it, never as a literal 3.
  await fit({ check: { ...CHECK, sigma: 1.7, sigma_bar: 4.2, station_nights: 1 }, reminder: null });
  check("a different bar is the one shown", /4\.2σ it takes/.test(box.textContent));
  check("...one station-night is said in the singular",
        /on 1 independent station-night of/.test(box.textContent));

  // An unbound orbit the engine could not EARN: believed only because the data refuse
  // the one bound orbit the comparator found, which on a long sparse arc is a local walk. The
  // free fit is shown, and the note says what was and was not tried.
  await fit({ check: { ...CHECK, kept: "unbound-unsearched", damaged: true, e_free: 38.4018,
                       rms_free: 0.436, rms_bound: 2.36, nights: 3, arc_days: 3131.9, arc_ok: false },
              reminder: null });
  check("an unearned unbound orbit (walk-only comparator) gets a note", !box.hidden);
  check("...that names the local walk and the unsearched alternative",
        /local walk/.test(box.textContent) && /not searched for/.test(box.textContent));
  check("...carries the free e and both rms values",
        /38\.4018/.test(box.textContent) && /0\.436/.test(box.textContent) && /2\.36/.test(box.textContent));
  check("...and the night count over the arc", /3 nights over 3131\.9 days/.test(box.textContent));
  check("...is informational, not severe", !box.classList.contains("severe"));
  await fit({ check: { ...CHECK, kept: "unbound-no-bound-fit", damaged: true, e_free: 1.0055,
                       rms_free: 10.33, nights: 3, arc_days: 6232.2, arc_ok: false }, reminder: null });
  check("an unbound orbit with no bound fit at all gets its own sentence",
        !box.hidden && /no orbit with e < 1 could be fitted/.test(box.textContent));

  await fit({ check: { ...CHECK, damaged: true, arc_ok: false, nights: 3, arc_days: 2.0 },
              reminder: null });
  check("too few nights: says so", /3 nights/.test(box.textContent) && /wrong/.test(box.textContent));
  check("too few nights: also explains Multi-step", /Multi-step/.test(box.textContent));

  await fit({ check: null, reminder: { text: "Unbound in its present osculating elements (e = 1.05700)" } });
  check("reminder shows the engine's text", !box.hidden && /osculating elements/.test(box.textContent));
  check("reminder head says not necessarily interstellar",
        /not necessarily interstellar/.test(box.textContent));

  await fit(null);
  check("hidden for a null block", box.hidden);

  await fit(undefined);
  check("hidden for a payload without the block (legacy)", box.hidden);

  console.log(failures ? `${failures} FAILED` : "all passed");
  process.exit(failures ? 1 : 0);
})();
