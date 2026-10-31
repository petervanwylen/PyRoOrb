// jsdom test for the Google Analytics events (static/analytics.js + its call sites in
// app.js / randomobj.js): each USER action sends its one event through gtag("event", ...),
// a pane switch or force-model change made by CODE (a deep link, a re-fit) sends none,
// and a gtag that throws cannot break the page. The other jsdom tests never load
// analytics.js, which is the "analytics absent" half: they must keep passing untouched.
//
//     npm i jsdom && node tests/web/test_analytics_events.js
//
// Skips (exit 0) when jsdom is not installed.
const fs = require("fs");
const path = require("path");
let JSDOM;
try { ({ JSDOM } = require("jsdom")); }
catch { console.log("SKIP: jsdom not installed (npm i jsdom to run this test)"); process.exit(0); }

const ROOT = path.join(__dirname, "..", "..", "src", "pyroorb", "web", "static");
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8");
const appjs = ["analytics.js", "fitjob.js", "app.js"]
  .map((f) => fs.readFileSync(path.join(ROOT, f), "utf8")).join("\n");

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



const events = [];
w.gtag = (kind, name, params) => { if (kind === "event") events.push({ name, params }); };

const scr = w.document.createElement("script");
scr.textContent = appjs;
w.document.body.appendChild(scr);

const tick = (ms = 20) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const $ = (id) => w.document.getElementById(id);
  const assert = (c, m) => { if (!c) { console.error("FAIL:", m); process.exit(1); }
                             else console.log("  PASS ", m); };
  const sent = (name) => events.filter((e) => e.name === name);
  const change = (el) => el.dispatchEvent(new w.Event("change", { bubbles: true }));

  $("desig").value = "TEST 1";
  await w.eval("runFit()");
  await tick();
  assert(sent("fit_object").length === 1 &&
         sent("fit_object")[0].params.source === "designation", "a typed fit sends fit_object");

  // code switching panes (deep link / re-fit path) is not a user choice
  events.length = 0;
  await w.eval("setCentralPane('orbarium'); setCentralPane('obs')");
  assert(!events.some((e) => e.name.startsWith("switch_to_")),
         "setCentralPane from code sends no switch event");

  $("paneSelect").value = "orbarium"; change($("paneSelect"));
  await tick();
  assert(sent("switch_to_orbarium_mode").length === 1 &&
         sent("switch_to_orbarium_mode")[0].params.pane === "orbarium",
         "picking Orbarium sends switch_to_orbarium_mode");
  $("paneSelect").value = "obs"; change($("paneSelect"));
  assert(sent("switch_to_obs_mode").length === 1, "and back sends switch_to_obs_mode");

  $("exportBtn").disabled = false;
  $("exportMenu").querySelector("button[data-export='elements']").click();
  await tick();
  assert(sent("export").length === 1 && sent("export")[0].params.export_type === "elements",
         "an Export menu product sends export {export_type}");

  // force model: the user's dropdown and toggle send it ...
  const sel = $("ngSelect");
  sel.innerHTML = '<option value="none">none</option><option value="yarkovsky">yarkovsky</option>';
  sel.value = "yarkovsky"; change(sel);
  await tick();
  const ng = sent("change_force_model");
  assert(ng.length === 1 && ng[0].params.control === "nongrav" &&
         ng[0].params.model === "yarkovsky", "the non-grav dropdown sends change_force_model");
  $("astToggle").querySelector("button[data-ast='1']").click();
  await tick();
  const ast = sent("change_force_model");
  assert(ast.length === 2 && ast[1].params.model === "16ast",
         "the +16 ast toggle sends change_force_model");
  // ... the engine calls a re-fit makes do not
  events.length = 0;
  await w.eval("runNongrav('none'); setAsteroids(false)");
  await tick();
  assert(sent("change_force_model").length === 0,
         "runNongrav / setAsteroids called by code send nothing");

  $("fsBtn").click();
  assert(sent("fullscreen_orbit").length === 1, "the orbit fullscreen button sends fullscreen_orbit");

  $("pasteText").value = "line one\n\nline two\n";
  $("pasteFit").click();
  await tick();
  const pa = sent("paste_observations");
  assert(pa.length === 1 && pa[0].params.action === "fit" && pa[0].params.n_lines === 2,
         "Fit pasted observations sends paste_observations {action: fit, n_lines}");

  // analytics may never break the page
  w.gtag = () => { throw new Error("blocked"); };
  let threw = false;
  try { w.eval("gaEvent('x', {})"); } catch (e) { threw = true; }
  assert(!threw, "a throwing gtag is swallowed");

  console.log("\nPASS: analytics events");
  process.exit(0);
})();
