// jsdom test: a ?object= deeplink answered from the DISK CACHE still shows the force model.
// A disk payload carries no session (and no restore token) until the viewer clicks; the Force
// model panel used to require a session, so it stayed hidden on its HTML default ("gravity")
// and 352870 read as gravity only while its orbit ran in +16 ast +(471) Papagena. The panel,
// the toggle, the "Beyond the 16" line and the auto-perturber banner now show for any payload
// that CAN become live (session, restore or refit); every control claims a session first.
//
//     NODE_PATH=…/node_modules node tests/web/test_deeplink_disk_panel.js
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

const BASE = {
  ok: true, object: "2016 RM33", combined: true, designations: ["2016 RM33", "2005 SA198"],
  mp_names: [], other_designations: [], force_model: "ASSIST(DE440+GR+16ast)", iod_method: "x",
  summary: { rms_residual: 0.27, rms_normalized: 0.6, n_obs: 8, n_obs_total: 8,
             n_excluded: 0, n_oppositions: 2, arc_days: 3000, obs_codes: 1, systematic: null },
  epoch: { jd: 2461000.5, iso: "2026-01-01T00:00:00" },
  nongrav: { model: null, params: {}, sigmas: {} },
  constraints: {}, asteroids: true, asteroid_extras: [], extra_perturbers: [],
  physical: { class: "MBA", pha: false, moid_earth_au: 1.0, tisserand_jupiter: 3.5,
              U: 1, H: 17, H_sigma: 0.1, G: 0.15, n_mags: 5, frame: "J2000" },
  elements: [], planetocentric: null,
  residuals: Array.from({ length: 8 }, (_, i) => ({
    i, jd_utc: 2460990.5 + i, obscode: "F51", designation: "2016 RM33",
    ra_deg: 10 + i, dec_deg: 5, mag: 20, band: "G", computed_mag: 20, dmag: 0,
    r_au: 2.5, delta_au: 1.6, d_ra_arcsec: 0.1, d_dec_arcsec: -0.1,
    sigma_ra: 0.5, sigma_dec: 0.5, resid_sigmas: 0.3, included: true,
    obs_id: null, trksub: null, ades_rms_ra: null, ades_rms_dec: null,
    mag_sigma: null, log_snr: null, obs80: " ".repeat(80),
  })),
  observatories: [{ obscode: "F51", name: "Pan-STARRS", n: 8, n_used: 8, median_resid: 0.1 }],
  history: [], orbit3d: { a: 2.37, e: 0.19, i: 0.4, node: 10, peri: 20, M: 30,
    epoch_jd: 2461000.5, n_deg_per_day: 0.27, q: 1.9, Q: 2.8, path: null, position: null },
  stage_residuals: [], session: "s1", meta: {},
};

const PAPAGENA = { number: 471, name: "Papagena", dchi2: 346.6, closest_au: 0.000955,
                   closest_date: "2021-05-10" };

// What the ephemeral disk tier hands a ?object= deeplink: the stored response, no session,
// no restore token -- only `refit`, the designation a click re-fits to become live.
function diskPayload(refit) {
  const p = JSON.parse(JSON.stringify(BASE));
  delete p.session;
  if (refit) p.refit = "352870";
  p.object = "352870";
  p.force_model = "ASSIST(DE440+GR+16ast+(471))";
  p.asteroids = true;
  p.asteroids_auto = true;
  p.asteroid_extras = [471];
  p.extra_perturbers = [PAPAGENA];
  return p;
}

async function openDeeplink(payload) {
  const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                        { url: "http://localhost/?object=352870", runScripts: "dangerously",
                          pretendToBeVisual: true });
  const w = dom.window;
  w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
  w.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
  w.requestAnimationFrame = (f) => setTimeout(f, 0);
  w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
    get: (t, k) => (k === "measureText" ? () => ({ width: 1 }) : () => {}) });
  w.fetch = async (url) => {
    const u = String(url);
    let body = { ok: true };
    if (u.startsWith("/api/fit_start"))
      body = { ok: true, job: null, state: "done", done: true, cached: "disk", result: payload };
    return { ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body) };
  };
  const s = w.document.createElement("script");
  s.textContent = appjs;
  w.document.body.appendChild(s);
  await new Promise((r) => setTimeout(r, 400));
  return w.document;
}

let fails = 0;
function check(cond, what) {
  if (!cond) { fails++; console.log("FAIL:", what); } else console.log("ok:", what);
}

(async () => {
  let d = await openDeeplink(diskPayload(true));
  check(!d.getElementById("workspace").hidden, "the deeplinked fit renders");
  check(!d.getElementById("ngPanel").hidden, "Force model panel shown without a session");
  const act = [...d.querySelectorAll("#astToggle .seg.active")].map((e) => e.dataset.ast);
  check(act.length === 1 && act[0] === "1", `toggle on the perturber field, not gravity (${act})`);
  const lbl = d.querySelector("#astToggle button[data-ast='1']").textContent;
  check(lbl.includes("(471)"), `toggle names the extra body (${lbl})`);
  const ex = d.getElementById("astExtras");
  check(!ex.hidden && /Papagena/.test(ex.textContent), "\"Beyond the 16\" line names Papagena");
  // the flash host only (document.body would also match the injected app.js source text)
  const banners = [...d.querySelectorAll("#flashHost .flash-msg")].map((e) => e.textContent);
  check(banners.some((t) => /beyond the 16 deflects this orbit/.test(t)),
        "the auto-perturber banner fires");

  // a payload that can never become live (no session, restore or refit) keeps the panel hidden
  d = await openDeeplink(diskPayload(false));
  check(d.getElementById("ngPanel").hidden, "no live route -> panel stays hidden");

  console.log(fails ? `${fails} FAILED` : "all passed");
  process.exit(fails ? 1 : 0);
})();
