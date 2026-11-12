// jsdom test: an orbit integrated with asteroid perturbers beyond the 16 says so on the fit
// page -- the Force model panel carries a standing line naming each extra body (with the
// screen's dchi2 and closest approach) and the +16 toggle reads "+16 ast +(19)"; an orbit with
// none shows neither. Drives REAL fits through app.js with a stubbed fetch (see test_dof.js).
//
//     NODE_PATH=…/node_modules node tests/web/test_extra_perturbers.js
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
const FORTUNA = { number: 19, name: "Fortuna", dchi2: 1178.5, closest_au: 0.00107,
                  closest_date: "2011-04-13" };

const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                      { url: "http://localhost/", runScripts: "dangerously",
                        pretendToBeVisual: true });
const w = dom.window;
w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
w.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
w.requestAnimationFrame = (f) => setTimeout(f, 0);
w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
  get: (t, k) => (k === "measureText" ? () => ({ width: 1 }) : () => {}) });

const BAMBERGA = { number: 324, name: "Bamberga", dchi2: 1402, closest_au: 0.0000612,
                   closest_date: "2005-10-29" };
let extras = [];
let nFit = 0;
let allCount = 0, allOn = false;
const posted = [];
w.fetch = async (url, opts) => {
  const u = String(url);
  const p = JSON.parse(JSON.stringify(BASE));
  p.perturbers_all_count = allCount;
  p.perturbers_all = allOn;
  p.extra_perturbers = extras;
  p.asteroids_auto = extras.length > 0;
  p.session = `s${nFit}`;                 // the banner fires once per session
  p.asteroid_extras = extras.map((e) => e.number);
  if (extras.length) p.force_model = "ASSIST(DE440+GR+16ast+(19))";
  let body = p;
  if (u.startsWith("/api/asteroids")) {
    const req = JSON.parse(opts.body);
    posted.push(req.on);
    allOn = req.on === "all";
    p.perturbers_all = allOn;
    p.asteroids = req.on !== false;
    p.asteroids_auto = false;
  }
  if (u.startsWith("/api/session")) body = { ok: true, session: "s1" };
  else if (u.startsWith("/api/fit_start")) body = { ok: true, job: "j", state: "done", done: true, result: p };
  else if (u.startsWith("/api/export/")) return { text: async () => "", json: async () => ({}) };
  return { json: async () => body, text: async () => JSON.stringify(body) };
};
const scr = w.document.createElement("script");
scr.textContent = appjs;
w.document.body.appendChild(scr);

let failures = 0;
function check(name, cond) {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}`);
  if (!cond) failures++;
}

async function fit(list) {
  extras = list;
  nFit += 1;
  w.document.getElementById("desig").value = "2016 RM33 = 2005 SA198";
  await w.eval("runFit()");
  await new Promise((r) => setTimeout(r, 20));
}

(async () => {
  await new Promise((r) => setTimeout(r, 0));
  await fit([FORTUNA]);
  const note = w.document.getElementById("astExtras");
  const btn = w.document.querySelector("#astToggle button[data-ast='1']");
  check("the extra-perturber line is shown", note && !note.hidden);
  check("it names the body", /\(19\) Fortuna/.test(note.textContent));
  check("it gives the date and distance", /2011-04-13/.test(note.textContent)
        && /0\.0011/.test(note.textContent));
  check("the toggle reads +16 ast +(19)", btn.textContent.trim() === "+16 ast +⁠(19)");
  check("no break possible inside +(19)", !/\+\(/.test(btn.textContent));

  await fit([]);
  check("no extra perturbers: line hidden", note.hidden);
  check("no extra perturbers: toggle reads +16 ast", btn.textContent.trim() === "+16 ast");

  // the banner: singular for one body, plural for several, the total counted
  const banners = () => [...w.document.querySelectorAll(".flash .flash-msg")].map((m) => m.textContent);
  await fit([BAMBERGA]);
  let b = banners().filter((t) => /beyond the 16/.test(t)).pop() || "";
  check("one body: singular", /^An asteroid perturber beyond the 16 deflects this orbit/.test(b));
  check("one body: 17 perturbers in all", /includes 17 asteroidal perturbers \(the 16 plus this one\)/.test(b));
  check("a 0.00006 au pass is not rounded to 0.0001", /passed 0\.000061 au/.test(b)
        && /0\.000061 au/.test(note.textContent));
  check("one body: the line says 'need it'", /need it\./.test(note.textContent));
  await fit([FORTUNA, BAMBERGA]);
  b = banners().filter((t) => /beyond the 16/.test(t)).pop() || "";
  check("two bodies: plural", /^2 asteroid perturbers beyond the 16 deflect this orbit/.test(b));
  check("two bodies: 18 perturbers in all", /includes 18 asteroidal perturbers \(the 16 plus these\)/.test(b));
  check("two bodies: the line says 'need them'", /need them\./.test(note.textContent));
  // the "all" segment: every body of the perturber kernel, screened or not
  const allBtn = w.document.querySelector("#astToggle button[data-ast='all']");
  allCount = 0;
  await fit([]);
  check("no kernel: the all toggle is hidden", allBtn.hidden);
  allCount = 343;
  await fit([]);
  check("with the kernel: the all toggle reads '+343 ast'",
        !allBtn.hidden && allBtn.textContent.trim() === "+343 ast");
  check("the 16 is the active segment", btn.classList.contains("active")
        && !allBtn.classList.contains("active"));
  allBtn.click();
  await new Promise((r) => setTimeout(r, 30));
  check("clicking it asks the server for on: 'all'", posted[posted.length - 1] === "all");
  check("the all segment is then active", allBtn.classList.contains("active")
        && !btn.classList.contains("active"));
  check("the line says every body is in the field", !note.hidden
        && /All 343 massive asteroids/.test(note.textContent));
  btn.click();
  await new Promise((r) => setTimeout(r, 30));
  check("back to the 16: on true", posted[posted.length - 1] === true
        && btn.classList.contains("active"));
  if (failures) { console.log(`FAIL: ${failures} check(s)`); process.exit(1); }
  console.log("PASS: extra perturbers shown on the fit page");
})();
