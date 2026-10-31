// Headless (jsdom) DOM test of the natural-satellite badge + heliocentric<->planetocentric
// element toggle. No arm64 headless browser, so we drive app.js under jsdom.
//   Run:  npm i jsdom  &&  node tests/web/test_planetocentric_ui.js
// Headless DOM test of the satellite badge + heliocentric⇄planetocentric element toggle.
const fs = require("fs");
let JSDOM;
try { ({ JSDOM } = require("jsdom")); }
catch (e) { console.log("SKIP: jsdom not installed (run `npm i jsdom` in this dir to enable)"); process.exit(0); }
const ROOT = require("path").join(__dirname, "..", "..", "src", "pyroorb", "web", "static");
const html = fs.readFileSync(ROOT + "/index.html", "utf8");
const appjs = fs.readFileSync(ROOT + "/app.js", "utf8");

const dom = new JSDOM(html, { runScripts: "dangerously", pretendToBeVisual: true, url: "http://localhost/" });
const { window } = dom;
// stubs for app.js load-time code
window.fetch = () => Promise.resolve({ ok: true, json: () => Promise.resolve({ models: [], elements: [] }) });
window.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
window.requestAnimationFrame = () => 0;
window.cancelAnimationFrame = () => {};
if (!window.matchMedia) window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });

// inject app.js, then a driver script (shares global lexical scope -> can see CURRENT/renderElements)
function run(js) {
  const s = window.document.createElement("script");
  s.textContent = js;
  window.document.body.appendChild(s);
}
try { run(appjs); } catch (e) { console.log("LOAD ERROR:", e.message); process.exit(1); }

const satPayload = {
  object: "SYN01", combined: false, mp_names: [],
  summary: { rms_residual: 0.012, rms_normalized: 0.5, n_obs: 16, n_obs_total: 16, arc_days: 150, n_oppositions: 1 },
  physical: { pha: false },
  epoch: { jd: 2460075.5, iso: "2023-06-15T00:00:00" }, session: null, constraint: null,
  elements: [ { name: "a", label: "semi-major axis", unit: "au", value: 5.20, sigma: 0.001 },
              { name: "e", label: "eccentricity", unit: "", value: 0.05, sigma: 0.001 },
              { name: "i", label: "inclination", unit: "deg", value: 1.3, sigma: 0.01 } ],
  planetocentric: {
    center: "jupiter", center_label: "Jupiter", class: "prograde satellite of Jupiter",
    bound: true, range_au: 0.0866, frame: "J2000 ecliptic (Jupiter system barycentre)", validated: true,
    elements: [ { name: "a", label: "semi-major axis", unit: "au", value: 0.0767, sigma: 0.0001 },
                { name: "e", label: "eccentricity", unit: "", value: 0.158, sigma: 0.001 },
                { name: "i", label: "inclination", unit: "deg", value: 28.98, sigma: 0.01 } ],
  },
};
const mbaPayload = JSON.parse(JSON.stringify(satPayload));
mbaPayload.object = "2023 QJ3"; mbaPayload.planetocentric = null;
mbaPayload.elements = [ { name: "a", label: "semi-major axis", unit: "au", value: 2.57, sigma: 0.001 } ];

const $ = (id) => window.document.getElementById(id);
let pass = 0, fail = 0;
const check = (name, cond) => { if (cond) { pass++; console.log("PASS", name); } else { fail++; console.log("FAIL", name); } };

run(`CURRENT = ${JSON.stringify(satPayload)}; renderStatus(CURRENT); renderElements(CURRENT);`);
check("badge shown for satellite", /chip sat/.test($("status").innerHTML) && /satellite of Jupiter/.test($("status").innerHTML));
check("toggle visible", !$("satToggle").hidden);
check("toggle has both buttons", $("satToggle").querySelectorAll("button.segbtn").length === 2);
check("default view = planetocentric (a≈0.0767)", $("elemBody").textContent.includes("0.0767"));
check("planeto view active button = Jupiter-centric",
      /active[^>]*data-ev="planeto"|data-ev="planeto"[^>]*active/.test($("satToggle").innerHTML));
// The "-centric" wording lives on the toggle button (checked above); the frame label
// annotates the panel with the precise reference frame that the shown elements are in.
check("frame label shows the planetocentric reference frame",
      /J2000 ecliptic \(Jupiter system barycentre\)/.test($("frameLbl").textContent));
check("epoch control hidden in sat view", $("epochCtl").hidden);

// click "heliocentric" -> switch to helio elements (a≈5.20)
run(`document.querySelector('#satToggle button[data-ev="helio"]').click();`);
check("toggle to heliocentric shows helio a≈5.20", $("elemBody").textContent.includes("5.2"));
check("heliocentric button now active", /active[^>]*data-ev="helio"|data-ev="helio"[^>]*active/.test($("satToggle").innerHTML));

// non-satellite payload -> no badge, toggle hidden
run(`CURRENT = ${JSON.stringify(mbaPayload)}; renderStatus(CURRENT); renderElements(CURRENT);`);
check("no badge for MBA", !/chip sat/.test($("status").innerHTML));
check("toggle hidden for MBA", $("satToggle").hidden);
check("MBA shows helio a≈2.57", $("elemBody").textContent.includes("2.57"));

console.log(`\n${pass}/${pass + fail} checks passed`);
process.exit(fail ? 1 : 0);