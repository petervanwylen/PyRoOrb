// jsdom test for the AMR line in the force-model panel (app.js ngAmrLine).
//
// find_orb prints the implied area/mass ratio on the elements' Epoch line for its
// 1-parameter SRP force model ONLY (elem_out.cpp).  The panel follows that rule: the AMR
// line appears on model `srp`, carries the ± once the profile sigmas are in, is absent on
// srp2/comet3 (server sends amr:null), and copies with the rest of the block.
//
//     npm i jsdom && node tests/web/test_ngamr.js
//
// Skips (exit 0) when jsdom is not installed.
const fs = require("fs");
const path = require("path");
let JSDOM;
try { ({ JSDOM } = require("jsdom")); }
catch { console.log("SKIP: jsdom not installed (npm i jsdom to run this test)"); process.exit(0); }

const ROOT = path.join(__dirname, "..", "..", "src", "pyroorb", "web", "static");
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8");
const src = fs.readFileSync(path.join(ROOT, "fitjob.js"), "utf8") + "\n" +
  fs.readFileSync(path.join(ROOT, "app.js"), "utf8") + "\n" +
  fs.readFileSync(path.join(ROOT, "ngcopy.js"), "utf8");

const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                      { url: "http://localhost/", runScripts: "dangerously",
                        pretendToBeVisual: true });
const w = dom.window;
w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
w.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
w.requestAnimationFrame = (f) => setTimeout(f, 0);
w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
  get: (t, k) => (k === "measureText" ? () => ({ width: 1 }) : () => {}) });
w.fetch = async () => ({ ok: true, json: async () => ({}), text: async () => "" });

let copied = null;
Object.defineProperty(w.navigator, "clipboard", {
  value: { writeText: async (t) => { copied = t; } }, configurable: true });

const s = w.document.createElement("script");
s.textContent = src;
w.document.body.appendChild(s);

function fail(msg) { console.error("FAIL: " + msg); process.exit(1); }

const doc = w.document;
doc.getElementById("ngSelect").innerHTML =
  `<option value="srp">Solar radiation pressure (radial, 1/r^2)</option>`;

function fail(msg) { console.error("FAIL: " + msg); process.exit(1); }
const amrText = () => {
  const el = doc.querySelector("#ngBody .ngamr");
  return el ? el.textContent.replace(/\s+/g, " ").trim() : null;
};

// (1) srp, sigmas not yet computed: the value alone, no ±.
w.renderNongrav({
  force_model: "ASSIST (DE440 + GR)", session: "s1", nongrav_request: "srp",
  nongrav: { model: "srp", params: { A1: 7.7e-9 }, amr: 0.034 },
});
if (amrText() !== "AMR 0.034 m²/kg") fail("AMR line before sigmas: " + JSON.stringify(amrText()));

// (2) same fit with the profile ± in (a symmetric interval -> a plain ±).
w.renderNongrav({
  force_model: "ASSIST (DE440 + GR)", session: "s1", nongrav_request: "srp",
  nongrav: { model: "srp", params: { A1: 7.7e-9 }, amr: 0.034,
             _sigmas: { A1: 4.5e-10 }, _unc: [],
             _amrSigma: 0.002, _amrInterval: [0.002, 0.00201] },
});
if (amrText() !== "AMR 0.034 ± 0.002 m²/kg") fail("AMR ± line: " + JSON.stringify(amrText()));

// (3) an ASYMMETRIC Δχ²=1 interval is shown as +plus / −minus, like the coefficient rows.
w.renderNongrav({
  force_model: "ASSIST (DE440 + GR)", session: "s1", nongrav_request: "srp",
  nongrav: { model: "srp", params: { A1: 7.7e-9 }, amr: 0.034,
             _sigmas: { A1: 4.5e-10 }, _unc: [],
             _amrSigma: 0.002, _amrInterval: [0.001, 0.006] },
});
if (amrText() !== "AMR 0.034 +0.006 / −0.001 m²/kg") fail("asymmetric: " + JSON.stringify(amrText()));

// (3b) a big AMR (foil, a sail, a rocket body) must not lose its trailing zeros.
w.renderNongrav({
  force_model: "ASSIST (DE440 + GR)", session: "s1", nongrav_request: "srp",
  nongrav: { model: "srp", params: { A1: 2.3e-5 }, amr: 100.0 },
});
if (amrText() !== "AMR 100 m²/kg") fail("large AMR: " + JSON.stringify(amrText()));

// (3c) an axis the profile never bounded: a LOWER bound on the ±, never a plain "±".
w.renderNongrav({
  force_model: "ASSIST (DE440 + GR)", session: "s1", nongrav_request: "srp",
  nongrav: { model: "srp", params: { A1: 7.7e-9 }, amr: 0.034,
             _sigmas: { A1: 4.5e-10 }, _unc: ["A1"],
             _amrSigma: 0.002, _amrInterval: [0.002, 0.00201] },
});
if (amrText() !== "AMR 0.034 > ± 0.002 m²/kg") fail("unconstrained: " + JSON.stringify(amrText()));

// (4) every other model: the server sends amr:null and the panel says nothing.
for (const model of ["srp2", "comet3"]) {
  w.renderNongrav({
    force_model: "ASSIST (DE440 + GR)", session: "s1", nongrav_request: model,
    nongrav: { model, params: { A1: 7.7e-9, A2: 1.4e-12, A3: -2.7e-11 }, amr: null },
  });
  if (amrText() !== null) fail(`AMR claimed on ${model}: ` + amrText());
}

// (5) it copies with the block (ngcopy.js picks .ngamr up in document order).
w.renderNongrav({
  force_model: "ASSIST (DE440 + GR)", session: "s1", nongrav_request: "srp",
  nongrav: { model: "srp", params: { A1: 7.7e-9 }, amr: 0.034,
             _sigmas: { A1: 4.5e-10 }, _unc: [], _amrSigma: 0.002,
             _amrInterval: [0.002, 0.00201] },
});
doc.querySelector("#ngPanel .ngcopybtn").dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
setTimeout(() => {
  const lines = (copied || "").split("\n");
  const i = lines.findIndex((l) => l.startsWith("AMR "));
  if (i < 0) fail("AMR line missing from the copied text:\n" + copied);
  if (lines[i] !== "AMR 0.034 ± 0.002 m²/kg") fail("copied AMR: " + JSON.stringify(lines[i]));
  if (!/^A1 /.test(lines[i - 1])) fail("AMR did not copy directly under A1: " + lines[i - 1]);
  console.log("PASS: AMR line — srp only, ± when computed, and it copies");
  process.exit(0);
}, 0);
