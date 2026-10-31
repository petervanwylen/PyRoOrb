// jsdom test for the force-model plain-text copy button (ngcopy.js).
//
// The panel is an HTML <table>, whose text/plain clipboard flavour is TAB-separated -- and a
// tab pasted into a rich-text mail composer collapses to one space or to nothing, running the
// coefficients together.  So the button must hand over a SPACE-aligned rendering: this drives
// app.js's real renderNongrav() and asserts the copied text has no tabs, keeps the jackknife
// sub-line under its coefficient, and lines the columns up.
//
//     npm i jsdom && node tests/web/test_ngcopy.js
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
  `<option value="comet3">comet A1+A2+A3 (Marsden)</option>`;

// The 2026-09 report that prompted this: three coefficients, each with a jackknife sub-row.
const mk = (se, z) => ({ se, z, spans_zero: false, n: 16, attempted: 16,
                         n_tracklets: 76, sampled: true, assessed: true });
w.renderNongrav({
  force_model: "ASSIST (DE440 + GR)", session: "s1", asteroids: false, nongrav_request: "comet3",
  nongrav: {
    model: "comet3",
    params: { A1: 1.453e-11, A2: 1.433e-12, A3: -2.715e-11 },
    _sigmas: { A1: 4.570e-11, A2: 5.641e-13, A3: 1.902e-12 },
    _unc: [],
    _jk: { A1: mk(3.137e-11, 0.5), A2: mk(2.497e-13, 5.7), A3: mk(4.173e-13, 65.1) },
  },
});

const btn = doc.querySelector("#ngPanel .ngcopybtn");
if (!btn) fail("no copy button in the force-model header");
if (!doc.querySelector("#ngPanel > .hd .hdright #astToggle"))
  fail("the gravity/+16ast toggle was not grouped with the button");

btn.dispatchEvent(new w.MouseEvent("click", { bubbles: true }));

setTimeout(() => {
  if (copied === null) fail("clicking the button copied nothing");
  const lines = copied.split("\n");
  if (/\t/.test(copied)) fail("copied text still contains a TAB");
  if (!/^force model: comet A1\+A2\+A3/.test(lines[0]))
    fail("first line does not name the model: " + JSON.stringify(lines[0]));

  const coeff = lines.filter((l) => /^A[123] /.test(l));
  if (coeff.length !== 3) fail("expected 3 coefficient lines, got " + coeff.length);
  for (const l of coeff) {
    if (!/A[123] +[-\d.e+]+ +± [-\d.e+]+ +[\d.]+σ$/.test(l))
      fail("coefficient line is not space-separated: " + JSON.stringify(l));
  }
  // Columns line up: the ± starts at the same offset on every coefficient line.
  const at = coeff.map((l) => l.indexOf("±"));
  if (new Set(at).size !== 1) fail("± column not aligned: " + at.join(","));

  // Each jackknife line follows its coefficient, indented, on its own line.
  const jk = lines.filter((l) => l.includes("robustness (jackknife)"));
  if (jk.length !== 3) fail("expected 3 jackknife lines, got " + jk.length);
  for (const [i, l] of jk.entries()) {
    if (!/^ {3,}robustness \(jackknife\): ± /.test(l))
      fail("jackknife line not indented: " + JSON.stringify(l));
    if (lines[lines.indexOf(l) - 1] !== coeff[i])
      fail("jackknife line is not directly under its coefficient");
    if (!/16 of 76 tracklets tried, 16 converged/.test(l))
      fail("jackknife coverage lost: " + JSON.stringify(l));
  }
  if (!copied.includes("± = profile-likelihood (Δχ²=1)")) fail("footnote lost");
  if (!copied.includes("A in au/day²")) fail("unit line lost");
  if (btn.textContent !== "✓") fail("button gave no copied feedback");

  console.log("PASS: ngcopy.js — " + lines.length + " space-aligned lines, no tabs");
  process.exit(0);
}, 0);
