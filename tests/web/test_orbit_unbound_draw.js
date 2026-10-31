// jsdom test for the two orbit scenes on an UNBOUND orbit (1I / 2I / 3I).
//
// Two defects, both in app.js's geometry rather than the server's sampling:
//   1. splitOrbitByEcliptic walked the path's wrap edge as if every path were a closed
//      ellipse, so the hyperbola came back with a straight chord joining its inbound end to
//      its outbound end. The server now says `closed: false` and the split honours it.
//   2. The date slider advanced M only through the ELLIPTIC Kepler solve, which the two
//      position functions skipped for e >= 1 -- so the planets moved and the object did not.
//      keplerXYZ now branches to the hyperbolic solve, and Orbarium admits an unbound orbit.
//
// Loads the real index.html + app.js and calls the geometry through window.eval, so a
// renamed or re-gated function fails here.
//
//     npm i jsdom && node tests/web/test_orbit_unbound_draw.js
//
// Skips (exit 0) when jsdom is not installed.
const fs = require("fs");
const nodePath = require("path");
let JSDOM;
try { ({ JSDOM } = require("jsdom")); }
catch { console.log("SKIP: jsdom not installed (npm i jsdom to run this test)"); process.exit(0); }

const ROOT = nodePath.join(__dirname, "..", "..", "src", "pyroorb", "web", "static");
const html = fs.readFileSync(nodePath.join(ROOT, "index.html"), "utf8");
const appjs = fs.readFileSync(nodePath.join(ROOT, "fitjob.js"), "utf8") + "\n" +
  fs.readFileSync(nodePath.join(ROOT, "app.js"), "utf8");

const dom = new JSDOM(html, { url: "http://localhost/", runScripts: "dangerously", pretendToBeVisual: true });
const w = dom.window;
w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
w.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
w.requestAnimationFrame = (f) => setTimeout(f, 0);
w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
  get: (t, k) => (k === "measureText" ? () => ({ width: 1 }) : () => {}) });
w.fetch = async () => ({ text: async () => "{}", json: async () => ({}) });
const scr = w.document.createElement("script");
scr.textContent = appjs;
w.document.body.appendChild(scr);

let failures = 0;
function check(name, cond) {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}`);
  if (!cond) failures++;
}
const norm = (p) => Math.hypot(p[0], p[1], p[2]);

// --- a 3I-like hyperbola, sampled the way engine_api._orbit_path samples it ---------
const a = -0.2638, e = 6.1394, inc = 175.11, node = 322.16, peri = 128.0;
const q = a * (1 - e);
const s = -a, sq = Math.sqrt(e * e - 1);
const rMax = Math.max(6, 4 * q), hmax = Math.acosh((rMax / s + 1) / e);
const N = 240, path = [];
for (let k = 0; k <= N; k++) {
  const H = -hmax + 2 * hmax * k / N;
  path.push(w.eval("rotEcl")(s * (e - Math.cosh(H)), s * sq * Math.sinh(H), node, peri, inc));
}
// mean motion sqrt(mu/(-a)^3) in deg/day, Gaussian k = 0.01720209895 rad/day
const nDeg = 0.01720209895 / Math.sqrt(s ** 3) * 180 / Math.PI;
const orbit3d = { a, e, i: inc, node, peri, M: 0, epoch_jd: 2460000.5, n_deg_per_day: nDeg,
  q, Q: null, path, position: path[N / 2], closed: false };

// 1. the split of an OPEN curve draws no seam ------------------------------------------
const split = w.eval("splitOrbitByEcliptic");
const open = split(path, false), wrapped = split(path, true);
const segPts = (r) => r.segments.reduce((n, sg) => n + sg.pts.length, 0);
const chord = (pts) => { let m = 0; for (let i = 1; i < pts.length; i++) m = Math.max(m, norm([pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1], pts[i][2] - pts[i-1][2]])); return m; };
const longestOpen = Math.max(...open.segments.map((sg) => chord(sg.pts)));
const longestWrapped = Math.max(...wrapped.segments.map((sg) => chord(sg.pts)));
check("open split keeps every sampled point and adds only node crossings",
  segPts(open) >= path.length && segPts(open) <= path.length + 2 * open.nodes.length);
check("open split has no chord longer than a sample step (no end-to-end seam)",
  longestOpen < 0.5 && longestWrapped > 5,
  );
check("open split reports at most the nodes the branch actually crosses",
  open.nodes.length <= 2 && open.nodes.length === wrapped.nodes.length - (wrapped.nodes.length > open.nodes.length ? 1 : 0));
const ellipse = [];
for (let k = 0; k <= 90; k++) { const E = 2 * Math.PI * k / 90; ellipse.push(w.eval("rotEcl")(2.5 * (Math.cos(E) - 0.1), 2.5 * Math.sqrt(1 - 0.01) * Math.sin(E), 10, 20, 5)); }
check("closed split still walks the seam by default (two nodes on an inclined ellipse)",
  split(ellipse).nodes.length === 2 && split(ellipse, true).nodes.length === 2);

// 2. the hyperbolic Kepler solve places the body on the branch --------------------------
const kxyz = w.eval("keplerXYZ");
const atPeri = kxyz(a, e, inc, node, peri, 0);
check("M = 0 sits at perihelion distance q", Math.abs(norm(atPeri) - q) < 1e-9 * q);
const before = kxyz(a, e, inc, node, peri, -30), after = kxyz(a, e, inc, node, peri, 30);
check("negative M is inbound, positive M outbound, symmetric in distance",
  norm(before) > q && norm(after) > q && Math.abs(norm(before) - norm(after)) < 1e-9);
// the solved point lies ON the sampled branch (nearest sample within a sample step)
const nearest = (p) => Math.min(...path.map((pp) => norm([pp[0]-p[0], pp[1]-p[1], pp[2]-p[2]])));
check("solved positions lie on the sampled branch", nearest(before) < 0.2 && nearest(after) < 0.2);

// 3. the date advance moves the object (Orbit3D's dot and Orbarium's tips both use it) --
const posAt = w.eval("objPosAt");
const p0 = posAt(orbit3d, orbit3d.epoch_jd), p1 = posAt(orbit3d, orbit3d.epoch_jd + 100), pm = posAt(orbit3d, orbit3d.epoch_jd - 100);
check("objPosAt moves an unbound object along its branch under the slider",
  p0 && p1 && pm && norm([p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]]) > 0.5 && norm(p1) > norm(p0) && norm(pm) > norm(p0));
const O3 = w.eval("Orbit3D");
const canvas = w.document.getElementById("orbitCanvas");
const view = new O3(canvas, orbit3d, w.document.getElementById("legend"));
const d0 = view.asteroidPos(); view.jd += 200; const d1 = view.asteroidPos();
check("Orbit3D.asteroidPos follows this.jd on an unbound orbit",
  d0 && d1 && norm([d1[0]-d0[0], d1[1]-d0[1], d1[2]-d0[2]]) > 0.5);
check("Orbit3D frames the far end of the branch when there is no aphelion", view.span > rMax);

// 4. Orbarium admits the unbound orbit and moves its tips with the observation dates ---
const Orb = w.eval("Orbarium");
const orb = new Orb(w.document.getElementById("orbariumCanvas"), w.document.getElementById("orbariumLegend"));
const residuals = [0, 20, 40, 60].map((dd) => ({ included: true, jd_utc: orbit3d.epoch_jd - 60 + dd, obscode: "F51" }));
orb.setData({ orbit3d, residuals });
check("Orbarium admits an unbound orbit (no 'needs a converged orbit' note)", orb.note === null && orb.arrows.length === 4);
const tips = orb.arrows.map((ar) => ar.obj);
check("Orbarium sightline tips are distinct points along the branch, not one static dot",
  tips.every((t, i) => i === 0 || norm([t[0]-tips[i-1][0], t[1]-tips[i-1][1], t[2]-tips[i-1][2]]) > 0.05));
orb.setData({ orbit3d: { ...orbit3d, n_deg_per_day: null }, residuals });
check("Orbarium still refuses an orbit with no mean motion", typeof orb.note === "string" && orb.note.includes("converged"));

console.log(failures ? `\n${failures} FAILED` : "\nall passed");
process.exit(failures ? 1 : 0);
