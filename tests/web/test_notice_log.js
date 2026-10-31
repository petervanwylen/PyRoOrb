// jsdom test for the notice log (noticelog.js). Covers the framework the two
// clarifications settled on:
//   * a banner is FILED (badge++, fly-dot launched) when it GOES AWAY, not when it
//     appears — for every banner kind;
//   * ITF-search / linkage banners are marked .sticky (no auto-fade) and only leave
//     on their × or on a click somewhere off the banners; a click ON a banner leaves
//     it be;
//   * the corner indicator opens a panel listing filed notices newest-first, and
//     Clear empties it.
//
//     npm i jsdom && node tests/web/test_notice_log.js
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
const logjs = fs.readFileSync(path.join(ROOT, "noticelog.js"), "utf8");

const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                      { url: "http://localhost/", runScripts: "dangerously",
                        pretendToBeVisual: true });
const w = dom.window;
w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
w.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
w.requestAnimationFrame = (f) => setTimeout(f, 0);
w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
  get: (t, k) => (k === "measureText" ? () => ({ width: 1 }) : () => {}) });
w.fetch = async () => ({ json: async () => ({}) });

// app.js reads every response body as TEXT and parses it (so a proxy answering
// a request it gave up on with a non-JSON page yields a readable message
// instead of "Unexpected token 'u'"). Give these mocks a text() for free.
w.fetch = ((f) => async (...a) => {
  const r = await f(...a);
  return r.text ? r : { ...r, text: async () => JSON.stringify(await r.json()) };
})(w.fetch);

for (const src of [appjs, logjs]) {
  const scr = w.document.createElement("script");
  scr.textContent = src;
  w.document.body.appendChild(scr);
}

const tick = () => new Promise((r) => setTimeout(r, 25));

(async () => {
  const $ = (id) => w.document.getElementById(id);
  const host = () => $("flashHost");
  const flashes = () => [...host().querySelectorAll(".flash")];
  const flydots = () => w.document.querySelectorAll(".banner-flydot").length;
  const assert = (c, m) => { if (!c) { console.error("FAIL:", m); process.exit(1); } };

  await tick();   // let jsdom arm the MutationObserver set up during script injection

  assert($("bannerLogBtn"), "corner indicator button exists");
  assert($("bannerLogCount").hidden, "count badge hidden before any banner");

  // --- a plain (timer) banner: appears, NOT filed yet, NOT sticky ---
  w.eval('flashBanner("First notice", "info")');
  await tick();
  assert(flashes().length === 1, "banner is showing");
  assert(!flashes()[0].classList.contains("sticky"), "plain banner is not sticky");
  assert($("bannerLogCount").hidden, "not filed while still visible");
  assert(flydots() === 0, "no fly-dot while the banner is still up");

  // its timer expiring == the element being removed -> filed + fly-dot
  flashes()[0].remove();
  await tick();
  assert($("bannerLogCount").textContent === "1", "filed on removal, count 1: " + $("bannerLogCount").textContent);
  assert($("bannerLogBtn").classList.contains("has-log"), "indicator lit after filing");
  assert(flydots() === 1, "a fly-dot was launched as it left");
  const dotBg = $("bannerLogBtn").querySelector(".log-dot").style.background;
  assert(/#56d4dd|86, ?212, ?221/.test(dotBg), "corner dot tinted info: " + dotBg);

  // --- an ITF banner is sticky and survives a click ON it ---
  w.eval('flashBanner("ITF search: no additional observations found", "info")');
  await tick();
  const itf = flashes().find((f) => /ITF/.test(f.textContent));
  assert(itf && itf.classList.contains("sticky"), "ITF banner is sticky");
  assert($("bannerLogCount").textContent === "1", "sticky banner not filed while visible");
  itf.querySelector(".flash-msg").dispatchEvent(new w.Event("click", { bubbles: true }));
  await tick();
  assert(itf.isConnected, "clicking ON the banner does not dismiss it");

  // --- a click OFF the banners dismisses the sticky one -> filed ---
  w.document.body.dispatchEvent(new w.Event("click", { bubbles: true }));
  await tick();
  assert(!itf.isConnected, "off-banner click dismisses the sticky banner");
  assert($("bannerLogCount").textContent === "2", "sticky banner filed on dismissal: " + $("bannerLogCount").textContent);
  assert(flydots() === 2, "fly-dot launched for the sticky banner too");

  // --- the log panel lists both, newest first ---
  $("bannerLogBtn").click();
  await tick();
  const panel = $("bannerLogPanel");
  assert(panel && !panel.hidden, "panel opens on indicator click");
  const rows = [...panel.querySelectorAll(".log-row")];
  assert(rows.length === 2, "two rows listed: " + rows.length);
  assert(/ITF search/.test(rows[0].textContent), "newest first: " + rows[0].textContent);
  assert(/First notice/.test(rows[1].textContent), "oldest last: " + rows[1].textContent);

  // --- the × button files a sticky banner (single dismissal path) ---
  $("bannerLogBtn").click(); await tick();          // close the panel first
  w.eval('flashBanner("Linkage submitted to the MPC", "info")');
  await tick();
  const link = flashes().find((f) => /Linkage/.test(f.textContent));
  assert(link && link.classList.contains("sticky"), "linkage banner is sticky too");
  link.querySelector(".flash-x").click();
  await tick();
  assert(!link.isConnected, "× dismisses the sticky banner");
  assert($("bannerLogCount").textContent === "3", "× dismissal files it: " + $("bannerLogCount").textContent);

  // --- Clear empties the log and resets the indicator ---
  $("bannerLogBtn").click();  await tick();          // reopen
  $("bannerLogClear").click(); await tick();
  assert($("bannerLogCount").hidden, "badge hidden after Clear");
  assert(!$("bannerLogBtn").classList.contains("has-log"), "indicator un-lit after Clear");
  assert(/No notices yet/.test($("bannerLogList").textContent), "panel shows empty state after Clear");

  console.log("PASS: notice log files banners on exit, stickies wait for ×/off-click, panel lists & clears");
  process.exit(0);
})();
