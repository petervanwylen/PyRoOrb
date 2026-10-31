// jsdom test for expirednote.js: a dead interactive session raises ONE sticky orange banner
// with a Re-fit button, not just the easy-to-miss #msg line; a repeat failure while that
// banner is still up adds nothing; once it is gone (its ×, its Re-fit, or a click anywhere
// off the banners -- noticelog's rule) the next failure raises a fresh one; and an
// `expired` answer that is NOT a session (a trace id no longer held) raises none.
//
// Drives real Full step / Multi-step clicks through a stubbed fetch, so an app.js that
// stopped calling the hook fails here.
//
//     npm i jsdom && node tests/web/test_expirednote.js
//
// Skips (exit 0) when jsdom is not installed.
const fs = require("fs");
const nodePath = require("path");
let JSDOM;
try { ({ JSDOM } = require("jsdom")); }
catch { console.log("SKIP: jsdom not installed (npm i jsdom to run this test)"); process.exit(0); }

const ROOT = nodePath.join(__dirname, "..", "..", "src", "pyroorb", "web", "static");
const html = fs.readFileSync(nodePath.join(ROOT, "index.html"), "utf8");
const srcs = ["fitjob.js", "app.js", "noticelog.js", "expirednote.js"]
  .map((f) => fs.readFileSync(nodePath.join(ROOT, f), "utf8"));

if (!/<script src="\/static\/expirednote\.js">/.test(html)) {
  console.error("FAIL: index.html does not load expirednote.js"); process.exit(1);
}

const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                      { url: "http://localhost/", runScripts: "dangerously", pretendToBeVisual: true });
const w = dom.window;
w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
w.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
w.requestAnimationFrame = (f) => setTimeout(f, 0);
w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
  get: (t, k) => (k === "measureText" ? () => ({ width: 1 }) : () => {}) });

let answer = { ok: false, error: "session expired -- re-fit the object", expired: true };
const posted = [];
w.fetch = async (url, opts) => {
  posted.push(String(url));
  const body = JSON.stringify(answer);
  return { status: 200, text: async () => body, json: async () => JSON.parse(body) };
};

for (const src of srcs) {
  const scr = w.document.createElement("script");
  scr.textContent = src;
  w.document.body.appendChild(scr);
}
const tick = () => new Promise((r) => setTimeout(r, 25));

let failures = 0;
function check(name, cond) {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}`);
  if (!cond) failures++;
}

(async () => {
  const $ = (id) => w.document.getElementById(id);
  const banners = () => [...(($("flashHost") || w.document.body).querySelectorAll(".flash.expired"))];
  await tick();   // let jsdom arm noticelog's MutationObserver

  // A fit is on screen with a live-looking session handle; the server has forgotten it.
  w.eval('CURRENT = { ok: true, session: "dead", residuals: [], object: "TEST 1" }');
  w.eval("runStep(true)");
  await tick();
  check("a Full step on a dead session posts and is answered expired", posted.some((u) => u.includes("/api/fullstep")));
  check("the #msg line still carries the failure", /expired/i.test($("msg").textContent));
  check("ONE sticky orange banner is raised", banners().length === 1
    && banners()[0].classList.contains("warn") && banners()[0].classList.contains("sticky"));
  check("the banner says what stops working", /Session expired/.test(banners()[0].textContent)
    && /re-fit/i.test(banners()[0].textContent));
  check("the banner carries a Re-fit button", !!banners()[0].querySelector(".flash-act"));

  // The user does not see it, clicks Multi-step, then Full step again: still ONE banner.
  w.eval("runStep(false)"); await tick();
  w.eval("runStep(true)"); await tick();
  check("repeat failures while the banner is still up add nothing", banners().length === 1);
  await new Promise((r) => setTimeout(r, 120));
  check("the banner does not fade on a timer", banners().length === 1);

  // Dismiss it with its ×; the next failure raises a fresh one.
  banners()[0].querySelector(".flash-x").click();
  await tick();
  check("× removes it", banners().length === 0);
  w.eval("runStep(true)"); await tick();
  check("after a dismissal the next failure raises a fresh banner", banners().length === 1);

  // A click anywhere off the banners (noticelog's sticky rule) also clears it.
  w.document.body.click(); await tick();
  check("an off-banner click clears it like every sticky banner", banners().length === 0);

  // Re-fit button: removes the banner and runs a fit.
  w.eval("runStep(true)"); await tick();
  let refitCalled = false;
  w.eval("runFit = function () { window.__refit = true; }");
  banners()[0].querySelector(".flash-act").click(); await tick();
  refitCalled = w.eval("window.__refit === true");
  check("Re-fit removes the banner and starts a fit", banners().length === 0 && refitCalled);

  // An `expired` that is not a session: a trace id the server no longer holds.
  answer = { ok: false, expired: true, error: "unknown trace id" };
  w.eval('jsonBody({ status: 200, text: async () => JSON.stringify(' + JSON.stringify(answer) + ') }, "trace")');
  await tick();
  check("a non-session `expired` (trace id) raises no session banner", banners().length === 0);
  // ...but the other two session wordings do.
  for (const msg of ["the current fit has expired -- re-run it before adding observations",
                     "that fit has expired -- re-run it to interact with it"]) {
    w.document.body.click(); await tick();
    answer = { ok: false, expired: true, error: msg };
    w.eval('jsonBody({ status: 200, text: async () => JSON.stringify(' + JSON.stringify(answer) + ') }, "x")');
    await tick();
    check(`"${msg.slice(0, 28)}..." raises the banner`, banners().length === 1);
  }

  // ensureLive failing to re-claim a restore token raises it too.
  w.document.body.click(); await tick();
  w.eval('CURRENT = { ok: true, restore: "gone", residuals: [], object: "TEST 1" }');
  w.eval("FitJob.ensureSession = async () => false");
  w.eval("ensureLive()"); await tick();
  check("a failed re-claim of an expired fit raises the banner", banners().length === 1);

  console.log(failures ? `\n${failures} FAILED` : "\nall passed");
  process.exit(failures ? 1 : 0);
})();
