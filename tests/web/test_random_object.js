// jsdom test for the top-bar random-object button (randomobj.js):
//   * the family menu is built from GET /api/random_families (never from UI literals);
//   * a draw sets the search box from GET /api/random_object and runs the normal fit;
//   * the 30-second rate limit disables the button (and the family picker) and counts
//     down, and a reload mid-cooldown resumes the REMAINING time rather than restarting;
//   * a failed draw does NOT spend the cooldown;
//   * picking a family both remembers the choice and draws from it.
//
//     npm i jsdom && node tests/web/test_random_object.js
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
const randjs = fs.readFileSync(path.join(ROOT, "randomobj.js"), "utf8");

const FAMILIES = {
  ok: true, max_oppositions: 7,
  families: [{ key: "any", label: "Any asteroid", hint: "anything" },
             { key: "neo", label: "Near-Earth (NEO)", hint: "q < 1.3 au" },
             { key: "tno", label: "Trans-Neptunian (TNO)", hint: "a > 30 au" }],
};

// One shared store so a "reload" (a second DOM with the same backing map) keeps the clock.
const STORE = new Map();

function makeDom() {
  const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                        { url: "http://localhost/", runScripts: "dangerously",
                          pretendToBeVisual: true });
  const w = dom.window;
  w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
  w.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
  w.requestAnimationFrame = (f) => setTimeout(f, 0);
  w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
    get: (t, k) => (k === "measureText" ? () => ({ width: 1 }) : () => {}) });
  Object.defineProperty(w, "localStorage", {              // shared across "reloads"
    value: { getItem: (k) => (STORE.has(k) ? STORE.get(k) : null),
             setItem: (k, v) => STORE.set(k, String(v)),
             removeItem: (k) => STORE.delete(k) },
    configurable: true,
  });
  w.__calls = [];
  w.__nextObject = { ok: true, family: "any", family_label: "Any asteroid",
                     designation: "2024 RL175", oppositions: 1,
                     a: 2.55, e: 0.21, incl: 10.3, H: 19.4 };
  w.fetch = async (url) => {
    const u = String(url);
    w.__calls.push(u);
    const body = u.startsWith("/api/random_families") ? FAMILIES
      : u.startsWith("/api/random_object") ? w.__nextObject
      : { ok: true };
    return { json: async () => body, text: async () => JSON.stringify(body) };
  };
  for (const src of [appjs, randjs]) {
    const scr = w.document.createElement("script");
    scr.textContent = src;
    w.document.body.appendChild(scr);
  }
  w.runFit = () => { w.__fits = (w.__fits || 0) + 1; };   // don't run the real fit path
  return w;
}

const tick = () => new Promise((r) => setTimeout(r, 30));
const assert = (c, m) => { if (!c) { console.error("FAIL:", m); process.exit(1); } };

(async () => {
  let w = makeDom();
  const $ = (id) => w.document.getElementById(id);
  await tick();

  // --- menu comes from the engine ---
  const items = [...$("randomMenu").querySelectorAll("button[data-family]")];
  assert(items.length === 3, "menu built from /api/random_families: " + items.length);
  assert(items.map((b) => b.dataset.family).join(",") === "any,neo,tno",
         "families in engine order: " + items.map((b) => b.dataset.family));
  assert(/✓/.test(items[0].textContent), "current family ticked: " + items[0].textContent);
  assert(/7 oppositions/.test($("randomBtn").title),
         "opposition cap comes from the payload: " + $("randomBtn").title);
  assert(!$("randomBtn").disabled, "button live before the first draw");

  // --- a draw fills the box and fits ---
  $("randomBtn").click();
  await tick();
  assert(w.__calls.some((u) => u === "/api/random_object?family=any"), "drew from ?family=any");
  assert($("desig").value === "2024 RL175", "designation dropped into the box: " + $("desig").value);
  assert(w.__fits === 1, "the normal fit ran once");
  assert(/2024 RL175/.test(w.document.querySelector(".flash").textContent),
         "banner names the object");

  // --- rate limit: disabled, counting down, and a click does nothing ---
  assert($("randomBtn").disabled && $("randomMenuBtn").disabled,
         "both halves disabled during the cooldown");
  assert(/^🎲 \d+s$/.test($("randomBtn").textContent),
         "button shows a countdown: " + $("randomBtn").textContent);
  const before = w.__calls.length;
  $("randomBtn").click();
  await tick();
  assert(w.__calls.length === before, "a click inside the cooldown draws nothing");
  assert(w.__fits === 1, "and does not re-fit");

  // --- a reload mid-cooldown resumes the REMAINING time, not a fresh 30 s ---
  const stamp = Number(STORE.get("pyroorb.randomLastDraw"));
  w = makeDom();
  await tick();
  const $2 = (id) => w.document.getElementById(id);
  assert($2("randomBtn").disabled, "still cooling down after a reload");
  assert(Number(STORE.get("pyroorb.randomLastDraw")) === stamp,
         "the reload did not restamp the clock (that would extend the wait)");

  // --- expired cooldown releases the button ---
  STORE.set("pyroorb.randomLastDraw", String(Date.now() - 31000));
  w = makeDom();
  await tick();
  const $3 = (id) => w.document.getElementById(id);
  assert(!$3("randomBtn").disabled, "released once 30 s have passed");

  // --- picking a family: remembered, ticked, and drawn from ---
  w.__nextObject = { ok: true, family: "neo", family_label: "Near-Earth (NEO)",
                     designation: "2019 AQ3", oppositions: 4, a: 0.6, e: 0.1,
                     incl: 47.2, H: 21.2 };
  $3("randomMenu").querySelector('button[data-family="neo"]').click();
  await tick();
  assert(w.__calls.some((u) => u === "/api/random_object?family=neo"), "drew from ?family=neo");
  assert($3("desig").value === "2019 AQ3", "NEO designation in the box: " + $3("desig").value);
  assert(STORE.get("pyroorb.randomFamily") === "neo", "family choice remembered");
  assert($3("randomMenu").querySelector('button[data-family="neo"]').textContent.startsWith("✓"),
         "the chosen family is the ticked one");

  // --- a failed draw must not spend the cooldown ---
  STORE.set("pyroorb.randomLastDraw", "0");
  w = makeDom();
  await tick();
  const $4 = (id) => w.document.getElementById(id);
  w.__nextObject = { ok: false, error: "MPCORB index unavailable" };
  $4("randomBtn").click();
  await tick();
  assert(!$4("randomBtn").disabled, "button still live after a failed draw");
  assert(!STORE.get("pyroorb.randomLastDraw") || STORE.get("pyroorb.randomLastDraw") === "0",
         "no cooldown stamped for a draw that produced nothing");
  assert([...w.document.querySelectorAll(".flash")].some((f) => /MPCORB index/.test(f.textContent)),
         "the failure is surfaced as a banner");

  console.log("OK: random-object button (menu from engine, draw, 30 s limit, family memory)");
})();
