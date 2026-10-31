// jsdom test for the /memory page's password unlock (static/memory.js).
//
// The whole-heap "misc" pass is gated server-side (server._misc_allowed), so the
// page has to do three things well and this guards each:
//
//   * a locked report is NOT an error -- the tables render and the misc section
//     says what is missing and what would unlock it;
//   * the password box appears only when a password would actually unlock it.
//     Where none is configured on the server there is no key, and offering the
//     box would invite the reader to guess at a lock that cannot open;
//   * the password travels in a HEADER, never in the URL -- a query parameter
//     lands in every access log and proxy trace on the way.
//
//     npm i jsdom && node tests/web/test_memory_lock.js
//
// Skips (exit 0) when jsdom is not installed.
const fs = require("fs");
const path = require("path");
let JSDOM;
try { ({ JSDOM } = require("jsdom")); }
catch { console.log("SKIP: jsdom not installed (npm i jsdom to run this test)"); process.exit(0); }

const ROOT = path.join(__dirname, "..", "..", "src", "pyroorb", "web", "static");
const html = fs.readFileSync(path.join(ROOT, "memory.html"), "utf8");
const js = fs.readFileSync(path.join(ROOT, "memory.js"), "utf8");

const BASE = {
  ok: true, when: "2026-08-22 12:00:00 UTC", pid: 42, cost_secs: 0.02,
  process: { rss_mb: 512.0, rss_peak_mb: 700.5, threads: 20 },
  container: { available_mb: 4000, sweep_min_free_mb: 1000 },
  searches: { running: 0, rows: [] },
  tables: [{ key: "sessions", title: "Interactive fit sessions (_SESSIONS)",
             note: "n", loaded: true, mb: 40.0, count: 2, cap: 12, rows: [] }],
  totals: { tables_gross_mb: 40, tables_combined_mb: 40, overlap_mb: 0,
            misc_mb: null, accounted_mb: null, rss_mb: 512.0, native_mb: null },
};

const LOCKED = Object.assign({}, BASE, {
  misc: null,
  misc_locked: { reason: "password-required", password_required: true,
                 message: "The whole-heap pass needs the memory password." },
});
const WRONG = Object.assign({}, BASE, {
  misc: null,
  misc_locked: { reason: "password-wrong", password_required: true,
                 message: "That password does not match." },
});
const NO_KEY = Object.assign({}, BASE, {
  misc: null,
  misc_locked: { reason: "remote-no-password", password_required: false,
                 message: "The whole-heap pass is limited to the local console." },
});
const UNLOCKED = Object.assign({}, BASE, {
  misc: { mb: 88.0, objects: 300000, truncated: false,
          by_type: [{ type: "ndarray", mb: 50.0, count: 10 }], biggest: [] },
});

// What the server hands back next, and what it was asked with.
let NEXT = LOCKED;
const calls = [];

const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                      { url: "http://localhost/memory", runScripts: "outside-only" });
const { window } = dom;
window.fetch = (url, opts) => {
  calls.push({ url, headers: (opts && opts.headers) || {} });
  return Promise.resolve({ status: NEXT.__status || 200,
                           json: () => Promise.resolve(NEXT) });
};
window.eval(js);

let failures = 0;
function check(name, cond, extra) {
  if (cond) { console.log("  ok   " + name); }
  else { console.log("  FAIL " + name + (extra ? "  -- " + extra : "")); failures++; }
}
const tick = (fn) => new Promise((res) => setTimeout(() => { fn(); res(); }, 30));

(async () => {
  const doc = window.document;

  // --- locked on load ------------------------------------------------------
  await tick(() => {});
  check("a locked report still renders the tables",
        doc.getElementById("body").textContent.includes("Interactive fit sessions"));
  check("the misc section says it is locked and why",
        doc.getElementById("body").textContent.includes("needs the memory password"));
  check("no error banner -- locked is not a fault",
        doc.getElementById("err").textContent === "",
        JSON.stringify(doc.getElementById("err").textContent));
  check("the password box is offered", doc.getElementById("pwBox").hidden === false);
  check("first request carries no password header",
        !("X-Memory-Password" in calls[0].headers));

  // --- a wrong password is reported, and does not vanish silently ----------
  NEXT = WRONG;
  doc.getElementById("pw").value = "wrong-one";
  doc.getElementById("unlock").dispatchEvent(new window.Event("click"));
  await tick(() => {});
  check("the password is sent as a header, never in the URL",
        calls[calls.length - 1].headers["X-Memory-Password"] === "wrong-one"
        && !calls[calls.length - 1].url.includes("wrong-one"),
        JSON.stringify(calls[calls.length - 1]));
  check("a wrong password is reported to the reader",
        doc.getElementById("err").textContent.includes("does not match"));
  check("the box stays open to try again",
        doc.getElementById("pwBox").hidden === false);
  check("the typed password is cleared from the field",
        doc.getElementById("pw").value === "");

  // --- the right one ------------------------------------------------------
  NEXT = UNLOCKED;
  doc.getElementById("pw").value = "correct-horse";
  doc.getElementById("unlock").dispatchEvent(new window.Event("click"));
  await tick(() => {});
  check("the whole-heap breakdown renders once unlocked",
        doc.getElementById("body").textContent.includes("ndarray"));
  check("the box is put away when nothing is locked",
        doc.getElementById("pwBox").hidden === true);
  check("a Lock button appears to forget the password",
        doc.getElementById("lock").hidden === false);

  // it must survive the auto-refresh, not just the unlocking request
  doc.getElementById("refresh").dispatchEvent(new window.Event("click"));
  await tick(() => {});
  check("the password rides every later request",
        calls[calls.length - 1].headers["X-Memory-Password"] === "correct-horse");

  // --- and can be forgotten ------------------------------------------------
  NEXT = LOCKED;
  doc.getElementById("lock").dispatchEvent(new window.Event("click"));
  await tick(() => {});
  check("Lock forgets it", !("X-Memory-Password" in calls[calls.length - 1].headers));
  check("...and it does not survive in session storage",
        !window.sessionStorage.getItem("pyroorb.memoryPassword"));

  // --- a server with no password configured must not offer the box ---------
  NEXT = NO_KEY;
  doc.getElementById("refresh").dispatchEvent(new window.Event("click"));
  await tick(() => {});
  check("no box where no password exists to type",
        doc.getElementById("pwBox").hidden === true);
  check("...but the reason is still on the page",
        doc.getElementById("body").textContent.includes("local console"));

  // --- the hosted proxy's own reply when the server did not answer -------
  // (Railway: {"status":"error","code":502,"message":...} -- no `ok`, no
  // `error`, which the page used to report as a bare "report failed")
  NEXT = { __status: 502, status: "error", code: 502,
           message: "Application failed to respond" };
  doc.getElementById("refresh").dispatchEvent(new window.Event("click"));
  await tick(() => {});
  const err = doc.getElementById("err").textContent;
  check("a proxy reply is named as the server not answering, with its status",
        err.includes("did not answer") && err.includes("502")
        && err.includes("Application failed to respond"), JSON.stringify(err));

  console.log(failures ? `\n${failures} FAILURE(S)` : "\nall memory-lock checks passed");
  process.exit(failures ? 1 : 0);
})();
