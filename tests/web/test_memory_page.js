// jsdom test for the /memory page (static/memory.js).
//
// The page's whole job is to make the biggest thing obvious, so what is worth
// guarding is the rendering CONTRACT, not the styling: every loaded table appears,
// tables are ordered largest-first, misc is shown as its own category with its
// type breakdown, and the headroom card turns red when the ITF sweep floor would
// bite. Plus the fetch wiring: ?tables=1 in fast mode.
//
//     npm i jsdom && node tests/web/test_memory_page.js
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

const REPORT = {
  ok: true, when: "2026-08-19 12:00:00 UTC", pid: 42, cost_secs: 0.6,
  process: { rss_mb: 512.0, rss_peak_mb: 700.5, threads: 20 },
  container: { available_mb: 400, sweep_min_free_mb: 1000 },
  searches: { running: 1, rows: [{ id: "abc12345…", elapsed_s: 90.5,
                                   rss_start_mb: 300.0, rss_delta_mb: 212.0 }] },
  sweeps: { inflight: [{ thread: "itf-worker-1", backend: "rebound", label: null,
                         elapsed_s: 3612.4, n: 300, at: 6, target_days: 512.25, wedged: true,
                         elements: { r_au: 1.0123, a_au: 0.98, e: 0.5123, q_au: 0.4781 } }],
            wedged: 1, wedged_sec: 600,
            slow: [{ finished: "2026-09-26 12:00:00", backend: "assist", label: "iod",
                     elapsed_s: 290.1, steps_done: 4123456, n: 3,
                     elements: { r_au: 2.0, a_au: 1.5, e: 0.99, q_au: 0.02 } }],
            slow_sec: 5 },
  tables: [
    { key: "sessions", title: "Interactive fit sessions (_SESSIONS)", note: "n", loaded: true,
      mb: 40.0, count: 12, cap: 12, rows: [{ id: "s7", label: "2023 QJ3", n_obs: 4356, mb: 7.3 }] },
    { key: "debias", title: "EFCC18 debias table", note: "n", loaded: true,
      mb: 19.5, count: 1, cap: null, rows: [], detail: "(49152, 26, 4) float32" },
    { key: "fitcache", title: "Finished-fit cache", note: "n", loaded: true,
      mb: 60.0, count: 16, cap: 16, rows: [] },
    { key: "rankers", title: "ML rankers", note: "n", loaded: false, mb: 0.0 },
  ],
  misc: { mb: 88.0, objects: 300000, truncated: false,
          by_type: [{ type: "ndarray", mb: 50.0, count: 10 },
                    { type: "dict", mb: 20.0, count: 40000 }],
          biggest: [{ mb: 20.0, what: "numpy.ndarray (49152, 26, 4) float32" }] },
  totals: { tables_gross_mb: 126.0, tables_combined_mb: 119.5, overlap_mb: 6.5,
            misc_mb: 88.0, accounted_mb: 207.5, python_heap_tracked_mb: 100.0,
            python_objects_tracked: 300000, rss_mb: 512.0, native_mb: 304.5 },
};

const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                      { url: "http://localhost/memory", runScripts: "outside-only" });
const { window } = dom;
const fetched = [];
window.fetch = (url) => {
  fetched.push(url);
  return Promise.resolve({ json: () => Promise.resolve(REPORT) });
};
window.eval(js);

let failures = 0;
function check(name, cond, extra) {
  if (cond) { console.log("  ok   " + name); }
  else { console.log("  FAIL " + name + (extra ? "  -- " + extra : "")); failures++; }
}

setTimeout(() => {
  const doc = window.document;
  const body = doc.getElementById("body").textContent;
  const cards = doc.getElementById("cards").textContent;

  check("fetches the report on load", fetched.length === 1 && fetched[0] === "/api/memory",
        JSON.stringify(fetched));
  check("shows RSS and peak", cards.includes("512.0") && cards.includes("700.5"));
  check("shows the running-search count", cards.includes("Searches running"));
  check("headroom below the sweep floor is flagged",
        doc.querySelector(".card.bad") !== null);

  check("every loaded table renders", body.includes("Interactive fit sessions")
        && body.includes("Finished-fit cache") && body.includes("EFCC18 debias table"));
  const order = ["Finished-fit cache", "Interactive fit sessions", "EFCC18"]
    .map((t) => body.indexOf(t));
  check("tables are ordered largest-first", order[0] < order[1] && order[1] < order[2],
        JSON.stringify(order));
  check("per-entry detail rows render", body.includes("2023 QJ3") && body.includes("4356 obs"));
  check("cap is shown as used / cap", body.includes("12 / 12"));
  check("the overlap line explains double-holding", body.includes("6.5"));
  check("not-loaded tables are named, not silently dropped", body.includes("rankers"));

  check("misc is its own category with its size", body.includes("Misc") && body.includes("88.0"));
  check("misc type breakdown renders", body.includes("ndarray") && body.includes("50.0"));
  check("largest misc objects render", body.includes("(49152, 26, 4) float32"));

  check("an in-flight sweep renders with its thread, age, progress and orbit",
        body.includes("Propagator sweeps in flight") && body.includes("itf-worker-1")
        && body.includes("3612.4") && body.includes("7 / 300") && body.includes("0.4781"));
  check("a wedged sweep is flagged", doc.querySelector("tr.bad") !== null);
  check("the slow-sweep ring renders with its step count",
        body.includes("Slowest sweeps lately") && body.includes("4123456")
        && body.includes("290.1"));
  check("running searches table renders", body.includes("Running ITF searches")
        && body.includes("+212.0"));
  check("footer states the native remainder", doc.getElementById("foot").textContent
        .includes("304.5"));

  // fast mode must ask the server to skip the expensive whole-heap pass
  doc.getElementById("fast").checked = true;
  doc.getElementById("fast").dispatchEvent(new window.Event("change"));
  setTimeout(() => {
    check("tables-only mode requests ?tables=1",
          fetched[fetched.length - 1] === "/api/memory?tables=1", JSON.stringify(fetched));
    console.log(failures ? `\n${failures} FAILURE(S)` : "\nall memory-page checks passed");
    process.exit(failures ? 1 : 0);
  }, 50);
}, 50);
