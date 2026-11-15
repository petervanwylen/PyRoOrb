// jsdom test for the sky check page (src/pyroorb/skycheck/static/skycheck.{html,js}):
// the form takes its defaults from /api/skycheck/status (never literals in the page), a
// field check renders its objects, a pending job stub is settled through FitJob, and an
// identify renders one panel per tracklet with its verdict.
//
//     NODE_PATH=<dir with jsdom>/node_modules node tests/web/test_skycheck_page.js
//
// Skips (exit 0) when jsdom is not installed.
const fs = require("fs");
const path = require("path");
let JSDOM;
try { ({ JSDOM } = require("jsdom")); }
catch { console.log("SKIP: jsdom not installed (npm i jsdom to run this test)"); process.exit(0); }

const ROOT = path.join(__dirname, "..", "..", "src", "pyroorb");
const html = fs.readFileSync(path.join(ROOT, "skycheck", "static", "skycheck.html"), "utf8");
const fitjob = fs.readFileSync(path.join(ROOT, "web", "static", "fitjob.js"), "utf8");
const page = fs.readFileSync(path.join(ROOT, "skycheck", "static", "skycheck.js"), "utf8");

const OBJ = { designation: "4647", name: "(4647) Syuji", kind: "asteroid", ra: 62.38487,
  dec: 14.69240, mag: 19.23, rate_ra: 0.63, rate_dec: 0.17, r: 3.247, delta: 3.553,
  phase: 16.4, elongation: 64.2, U: 0, oppositions: 20, offset_ra: 0.1, offset_dec: -0.2,
  separation: 0.25 };
const calls = [];
let fieldReply = { ok: true, pending: true, job: "j1" };

function reply(body) {
  return Promise.resolve({ ok: true, status: 200, json: async () => body,
                           text: async () => JSON.stringify(body) });
}

const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                      { url: "http://localhost/skycheck", runScripts: "dangerously" });
const w = dom.window;
w.fetch = (url, opts) => {
  calls.push([url, opts && opts.body ? JSON.parse(opts.body) : null]);
  if (url === "/api/skycheck/status")
    return reply({ ok: true, ready: true, orbits: 1579603, built: 1794689452,
                   defaults: { radius: 15, match: 30, motion: 4, max_obs: 3000 } });
  if (url === "/api/skycheck/field") return reply(fieldReply);
  if (url.startsWith("/api/fit_result"))
    return reply({ ok: true, done: true, state: "done",
                   result: { ok: true, time_utc: 2461119.60452, ra: 62.3848, dec: 14.6925,
                             radius_arcmin: 3, obscode: "G96", objects: [OBJ], notes: [] } });
  if (url === "/api/skycheck/identify")
    return reply({ ok: true, n_obs: 2, tracklets: [{
      label: "CEEK552", obscode: "G96", n_obs: 2, t_first: 2461119.60452,
      t_last: 2461119.61452, ra: 62.3848, dec: 14.6925, identification: "4647",
      ambiguous: false, candidates: [{ ...OBJ, mean_offset: 0.6, motion_rms: 0.86,
                                       match: true }] }] });
  return reply({ ok: false, error: "unexpected " + url });
};
w.eval(fitjob);
w.eval(page);

const $ = (id) => w.document.getElementById(id);
const tick = () => new Promise((r) => setTimeout(r, 20));
let failed = 0;
function check(cond, what) {
  if (!cond) { failed++; console.log("FAIL:", what); } else console.log("ok:", what);
}

(async () => {
  for (let k = 0; k < 10; k++) await tick();
  check($("fRadius").value === "15" && $("iMatch").value === "30" && $("iMotion").value === "4",
        "the form's defaults come from /api/skycheck/status");
  check(/1,579,603|1579603/.test($("catInfo").textContent), "the catalogue size is shown");

  $("fRa").value = "04 09 32.36"; $("fDec").value = "+14 41 32.9"; $("fTime").value = "2026 03 20.10452";
  $("fStation").value = "G96"; $("fRadius").value = "3";
  $("fGo").click();
  for (let k = 0; k < 60 && !/4647/.test($("results").innerHTML); k++) await tick();
  const sent = calls.find((c) => c[0] === "/api/skycheck/field")[1];
  check(sent.station === "G96" && sent.radius === "3" && sent.neocp === false,
        "the field request carries the form");
  check(calls.some((c) => c[0].startsWith("/api/fit_result")), "a pending stub is polled");
  check(/4647/.test($("results").innerHTML) && /Syuji/.test($("results").innerHTML),
        "the field's object is rendered");

  $("tabIdentify").click();
  check(!$("paneIdentify").classList.contains("hidden") && $("paneField").classList.contains("hidden"),
        "the tabs switch panes");
  $("iText").value = "     CEEK552  C2026 03 20.10452 04 09 32.36 +14 41 32.9 ... G96";
  $("iGo").click();
  for (let k = 0; k < 60 && !/CEEK552/.test($("results").innerHTML); k++) await tick();
  check(/= 4647/.test($("results").innerHTML) && /class="match"/.test($("results").innerHTML),
        "a tracklet panel shows its identification and marks the matching row");

  if (failed) { console.log(`${failed} check(s) failed`); process.exit(1); }
  console.log("all checks passed");
})();
