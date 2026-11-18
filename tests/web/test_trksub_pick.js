// jsdom test for the trksub picker (trksubpick.js + the runFit hook in app.js).
//
// A typed trksub that names several tracklets comes back as a failed fit carrying
// `trksub_question`; the page must list the tracklets, and a click must re-fit with the
// search box rewritten -- to one qualified id, or to all of them -- keeping the other
// typed designations.  A tracklet already designated is said, with a button that types
// the designation in.
//
//     NODE_PATH=<fo_clone>/node_modules node tests/web/test_trksub_pick.js
//
// Skips (exit 0) when jsdom is not installed.
const fs = require("fs");
const path = require("path");
let JSDOM;
try { ({ JSDOM } = require("jsdom")); }
catch { console.log("SKIP: jsdom not installed (npm i jsdom to run this test)"); process.exit(0); }

const ROOT = path.join(__dirname, "..", "..", "src", "pyroorb", "web", "static");
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8");
const scripts = ["fitjob.js", "app.js", "trksubpick.js"].map((f) =>
  fs.readFileSync(path.join(ROOT, f), "utf8"));

const QUESTION = {
  ok: false,
  error: "ZZ00001 names 2 undesignated tracklets",
  typed: ["ZZ00001", "2016 FV237"],
  trksub_question: {
    ZZ00001: {
      designated: [{ trksub: "ZZ00001", stn: "W88", date: "2022-01-25", n_obs: 3,
                     designation: "2022 AB1" }],
      n_designated: 1,
      choices: [
        { trksub: "ZZ00001", stn: "954", date: "2009-06-20", n_obs: 3, token: "ZZ00001@954/2009-06-20" },
        { trksub: "ZZ00001", stn: "I41", date: "2010-08-21", n_obs: 2, token: "ZZ00001@I41/2010-08-21" },
      ],
      n_choices: 2,
    },
  },
};

const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                      { url: "http://localhost/", runScripts: "dangerously",
                        pretendToBeVisual: true });
const w = dom.window;
w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
w.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
w.requestAnimationFrame = (f) => setTimeout(f, 0);
w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
  get: (t, k) => (k === "measureText" ? () => ({ width: 1 }) : () => {}) });

const asked = [];                    // every desig the page asked the server to fit
w.fetch = async (url, opts) => {
  const u = String(url);
  let body = { ok: false, error: "unexpected request " + u };
  if (u.startsWith("/api/fit_start")) {
    const desig = JSON.parse(opts.body).desig;
    asked.push(desig);
    // first ask: the question; any re-fit: a plain failure (rendering a whole fit is
    // other tests' business -- this one is about what the page asks for)
    const result = asked.length === 1 ? QUESTION : { ok: false, error: "stop" };
    body = { ok: true, job: "j" + asked.length, state: "done", done: true, result };
  }
  const text = JSON.stringify(body);
  return { ok: true, status: 200, json: async () => JSON.parse(text), text: async () => text };
};

for (const src of scripts) {
  const scr = w.document.createElement("script");
  scr.textContent = src;
  w.document.body.appendChild(scr);
}

(async () => {
  const $ = (id) => w.document.getElementById(id);
  const assert = (c, m) => { if (!c) { console.error("FAIL:", m); process.exit(1); } };
  const tick = () => new Promise((r) => setTimeout(r, 20));
  const buttons = () => Array.from($("msg").querySelectorAll("button"));
  const ask = async () => {
    asked.length = 0;
    $("desig").value = "ZZ00001 K16FN7V";
    await w.eval("runFit()");
    await tick();
  };
  const clickAndSettle = async (b) => {
    b.dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
    for (let i = 0; i < 50 && asked.length < 2; i++) await tick();
    await tick();
  };

  // the question is shown as choices, not as "Could not fit"
  await ask();
  const text = $("msg").textContent;
  assert(!/Could not fit/.test(text), "a trksub question is not reported as a failed fit");
  assert(/names 2 undesignated tracklets/.test(text), "the count of tracklets is said");
  assert(/is already\s+2022 AB1/.test(text), "a designated tracklet names its designation");
  const labels = buttons().map((b) => b.textContent);
  assert(labels.some((l) => /954 · 2009-06-20 · 3 obs/.test(l)), "each tracklet is a button");
  assert(labels.some((l) => /All 2 tracklets/.test(l)), "and there is an All button");

  // one tracklet: the box is rewritten to its qualified id, the other designation kept
  await clickAndSettle(buttons().find((b) => /I41/.test(b.textContent)));
  assert(asked[1] === "ZZ00001@I41/2010-08-21, 2016 FV237",
         "picking one re-fits that tracklet with the other typed id: got " + asked[1]);
  assert($("desig").value === asked[1], "the search box shows what was fitted");

  // all of them
  await ask();
  await clickAndSettle(buttons().find((b) => /All/.test(b.textContent)));
  assert(asked[1] === "ZZ00001@954/2009-06-20, ZZ00001@I41/2010-08-21, 2016 FV237",
         "All re-fits every listed tracklet: got " + asked[1]);

  // the designation instead
  await ask();
  await clickAndSettle(buttons().find((b) => /Use 2022 AB1/.test(b.textContent)));
  assert(asked[1] === "2022 AB1, 2016 FV237", "Use <designation> types it in: got " + asked[1]);

  // the rewrite keeps every other typed id in place
  assert(w.TrksubPick.rewrite(["a", "b"], "a", ["x", "y"]) === "x, y, b", "rewrite keeps order");
  console.log("PASS: trksub picker");
  process.exit(0);
})().catch((e) => { console.error("FAIL:", e); process.exit(1); });
