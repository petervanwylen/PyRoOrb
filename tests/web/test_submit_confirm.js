// jsdom test for the final submit confirmation (submitconfirm.js + the
// runItfSubmit hook in app.js). What it pins down:
//   * for an ITF find, the object's OWN packed designation is listed under
//     "designation(s) the document names", NOT among the tracklets being
//     attached (the bug: window.confirm lumped them into one "Tracklets:" list,
//     so K25T66D read as an ITF tracklet of 2025 TD66);
//   * the full JSON document is shown verbatim;
//   * Cancel resolves false and POSTs nothing; Submit resolves true and POSTs
//     the session + acknowledged warnings;
//   * a user-assembled identification ("kind":"ident") still frames every id as
//     one claim.
//
//     npm i jsdom && node tests/web/test_submit_confirm.js
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
const confjs = fs.readFileSync(path.join(ROOT, "submitconfirm.js"), "utf8");

const dom = new JSDOM(html.replace(/<script[^>]*src=[^>]*><\/script>/g, ""),
                      { url: "http://localhost/", runScripts: "dangerously",
                        pretendToBeVisual: true });
const w = dom.window;
w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
w.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
w.requestAnimationFrame = (f) => setTimeout(f, 0);
w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
  get: (t, k) => (k === "measureText" ? () => ({ width: 1 }) : () => {}) });

const posts = [];      // only the submissions -- app.js also fetches settings on boot
w.fetch = async (url, opts) => {
  if (String(url).includes("itf_submit")) posts.push({ url, body: JSON.parse((opts || {}).body || "{}") });
  return { json: async () => ({ ok: true, accepted: true, status: 200, message: "thanks" }) };
};

// Queued fitting: runFit goes through FitJob, which POSTs /api/fit_start and reads
// the payload off the finished job.  Wrap the stub rather than short-circuit it, so
// whatever this test's own stub does to the payload still applies.
w.fetch = ((f) => async (url, opts) => {
  const u = String(url);
  if (u.startsWith("/api/session")) {
    const body = { ok: true, session: "s1" };
    return { json: async () => body, text: async () => JSON.stringify(body) };
  }
  if (!u.startsWith("/api/fit_start")) return f(url, opts);
  const inner = await f("/api/fit", opts);
  const job = { ok: true, job: "job1", state: "done", done: true,
                result: await inner.json() };
  return { json: async () => job, text: async () => JSON.stringify(job) };
})(w.fetch);

w.confirm = () => true;      // the per-caution click-throughs

// app.js reads every response body as TEXT and parses it (so a proxy answering
// a request it gave up on with a non-JSON page yields a readable message
// instead of "Unexpected token 'u'"). Give these mocks a text() for free.
w.fetch = ((f) => async (...a) => {
  const r = await f(...a);
  return r.text ? r : { ...r, text: async () => JSON.stringify(await r.json()) };
})(w.fetch);

for (const src of [appjs, confjs]) {
  const scr = w.document.createElement("script");
  scr.textContent = src;
  w.document.body.appendChild(scr);
}

const tick = () => new Promise((r) => setTimeout(r, 25));
const $ = (id) => w.document.getElementById(id);
const assert = (c, m) => { if (!c) { console.error("FAIL:", m); process.exit(1); } };

// the ITF find the report described: one object designation + one new tracklet
const ITF_DOC = {
  identifications: [],
  links: {
    link_0: {
      designations: ["K25T66D"],
      trksubs: [["5e85AFD", "2025 03 08.29123", "T05"],
                ["5e85AFE", "2025 03 09.31007", "T05"]],
    },
  },
};
const FIND = {
  session: "sess-1", object: "2025 TD66", designations: ["2025 TD66"],
  itf: { found: true, submittable: true, kind: "itf", neocp: false,
         warnings: ["fit rms grew from 0.31\" to 0.44\""],
         tracklets: [{ trksub: "5e85AFD", date: "2025 03 08.29123", stn: "T05" },
                     { trksub: "5e85AFE", date: "2025 03 09.31007", stn: "T05" }],
         linkage: ITF_DOC },
};

(async () => {
  await tick();
  // a remembered submitter identity (askIdentity in app.js), so the submit flow
  // goes straight to the dialog -- jsdom implements no window.prompt
  w.localStorage.setItem("itfLinkName", "J. Doe");
  w.localStorage.setItem("itfLinkEmail", "j.doe@example.org");

  // ---------- ITF find: designation vs tracklets kept apart ----------
  w.eval(`CURRENT = ${JSON.stringify(FIND)}`);
  const run = w.runItfSubmit();          // opens the dialog, awaits the click
  await tick();

  const ov = $("submitConfirmOverlay");
  assert(ov && !ov.hidden, "confirmation dialog is open");
  const body = $("submitConfirmBody");
  const items = [...body.querySelectorAll("li")].map((li) => li.textContent);
  const text = body.textContent;

  // the packed designation is present, but NOT in the tracklet list
  const trkLi = [...body.querySelectorAll("p")]
    .find((p) => /tracklet\(s\) being attached/i.test(p.textContent))
    .nextElementSibling;
  const trkItems = [...trkLi.querySelectorAll("li")].map((li) => li.textContent);
  assert(trkItems.length === 2, `two tracklets listed, got ${trkItems.length}`);
  assert(!trkItems.some((t) => t.includes("K25T66D")),
         "the object's designation must NOT be listed as a tracklet");
  assert(trkItems.some((t) => t.includes("5e85AFD (2025 03 08.29123 T05)")),
         "tracklet shows trksub + date + station");
  assert(items.includes("K25T66D"), "the packed designation is listed (as a designation)");
  assert(/Designation\(s\) the document names/i.test(text),
         "designations get their own labelled section");
  assert(/2025 TD66/.test(text), "object is named");
  assert(/quality caution/i.test(text) && /rms grew/.test(text),
         "acknowledged cautions are restated");

  // ---------- the full JSON is shown verbatim ----------
  const pre = $("submitConfirmJson");
  assert(pre, "JSON block present");
  // the server holds a header-less DRAFT; what is shown is that draft SIGNED with
  // the person's own identity -- never an empty header
  const shown = JSON.parse(pre.textContent);
  assert(JSON.stringify(shown.links) === JSON.stringify(ITF_DOC.links),
         "the JSON shown carries the document's links exactly");
  assert(shown.header && shown.header.name === "J. Doe" &&
         shown.header.email === "j.doe@example.org",
         "the JSON shown is signed with the submitter's own name / e-mail");
  assert(/\n/.test(pre.textContent), "the JSON is pretty-printed, not one line");

  // ---------- Cancel: nothing is transmitted ----------
  $("submitConfirmCancel").dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await run;
  await tick();
  assert(ov.hidden, "dialog closed on Cancel");
  assert(posts.length === 0, `Cancel must POST nothing, got ${posts.length}`);

  // ---------- Submit: POSTs session + acked warnings ----------
  w.eval(`CURRENT = ${JSON.stringify(FIND)}`);
  const run2 = w.runItfSubmit();
  await tick();
  $("submitConfirmGo").dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await run2;
  await tick();
  assert(posts.length === 1, `Submit POSTs once, got ${posts.length}`);
  assert(posts[0].url === "/api/itf_submit", "POSTs to /api/itf_submit");
  assert(posts[0].body.session === "sess-1", "POSTs the session");
  assert(posts[0].body.acked_warnings.length === 1, "echoes the acknowledged caution");

  // ---------- user-assembled identification: one-object framing ----------
  w.eval(`CURRENT = ${JSON.stringify({
    session: "sess-2", object: "2015 XX", designations: ["2015 XX", "2019 YY"],
    itf: { found: true, submittable: true, kind: "ident", warnings: [],
           linkage: { links: { link_0: { designations: ["K15X00X", "K19Y00Y"] } } } },
  })}`);
  const run3 = w.runItfSubmit();
  await tick();
  const t3 = $("submitConfirmBody").textContent;
  assert(/ONE object/.test(t3), "identification is framed as one-object claim");
  assert(/2015 XX = 2019 YY/.test(t3), "both designations named in the header");
  assert(/K15X00X/.test(t3) && /K19Y00Y/.test(t3), "both packed ids listed");
  $("submitConfirmCancel").dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await run3;

  console.log("PASS: submit confirmation shows designations, tracklets and the full JSON");
  process.exit(0);
})();
