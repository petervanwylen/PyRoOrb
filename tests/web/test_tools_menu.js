// jsdom test for the top-bar Tools menu (static/toolsmenu.js + index.html).
//
// Guards: Random / ITF search / ITF trace live INSIDE the menu, and so does the
// marker where the server places web add-ons' links; the menu sits between Paste obs
// and Export; Submit linkage stays top-level; an acting item closes the menu while the
// random family caret keeps it open.
//
//     npm i jsdom && node tests/web/test_tools_menu.js
//
// Skips (exit 0) when jsdom is not installed.
const fs = require("fs");
const path = require("path");
let JSDOM;
try { ({ JSDOM } = require("jsdom")); }
catch { console.log("SKIP: jsdom not installed (npm i jsdom to run this test)"); process.exit(0); }

const ROOT = path.join(__dirname, "..", "..", "src", "pyroorb", "web", "static");
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8")
  .replace(/<script[\s\S]*?<\/script>/g, "");
const js = fs.readFileSync(path.join(ROOT, "toolsmenu.js"), "utf8");

let failures = 0;
const check = (name, ok, extra) => {
  console.log((ok ? "ok   " : "FAIL ") + name + (ok || !extra ? "" : "  -- " + extra));
  if (!ok) failures++;
};

function page() {
  const dom = new JSDOM(html, { url: "http://localhost/", runScripts: "outside-only" });
  const { window } = dom;
  window.fetch = () => Promise.resolve({ json: () => Promise.resolve({ ok: false }) });
  window.eval(js);
  return window;
}

const w = page();
const doc = w.document;
const menu = doc.getElementById("toolsMenu");
for (const id of ["randomBtn", "randomMenuBtn", "itfBtn", "itfTraceBtn"])
  check(`${id} is inside the Tools menu`, menu.contains(doc.getElementById(id)));
check("web add-on links land inside the Tools menu",
      menu.innerHTML.includes("<!-- web add-ons: menu -->"));
check("no add-on is named in the public page",
      !/\/neocp|neocp[a-z]*\.js|mislinkage|mislink[a-z]*\.(js|css)|identifications[a-z-]*\.html/i
        .test(html));
for (const id of ["itfSubmitBtn", "itfDownloadBtn"])
  check(`${id} stays top-level`, !menu.contains(doc.getElementById(id)));
const kids = [...doc.getElementById("toolsWrap").parentElement.children];
const pos = (el) => kids.indexOf(el);
const toolsAt = pos(doc.getElementById("toolsWrap"));
check("Tools sits right after Paste obs", toolsAt === pos(doc.getElementById("pasteBtn")) + 1);
check("Tools sits before Export", toolsAt < pos(doc.getElementById("exportBtn").parentElement));
// the Submit button lives in its split-button wrapper (#itfSubmitWrap), like Export in its own
const submitAt = pos(doc.getElementById("itfSubmitBtn").parentElement);
check("Submit linkage sits between Tools and Export",
      toolsAt < submitAt && submitAt < pos(doc.getElementById("exportBtn").parentElement));

const btn = doc.getElementById("toolsBtn");
check("closed at load", menu.hidden);
btn.click();
check("opens on click", !menu.hidden && btn.getAttribute("aria-expanded") === "true");
doc.getElementById("randomMenuBtn").click();
check("the family caret keeps it open", !menu.hidden);
doc.body.click();
check("an outside click closes it", menu.hidden);

setTimeout(() => {
  btn.click();
  const itf = doc.getElementById("itfBtn");
  itf.disabled = false;
  itf.click();
  setTimeout(() => {
    check("an acting item closes it", menu.hidden);
    process.exit(failures ? 1 : 0);
  }, 20);
}, 20);
