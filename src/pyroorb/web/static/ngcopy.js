"use strict";
// PyRoOrb web UI — copy the force-model / non-gravitational block as plain text.
// Vanilla JS, no build step. Loaded AFTER app.js.
//
// The block is rendered as an HTML <table>, so the clipboard's text/plain flavour is the
// browser's table serialisation: cells joined by TAB.  Pasted into a rich-text composer
// (Gmail's, most webmail) that tab is ordinary whitespace under `white-space: normal`, so
// HTML collapsing applies -- a run containing a tab renders as ONE space, and a tab next to
// a line break collapses to nothing at all, which runs "A1 1.453e-11 ± 4.570e-11" together
// and loses every column.  Runs of literal SPACES are protected (the composer rewrites them
// as &nbsp;); tabs are not, and no setting on the receiving end fixes that.  So this button
// hands over a space-aligned rendering built from the panel's own DOM.

(function () {
  const $ = (id) => document.getElementById(id);

  // Cell text with internal whitespace normalised: the innerHTML carries newlines and
  // indentation from the template literals that built it.
  const cellText = (el) => (el.textContent || "").replace(/\s+/g, " ").trim();

  // Build the plain-text block from what is on screen (never from the payload -- the panel
  // is the thing the user is looking at, and it already resolved every "not assessed" case).
  function panelText() {
    const body = $("ngBody");
    if (!body) return "";
    const lines = [];
    const sel = $("ngSelect");
    const opt = sel && sel.selectedOptions && sel.selectedOptions[0];
    if (opt) lines.push(`force model: ${cellText(opt)}`);

    const table = body.querySelector("table.ngtab");
    if (table) {
      // Main rows are [A1, value, ±, σ]; the jackknife sub-row is one full-width
      // continuation line that belongs under the coefficient above it.
      const rows = [];
      // tbody only: the "formal ±" header cell sits over an empty first column and copies
      // as a dangling half-line -- the footnote below the table says the same thing.
      table.querySelectorAll("tbody tr").forEach((tr) => {
        if (tr.classList.contains("ngjkrow")) {
          const c = tr.querySelector("td.ngjk");
          if (c && rows.length) rows[rows.length - 1].cont.push(cellText(c));
          return;
        }
        const cells = Array.from(tr.children).map(cellText);
        if (cells.some((c) => c)) rows.push({ cells, cont: [] });
      });
      const width = [];
      rows.forEach((r) => r.cells.forEach((c, i) => {
        width[i] = Math.max(width[i] || 0, c.length);
      }));
      const indent = " ".repeat((width[0] || 0) + 2);
      rows.forEach((r) => {
        const line = r.cells
          .map((c, i) => (i === r.cells.length - 1 ? c : c.padEnd(width[i] || 0)))
          .join("  ").replace(/\s+$/, "");
        if (line) lines.push(line);
        r.cont.forEach((c) => lines.push(indent + c));
      });
    }
    // The AMR line, the footnotes, and the gravity-only note (all there is when no table
    // rendered). Document order, so AMR lands directly under the coefficients as on screen.
    body.querySelectorAll(".ngamr, .ngnote, .ngunit").forEach((n) => {
      const t = cellText(n);
      if (t) lines.push(t);
    });
    return lines.join("\n");
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);          // needs a secure context
    } catch (_) {                                          // blocked (plain http) -> legacy path
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus(); ta.select();
      try { document.execCommand("copy"); } finally { ta.remove(); }
    }
  }

  async function onCopy(ev) {
    const btn = ev.currentTarget;
    const text = panelText();
    if (!text) return;
    await copyText(text);
    const was = btn.textContent;
    btn.textContent = "✓";
    btn.classList.add("done");
    setTimeout(() => { btn.textContent = was; btn.classList.remove("done"); }, 1200);
  }

  // The header is `<div class="hd">Force model <span id="astToggle">…</span></div>` under
  // `justify-content: space-between`, so a third child would spread the toggle to the middle.
  // Group the toggle and the button on the right instead.
  function ensureBtn() {
    const hd = document.querySelector("#ngPanel > .hd");
    if (!hd || hd.querySelector(".ngcopybtn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ngcopybtn";
    btn.textContent = "⧉";
    btn.setAttribute("aria-label", "copy force model block");
    btn.title = "Copy this block as plain text (space-aligned: an HTML table pastes into " +
      "mail composers with its tabs collapsed)";
    btn.addEventListener("click", onCopy);
    const seg = hd.querySelector("#astToggle");
    if (seg && seg.parentElement === hd) {
      const wrap = document.createElement("span");
      wrap.className = "hdright";
      hd.insertBefore(wrap, seg);
      wrap.appendChild(seg);
      wrap.appendChild(btn);
    } else {
      hd.appendChild(btn);
    }
  }

  ensureBtn();
  document.addEventListener("DOMContentLoaded", ensureBtn);
  window.addEventListener("pyroorb:render", ensureBtn);
  window.ngCopyText = panelText;        // for the jsdom test
})();
