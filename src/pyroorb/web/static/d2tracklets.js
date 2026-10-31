"use strict";
// PyRoOrb web UI — digest2 tracklet by tracklet, as a column of the observations table.
// Vanilla JS, no build step. Loaded AFTER app.js and d2note.js.
//
// For an orbit with q < 1.8 (payload.d2_tracklets.eligible -- the nominal
// q, or the low end of the fitting family's q when the orbit is suppressed), a "D2 per
// tracklet" toggle beside "＋ addl info" adds a column right after the station code holding
// each tracklet's digest2 NEO score on the FIRST row of that tracklet only. A tracklet is one
// station's observations under the submitter's own tracklet id (ADES trkSub) when there is
// one, else one station's observations within 4 hours of its first (d2score.tracklets). The
// scores are fetched on demand from POST /api/d2_tracklets -- an object with a thousand
// tracklets never pays for them unless asked -- and remembered per payload.
//
// Clicking the column HEADER flips raw <-> no-ID: the one flavour setting
// d2note.js owns (window.D2Flavour), so the note above the elements and this column agree.
//
// Hooked in without touching app.js: obsRowHtml (one row; used by both the full render and the
// virtual window) and renderObs (which writes the <thead>) are top-level function declarations,
// i.e. properties of the global object that app.js's own calls resolve through, so wrapping
// them here changes what app.js renders. A row gains its cell right after the code cell; the
// header gains its <th> in the same place after renderObs has written it.

(function () {
  if (typeof window.obsRowHtml !== "function" || typeof window.renderObs !== "function") return;
  const KEY = "pyroorb.d2tracklets";
  let on = false;
  try { on = localStorage.getItem(KEY) === "1"; } catch (e) { /* default off */ }
  let byFirst = null;          // first-row index -> tracklet entry, for the payload below
  let forPayload = null;       // the payload those scores belong to
  let pending = null;          // payload whose scores are being fetched
  let status = "";             // shown in the header's tooltip when the fetch failed

  const flavour = () => (window.D2Flavour ? window.D2Flavour.get() : "raw");
  const flavLabel = () => (window.D2Flavour ? window.D2Flavour.label() : "raw");
  const eligible = (p) => !!(p && p.d2_tracklets && p.d2_tracklets.eligible);
  const active = () => on && typeof CURRENT !== "undefined" && eligible(CURRENT);

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function cellFor(r) {
    const t = byFirst && byFirst.get(r.i);
    if (!t) return `<td class="d2c"></td>`;
    const noid = flavour() === "noid";
    const sc = (noid ? t.noid_scores : t.scores) || {};
    const neo = sc.NEO;
    const shown = (noid ? t.shown_noid : t.shown) || [];
    const tip = `digest2 (${noid ? "no-ID" : "raw"}) for this tracklet: ${t.n} obs from ${t.station}` +
      (shown.length ? " — " + shown.map(([a, , s]) => `${a} ${s}`).join(", ") : "");
    const v = (typeof neo === "number" && isFinite(neo)) ? Math.round(neo) : "—";
    const hot = typeof neo === "number" && neo >= 65 ? " d2hot" : "";
    return `<td class="d2c${hot}" title="${esc(tip)}">${v}</td>`;
  }

  const origRow = window.obsRowHtml;
  window.obsRowHtml = function (r, next, maxSig, chrono) {
    const html = origRow(r, next, maxSig, chrono);
    if (!active()) return html;
    return html.replace(/(<td class="l code">[^<]*<\/td>)/, `$1${cellFor(r)}`);
  };

  const origRender = window.renderObs;
  window.renderObs = function (p) {
    if (p !== forPayload) byFirst = null;      // never another payload's scores, even briefly
    origRender(p);
    syncButton(p);
    if (!active()) return;
    const tr = document.querySelector("#obsScroll table.obs thead tr");
    if (!tr || tr.querySelector("th.d2h")) return;
    const codeTh = tr.children[2];
    if (!codeTh) return;
    const th = document.createElement("th");
    th.className = "d2h";
    th.textContent = `D2 ${flavLabel()}`;
    th.title = (status ? status + " — " : "") +
      "digest2 NEO score for each tracklet (on its first row): one station's observations, " +
      "by tracklet id where the submitter gave one, else within 4 hours. Click to switch " +
      `between the raw and no-ID populations (now: ${flavLabel()}).`;
    th.style.cursor = "pointer";
    th.addEventListener("click", () => {
      if (window.D2Flavour) window.D2Flavour.set(flavour() === "noid" ? "raw" : "noid");
    });
    codeTh.insertAdjacentElement("afterend", th);
    if (forPayload !== p) fetchFor(p);
  };

  function ensureButton() {
    let b = document.getElementById("d2TrkBtn");
    if (!b) {
      const anchor = document.getElementById("addlBtn");
      if (!anchor) return null;
      b = document.createElement("button");
      b.id = "d2TrkBtn";
      b.type = "button";
      b.textContent = "D2 per tracklet";
      b.title = "Show each tracklet's digest2 NEO score in a column after the station code " +
                "(offered for q < 1.8). Click the column header to switch raw / no-ID.";
      b.hidden = true;
      anchor.insertAdjacentElement("afterend", b);
      b.addEventListener("click", () => {
        on = !on;
        try { localStorage.setItem(KEY, on ? "1" : "0"); } catch (e) { /* not remembered */ }
        if (typeof CURRENT !== "undefined" && CURRENT) window.renderObs(CURRENT);
      });
    }
    return b;
  }

  function syncButton(p) {
    const b = ensureButton();
    if (!b) return;
    b.hidden = !eligible(p);
    b.classList.toggle("active", on && eligible(p));
  }

  async function fetchFor(p) {
    if (pending === p) return;
    pending = p;
    status = "";
    try {
      if (!p.session && !p.fit_key && typeof ensureLive === "function") await ensureLive();
      const cur = (typeof CURRENT !== "undefined" && CURRENT) || p;
      const r = await fetch("/api/d2_tracklets", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session: cur.session || null, fit_key: cur.fit_key || null }),
      });
      let res = await r.json();
      if (typeof FitJob !== "undefined" && FitJob.settle) res = await FitJob.settle(res, () => {});
      if (pending !== p) return;                 // a newer payload took over
      if (!res || !res.ok) {
        status = (res && res.error) || "digest2 scores unavailable";
        byFirst = new Map();
      } else {
        byFirst = new Map((res.tracklets || []).map((t) => [t.first, t]));
        if (res.truncated) status = "only the first tracklets were scored";
      }
      forPayload = p;
    } catch (e) {
      status = `digest2 scores unavailable: ${e.message}`;
      byFirst = new Map();
      forPayload = p;
    } finally {
      if (pending === p) pending = null;
    }
    if (typeof CURRENT !== "undefined" && CURRENT) window.renderObs(CURRENT);
  }

  document.addEventListener("pyroorb:render", (e) => {
    const p = e.detail;
    if (p !== forPayload) { byFirst = null; forPayload = null; }
    syncButton(p);
  });
  document.addEventListener("pyroorb:d2flavour", () => {
    if (active() && typeof CURRENT !== "undefined" && CURRENT) window.renderObs(CURRENT);
  });
})();
