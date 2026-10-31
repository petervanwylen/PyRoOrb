"use strict";
// PyRoOrb web UI — cross-listed duplicate note. Vanilla JS, no build step. Loaded AFTER app.js.
//
// The MPC republishes a NEOCP tracklet under the designation it earned (P12pwHF -> 2013 PY26)
// and cross-lists linked survey tracklets, so a combined fetch used to carry the same
// detection twice.  obs_sources.dedup_observations now keeps one copy per detection (same
// station, instant and sky position) and records what it dropped as
// payload.meta.duplicates_dropped.  This module states that under the constraint banner:
// the obs count is smaller than the sum of the designations' published lines, and the
// user should know why rather than suspect a lost tracklet.  Informational, never a warning.
//
// Listens on the "pyroorb:render" event app.js's render() dispatches (like convergence.js).

(function () {
  function ensureBox() {
    let box = document.getElementById("dedupNote");
    if (!box) {
      const anchor = document.getElementById("constraintBanner");
      if (!anchor) return null;
      box = document.createElement("div");
      box.id = "dedupNote";
      box.className = "constraint-warn";
      box.hidden = true;
      anchor.insertAdjacentElement("afterend", box);
    }
    return box;
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function refresh(p) {
    const box = ensureBox();
    if (!box) return;
    const dropped = (p && p.meta && Array.isArray(p.meta.duplicates_dropped))
      ? p.meta.duplicates_dropped : [];
    box.hidden = dropped.length === 0;
    if (!dropped.length) { box.innerHTML = ""; return; }
    const pairs = new Map();
    for (const d of dropped) {
      const k = `${d.designation} → kept as ${d.kept_designation}`;
      pairs.set(k, (pairs.get(k) || 0) + 1);
    }
    const detail = [...pairs].map(([k, n]) => `${esc(k)} (${n})`).join(", ");
    box.innerHTML =
      `<div class="cw-head">ℹ ${dropped.length} cross-listed observation${dropped.length === 1 ? "" : "s"} shown once</div>` +
      `<div class="cw-detail">The same detection (same station, instant and position) was published ` +
      `under two designations; one copy is fitted. ${detail}.</div>`;
  }

  document.addEventListener("pyroorb:render", (e) => refresh(e.detail));
})();
