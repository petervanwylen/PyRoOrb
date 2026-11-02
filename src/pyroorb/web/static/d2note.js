"use strict";
// PyRoOrb web UI — digest2 class scores for a single tracklet or an undetermined orbit.
// Vanilla JS, no build step. Loaded AFTER app.js.
//
// src/pyroorb/d2score.py runs digest2 (the MPC's short-arc classifier, vendored as src/pyroorb/digest2/) on
// the fitted arc and ships BOTH flavours as payload.digest2: RAW, the full modelled population
// ("what kind of object could this be"), and NO-ID, that population minus objects already
// identifiable ("what kind of NEW object" -- the flavour the MPC posts on the NEOCP).
// Shown for every single tracklet, determined or not (it used to appear only under
// the "essentially unconstrained" banner, so a tracklet whose distance WAS pinned showed none),
// with a raw / no-ID switch. Each score is digest2's own 0-100 number for one class -- the
// classes overlap (NEO contains N22 and N18) and do not sum to 100, so they are listed, never
// drawn as shares of a whole. Display only: nothing in the engine acts on them.
//
// The flavour is ONE setting shared with the per-tracklet column (d2tracklets.js), whose header
// flips it too: window.D2Flavour.get()/.set(f), announced as the "pyroorb:d2flavour" event and
// remembered per browser (localStorage, wrapped: a private window simply starts at raw).
//
// Listens on the "pyroorb:render" event app.js's render() dispatches (like dedupnote.js).

(function () {
  const KEY = "pyroorb.d2flavour";  // a localStorage key name, not a credential  gitleaks:allow
  let flavour = "raw";
  try { if (localStorage.getItem(KEY) === "noid") flavour = "noid"; } catch (e) { /* none */ }

  window.D2Flavour = {
    get: () => flavour,
    set: (f) => {
      const next = f === "noid" ? "noid" : "raw";
      if (next === flavour) return;
      flavour = next;
      try { localStorage.setItem(KEY, flavour); } catch (e) { /* not remembered */ }
      document.dispatchEvent(new CustomEvent("pyroorb:d2flavour", { detail: flavour }));
    },
    label: (f) => (f || flavour) === "noid" ? "no-ID" : "raw",
  };

  let last = null;

  function ensureBox() {
    let box = document.getElementById("d2Note");
    if (!box) {
      const anchor = document.getElementById("constraintBanner");
      if (!anchor) return null;
      box = document.createElement("div");
      box.id = "d2Note";
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

  const TIP = "digest2 (Keys et al. 2019) sweeps every bound orbit this astrometry admits and " +
              "asks what share of the modelled Solar System population looks like each class. " +
              "RAW uses the full population; NO-ID removes objects already identifiable, which " +
              "is the flavour the MPC posts on the NEOCP (it scores the tracklet as first " +
              "submitted, so it can differ). Each class is scored on its own, 0-100; the " +
              "classes overlap and do not add up. Not a probability, and nothing in the fit " +
              "acts on it.";

  function payloadD2(p) {
    if (!p) return null;
    return p.digest2 || (p.orbit_suppressed && p.constraint && p.constraint.digest2) || null;
  }

  function refresh(p) {
    last = p;
    const box = ensureBox();
    if (!box) return;
    const d2 = payloadD2(p);
    box.hidden = !d2;
    if (!d2) { box.innerHTML = ""; return; }
    const noid = flavour === "noid" && Array.isArray(d2.shown_noid);
    const shown = noid ? d2.shown_noid : (Array.isArray(d2.shown) ? d2.shown : []);
    const cells = shown.length
      ? shown.map(([abbr, heading, score]) =>
          `<span class="d2-cls" title="${esc(heading)}">${esc(abbr)} <b>${esc(score)}</b></span>`)
          .join(" · ")
      : "no class scores above 1";
    const sw = (f, lab) => `<button type="button" class="d2-flav${flavour === f ? " active" : ""}" ` +
      `data-flav="${f}" aria-pressed="${flavour === f}">${lab}</button>`;
    box.innerHTML =
      `<div class="cw-head" title="${esc(TIP)}">ℹ What kind of object could this be? ` +
      `<span class="d2-src">digest2, ${noid ? "no-ID" : "raw"} population</span> ` +
      `<span class="d2-switch">${sw("raw", "raw")}${Array.isArray(d2.shown_noid) ? sw("noid", "no-ID") : ""}</span></div>` +
      `<div class="cw-detail">${cells}</div>`;
    box.querySelectorAll("button.d2-flav").forEach((b) =>
      b.addEventListener("click", () => window.D2Flavour.set(b.dataset.flav)));
  }

  document.addEventListener("pyroorb:render", (e) => refresh(e.detail));
  document.addEventListener("pyroorb:d2flavour", () => { if (last) refresh(last); });
})();
