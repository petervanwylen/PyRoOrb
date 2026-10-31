"use strict";
// PyRoOrb web UI — convergence banner. Vanilla JS, no build step. Loaded AFTER app.js.
//
// The engine flags a delivered fit whose final LM stopped at its iteration cap rather
// than at a minimum (payload.converged === false, from FittedOrbit.converged): the
// elements are then a mid-iteration state and the ± sigmas are not formal
// uncertainties. This module surfaces that as a red banner in the elements panel.
//
// app.js's render() ends by dispatching "pyroorb:render" carrying the payload, and
// that is what this listens on — so a rework of how the elements panel draws its rows
// cannot silently strand the banner. Styling reuses .constraint-warn so it matches the
// unconstrained-orbit heads-up it sits under.

(function () {
  function ensureBanner() {
    let box = document.getElementById("convergedBanner");
    if (!box) {
      const anchor = document.getElementById("constraintBanner");
      if (!anchor) return null;
      box = document.createElement("div");
      box.id = "convergedBanner";
      box.className = "constraint-warn severe";
      box.hidden = true;
      anchor.insertAdjacentElement("afterend", box);
    }
    return box;
  }

  function refresh(p) {
    const box = ensureBanner();
    if (!box) return;
    const bad = !!(p && p.converged === false);
    box.hidden = !bad;
    box.innerHTML = bad
      ? '<div class="cw-head">⚠ Fit did not converge</div>' +
        '<div class="cw-detail">The final least-squares stopped at its iteration cap, ' +
        "not at a minimum: the elements are a mid-iteration state and the ± sigmas " +
        "are not formal uncertainties.</div>"
      : "";
  }

  document.addEventListener("pyroorb:render", (e) => refresh(e.detail));
})();
