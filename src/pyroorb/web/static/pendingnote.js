"use strict";
// PyRoOrb web UI — pending-linkage verdict note. Vanilla JS, no build step. Loaded AFTER app.js.
//
// An ITF search that files its find as a POSSIBLE (one lone new night, a fragile 1.75 with a
// caution) raises no "Submit linkage" button, and a Full step / Multi-step on the SAME
// observations keeps it that way (engine_api._pending_verdict).  The button appears after a
// step in two cases, and in both the user should be told WHY a button they did not have a
// moment ago is there: observations were added or switched on since the search AND the grown
// set meets the MPC's identification criteria; or the search refused the find on OPPOSITION PROFILE ALONE
// (two new oppositions of one night each, nothing doubtful about the fit) and the user has now
// re-solved it, which offers it at their own judgement.  Conversely a set that grew but still
// fails says which rule refused it, so the next observation to look for is named -- as does a
// set that CHANGED without growing, where an observation the search judged was switched off
// (a swap is not growth).  That one is a refusal the user's own edit provoked, so
// it has to name itself; only the untouched set is silent, having been announced already.
//
// One banner per distinct verdict (keyed on the engine's reason text), so re-clicking Full
// step on an unchanged set says nothing new.  Listens on the "pyroorb:render" event.

(function () {
  let lastKey = null;

  function refresh(p) {
    const itf = p && p.itf;
    if (!itf || !itf.pending || itf.kind === "ident" || !itf.found) { lastKey = null; return; }
    const changed = /added or switched on since the search|re-solved after a step|no longer in the fit/
      .test(itf.reason || "");
    if (!changed) return;                     // the search's own verdict; already announced
    const key = `${p.session}|${itf.submittable}|${itf.reason}`;
    if (key === lastKey) return;
    lastKey = key;
    if (typeof flashBanner !== "function") return;
    flashBanner(itf.submittable
      ? `Linkage now submittable: ${itf.reason}.`
      : `Linkage still not submittable: ${itf.reason}.`,
      itf.submittable ? "info" : "warn");
  }

  document.addEventListener("pyroorb:render", (e) => refresh(e.detail));
})();
