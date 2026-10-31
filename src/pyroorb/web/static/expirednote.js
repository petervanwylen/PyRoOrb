"use strict";
// PyRoOrb web UI — the session-expired banner. Vanilla JS, no build step. Loaded
// AFTER app.js and noticelog.js.
//
// An interactive session (Full step, Multi-step, a model or force change, an epoch
// edit, adding observations) dies when the server evicts it or restarts, and every
// such request then answers `{ok:false, expired:true, error:"session expired -- re-fit
// the object"}`. That used to land ONLY in the small #msg line under the form, so the
// user kept clicking Full step and watching nothing change, never seeing why.
// Now it is also a STICKY orange banner -- the precedent for a caution
// that must not vanish on a timer (the ITF / linkage cautions in noticelog.js) -- with
// a Re-fit button in it.
//
// ONE banner at a time: while the previous one is still on screen, a repeat failure
// (the user clicks Multi-step next, then a nongrav toggle) adds nothing. A sticky
// banner leaves only by its ×, its Re-fit button, or a click anywhere off the banners
// (noticelog's rule) -- after which the next failure raises a fresh one, so a user who
// dismissed it and kept clicking is told again.
//
// app.js calls noteSessionExpired(p) from jsonBody on every expired answer and from
// ensureLive when a re-claim fails; the payload is inspected here so an `expired` that
// means something else (a trace id no longer held, a lost ITF job, no linkage on this
// session) does not raise a "session expired" banner for a different loss.

(function () {
  const $ = (id) => document.getElementById(id);
  // The server's own words for a dead session, across every interactive route:
  //   "session expired -- re-fit the object"            (fullstep, model, force, epoch, mask)
  //   "the current fit has expired -- re-run it ..."     (add observations)
  //   "that fit has expired -- re-run it to interact"    (a restore token the pool no longer holds)
  const SESSION_RE = /session expired|fit has expired/i;

  function noteSessionExpired(p) {
    if (!p || !p.expired) return false;
    if (p.error && !SESSION_RE.test(p.error)) return false;
    const host = $("flashHost");
    if (host && host.querySelector(".flash.expired")) return false;   // one is already up
    if (typeof flashBanner !== "function") return false;
    flashBanner("Session expired — the fit on screen can no longer be edited. Steps, model and "
                + "force changes do nothing until you re-fit the object.", "warn");
    const el = $("flashHost") && $("flashHost").lastElementChild;
    if (!el) return false;
    el.classList.add("expired", "sticky");      // .sticky: CSS cancels the auto-fade
    const act = document.createElement("button");
    act.type = "button"; act.className = "flash-act"; act.textContent = "Re-fit";
    act.addEventListener("click", (ev) => {
      ev.stopPropagation();
      el.remove();
      if (typeof runFit === "function") runFit();
    });
    el.insertBefore(act, el.querySelector(".flash-x"));
    return true;
  }

  window.noteSessionExpired = noteSessionExpired;
})();
