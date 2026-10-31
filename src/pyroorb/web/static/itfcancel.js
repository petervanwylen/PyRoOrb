"use strict";
// PyRoOrb web UI — stop a running ITF search. Vanilla JS, no build step.
// Loaded AFTER app.js, whose runItfSearch() records the running search's job
// handle in `_itfActiveJob` and calls itfStopSync() when the search ends.
//
// A search the user no longer wants keeps a server search slot and its memory
// until it finishes on its own (minutes, for a variant scan). "stop search" (next
// to "see trace") and the trace viewer's "Stop search" button POST
// /api/itf_cancel; the server asks the search to stop, it unwinds within a
// moment, and app.js's own poll then reports it as stopped. Nothing here ends
// the poll: the slot is free only once the server says the search is over.

let _itfStopping = false;

function itfStopSync() {
  // the trace viewer's button: only while the feed on screen is the RUNNING search
  const b = document.getElementById("itfTraceStop");
  if (!b) return;
  const live = !!_itfActiveJob && typeof _traceId !== "undefined" &&
               _traceId === _itfActiveJob;
  b.hidden = !live;
  b.disabled = _itfStopping && live;
  b.textContent = _itfStopping && live ? "Stopping…" : "Stop search";
  if (!_itfActiveJob) _itfStopping = false;
}

async function stopItfSearch() {
  const job = _itfActiveJob;
  if (!job || _itfStopping) return;
  _itfStopping = true;
  const el = document.getElementById("itfElapsed");
  if (el) el.textContent = "stopping the search… ";
  const link = document.getElementById("itfStopLink");
  if (link) link.hidden = true;
  itfStopSync();
  try {
    const r = await fetch("/api/itf_cancel", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job }),
    });
    const p = await r.json();
    // finished in the meantime, or already gone: the poll reports the outcome
    if (!p.ok && !p.expired) throw new Error(p.error || "could not stop it");
  } catch (e) {
    _itfStopping = false;
    if (link) link.hidden = false;
    itfStopSync();
    if (typeof flashBanner === "function")
      flashBanner("Could not stop the ITF search: " + e.message, "alert");
  }
}

document.addEventListener("click", (e) => {
  const t = e.target;
  if (!t || (t.id !== "itfStopLink" && t.id !== "itfTraceStop")) return;
  e.preventDefault();
  stopItfSearch();
});

// every way the trace viewer opens (the "see trace" link, the toolbar's "ITF
// trace" button) goes through app.js's openItfTrace(id): re-check the button
// whenever it switches feeds
if (typeof openItfTrace === "function") {
  const _open = openItfTrace;
  openItfTrace = function (id) {   // eslint-disable-line no-global-assign
    _open(id);
    itfStopSync();
  };
}
