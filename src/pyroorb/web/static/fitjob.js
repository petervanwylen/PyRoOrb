// Client half of the fit admission queue (see src/pyroorb/web/fitqueue.py).
//
// A fit is no longer computed inside the request that asks for it: the server
// queues it, so the browser starts a job and polls. Two things follow, and both
// are why this module exists rather than a one-line fetch in app.js:
//
//   * The wait is now VISIBLE. Under a crowd a fit may sit behind others, and
//     "3rd in line, about 40s" is a far better experience than a spinner with
//     no end -- it is also the honest thing to show, since that is exactly what
//     the server is doing.
//   * A fit shared by many viewers (the server coalesces identical work) hands
//     back a `restore` token instead of an interactive `session` handle, because
//     one session mutated by a hundred people would be one hundred people
//     editing each other's orbit. A private session is claimed from the token.
//
// Session expiry is RECOVERABLE for the first time as a result: the server's
// session table is small and evicts under load, but the job behind `restore`
// outlives it, so an expired session can be re-claimed instead of forcing a
// re-fit. See ensureSession().
(function () {
  "use strict";

  const POLL_FAST_MS = 600;      // while running: responsive
  const POLL_SLOW_MS = 2000;     // once it is clearly a long one: cheap
  const FAST_FOR_MS = 15000;     // how long to stay responsive before backing off

  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await r.text();
    try {
      return JSON.parse(text);
    } catch (e) {
      // A hosted proxy that gives up answers with a non-JSON body ("upstream
      // error"). Say so in words the user can act on rather than letting a
      // JSON parse error surface as gibberish.
      return { ok: false, error: "the server did not answer in a readable way " +
                                 "(" + r.status + "). Please try again." };
    }
  }

  // "3rd in line, about 40 s" -- or "fitting…" once a worker has it.
  function describe(st) {
    if (!st) return "fitting…";
    if (st.state === "queued") {
      const ahead = st.ahead || 0;
      const eta = st.eta_sec ? `, about ${Math.round(st.eta_sec)} s` : "";
      if (ahead <= 0) return `next in line${eta}`;
      return `${ahead} fit${ahead === 1 ? "" : "s"} ahead of you${eta}`;
    }
    if (st.state === "running") {
      const el = st.elapsed ? ` (${Math.round(st.elapsed)} s)` : "";
      return `fitting${el}…`;
    }
    return "fitting…";
  }

  // Poll GET /api/fit_result?id=job to completion. `onProgress(text, snapshot)`
  // fires whenever the queue picture changes; `initial` is the snapshot already
  // in hand (from /api/fit_start, or from a `pending` job handle any of the
  // one-off fit routes -- paste, add-obs, re-auto -- hand back past their own
  // inline deadline). Resolves with the fit payload, or an {ok:false} the
  // caller renders as-is.
  //
  // Factored out of run() so a job that did NOT arrive via /api/fit_start --
  // one of the one-off routes above, all of which run through the same
  // fitqueue pool and so can time out the same way -- is polled out identically
  // rather than rendered as a finished fit while it is still `pending`. That
  // was a real bug: a slow paste or re-auto answered with a job stub carrying
  // `ok: true`, and the caller rendered it as the result.
  async function pollUntilDone(job, onProgress, initial) {
    let last = "";
    const t0 = Date.now();
    let snap = initial || { state: "queued" };
    for (;;) {
      const text = describe(snap);
      if (onProgress && text !== last) { onProgress(text, snap); last = text; }
      if (snap.done && snap.result) return snap.result;
      const wait = (Date.now() - t0) < FAST_FOR_MS ? POLL_FAST_MS : POLL_SLOW_MS;
      await new Promise((res) => setTimeout(res, wait));
      const r = await fetch("/api/fit_result?id=" + encodeURIComponent(job));
      let out;
      try { out = await r.json(); } catch (e) { out = null; }
      if (!out) continue;                       // a dropped poll is not a failure
      if (out.expired) {
        return { ok: false, error: "that fit expired before it could be collected. " +
                                   "Please try again." };
      }
      snap = out;
      if (snap.done) return snap.result || { ok: false, error: "the fit returned nothing" };
    }
  }

  // Start a fit and poll it to completion. `onProgress(text, snapshot)` is
  // called whenever the queue picture changes. Resolves with the fit payload,
  // or with an {ok:false} the caller renders as-is (a refusal when the server
  // is at capacity reads as a normal error, which is what it is).
  async function run(desig, force, onProgress, fresh) {
    const started = await postJSON("/api/fit_start",
      { desig: desig, force: !!force, fresh: !!fresh });
    if (!started.ok) return started;
    return pollUntilDone(started.job, onProgress, started);
  }

  // Make sure `payload` carries a usable interactive session, claiming a
  // private one from its restore token if it does not (or no longer does).
  // Returns true when the payload is safe to interact with.
  //
  // Called both on first interaction and after a session expires: the two cases
  // are indistinguishable from here and want the same answer.
  async function ensureSession(payload, onProgress) {
    if (!payload) return false;
    if (payload.session) return true;
    if (payload.restore) {
      const out = await postJSON("/api/session", { job: payload.restore });
      if (out && out.ok && out.session) {
        payload.session = out.session;
        return true;
      }
      if (out && out.expired) payload.restore = null;  // don't retry a dead token
    }
    // No live template: this payload came off the ephemeral disk cache (or its
    // job has since been evicted). The orbit on screen is real and was cheap to
    // serve; making it INTERACTIVE is what costs a fit, so it is paid here, by
    // the one person in a hundred who actually clicks something -- which is the
    // whole reason the disk tier stores the response and not the state.
    const desig = payload.refit;
    if (!desig) return false;
    // fresh:true -- skip the disk tier.  A disk hit is what put us here, and
    // it carries no template, so asking again without this returns the same
    // cached answer and the object can never become interactive.
    const fresh = await run(desig, false, onProgress, true);
    if (!fresh || !fresh.ok) return false;
    if (fresh.session) {
      payload.session = fresh.session;
      return true;
    }
    if (fresh.restore) {
      const out2 = await postJSON("/api/session", { job: fresh.restore });
      if (out2 && out2.ok && out2.session) {
        payload.session = out2.session;
        return true;
      }
    }
    return false;
  }

  // True for the job-stub shape run_bounded() returns when a one-off fit (a
  // paste, an add-obs, a re-auto-solve) outlives its inline deadline: `ok`,
  // a `job` id, and `pending`, with no `result` yet. Exported so a caller
  // needs no knowledge of the shape beyond "is this done, or do I poll?".
  function isPending(p) {
    return !!(p && p.ok && p.pending && p.job);
  }

  // Resolve a response that MAY be a pending job stub into the finished
  // payload, leaving an already-finished one untouched.
  //
  // Every endpoint that runs engine work through the pool (a paste, an
  // add-obs, a re-auto-solve, and each of the interactive re-solves: Full
  // step, nongrav, +-, constrain, +16ast, set-epoch) answers EITHER with the
  // payload or, past the server's inline deadline, with that stub -- so every
  // caller needs the same two lines. Written once here because the failure
  // when a caller forgets is silent and confusing: the stub renders AS the
  // result, i.e. an empty fit replaces a good one on screen.
  //
  // `onProgress(text)` is optional; without it a slow job simply resolves late.
  async function settle(p, onProgress) {
    if (!isPending(p)) return p;
    return pollUntilDone(p.job, onProgress ? (t) => onProgress(t) : null, p);
  }

  window.FitJob = { run: run, pollUntilDone: pollUntilDone, isPending: isPending,
                    settle: settle, ensureSession: ensureSession, describe: describe };
})();
