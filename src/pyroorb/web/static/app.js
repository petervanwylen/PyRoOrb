"use strict";
// PyRoOrb web UI — Phase 0 read-only viewer. Vanilla JS, no build step.
// Renders one FittedOrbit payload (from /api/fit) across the workspace panels.

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v))
  ? "—" : Number(v).toFixed(d);
// A close-approach distance in au: 4 decimals, or 2 significant figures under 0.001 au
// (a 0.00006 au pass must not read as 0.0001).
const fmtAu = (v) => (v === null || v === undefined || Number.isNaN(v)) ? "—"
  : (Number(v) < 0.001 ? Number(Number(v).toPrecision(2)).toString() : Number(v).toFixed(4));

// Minimal HTML escaping for free-text values (e.g. MPC observatory names) spliced
// into innerHTML / attribute strings — keeps a stray &, <, or quote from breaking markup.
const escHtml = (s) => String(s).replace(/[&<>]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const escAttr = (s) => escHtml(s).replace(/"/g, "&quot;");
// `s` ending like a sentence. The engine writes most reasons as PHRASES, to be read
// after a lead-in ("nothing shown: ", "but not cleanly — "), so none of them
// punctuates itself; a caller that then appends a second sentence runs the two
// together. Mirrors search._sentence() on the Python side, deliberately — the same
// rule at the two places that compose a reason with something else.
const sentence = (s) => {
  const t = String(s == null ? "" : s).replace(/\s+$/, "");
  return !t || /[.!?]$/.test(t) ? t : t + ".";
};

// JD (UTC) -> "YYYY-MM-DD HH:MM" via the standard Fliegel–Van Flandern conversion.
function jdToDate(jd) {
  const Z = Math.floor(jd + 0.5), F = jd + 0.5 - Z;
  let A = Z;
  if (Z >= 2299161) { const a = Math.floor((Z - 1867216.25) / 36524.25);
    A = Z + 1 + a - Math.floor(a / 4); }
  const B = A + 1524, C = Math.floor((B - 122.1) / 365.25),
        D = Math.floor(365.25 * C), E = Math.floor((B - D) / 30.6001);
  const day = B - D - Math.floor(30.6001 * E) + F;
  const month = E < 14 ? E - 1 : E - 13, year = month > 2 ? C - 4716 : C - 4715;
  const di = Math.floor(day), frac = day - di, hh = frac * 24;
  const p2 = (n) => String(Math.floor(n)).padStart(2, "0");
  return `${year}-${p2(month)}-${p2(di)} ${p2(hh)}:${p2((hh % 1) * 60)}`;
}
// Sexagesimal for DISPLAY. RA carries 0.01s of time (~0.15") and Dec 0.1" --
// one digit finer than they used to be, because these sit in the observations
// table beside residuals quoted to 0.01", and a Dec rounded to the arcsecond
// cannot be checked against them. Rounding happens ONCE, on the whole angle,
// before it is split: rounding the seconds field alone prints 59.996 as 60.00.
function sexagesimal(value, decimals) {
  const p = Math.pow(10, decimals);
  let ticks = Math.round(value * 3600 * p);          // integer at printed precision
  const dd = Math.floor(ticks / (3600 * p)); ticks -= dd * 3600 * p;
  const mm = Math.floor(ticks / (60 * p));   ticks -= mm * 60 * p;
  const w = decimals ? 3 + decimals : 2;
  return [String(dd).padStart(2, "0"), String(mm).padStart(2, "0"),
          (ticks / p).toFixed(decimals).padStart(w, "0")];
}
function raHMS(deg) {
  const [hh, mm, ss] = sexagesimal((((deg % 360) + 360) % 360) / 15, 2);
  return `${hh === "24" ? "00" : hh}:${mm}:${ss}`;    // 23:59:59.999 rounds up
}
function decDMS(deg) { const s = deg < 0 ? "−" : "+";
  const [dd, mm, ss] = sexagesimal(Math.abs(deg), 1);
  return `${s}${dd}:${mm}:${ss}`; }

let CURRENT = null;       // last payload
let _lastRenderedObj = null;  // object name of the last render() (to detect a new object)
let _pendingView = null;  // &view= read off a deep-linked URL, applied once the fit it names lands
let orbit3d = null;       // Orbit3D renderer instance
let viewStage = null;     // null = final fit; else index into history/stage_residuals (replay)
let workingMask = null;    // per-obs include flags being edited (by obs index) before a step re-fit
let obsAdded = false;      // true once obs were merged into the open fit via the paste window
                           // (shows the Re-auto-solve button); reset by a fresh fetch/paste
let ocUnit = "arcsec";     // O−C plot y-unit: "arcsec" (raw O−C) or "sigma" (normalised by per-obs σ)
// Which residual component the O−C plot draws. The colour key doubles as the switch,
// so RA-only / Dec-only costs no header room. One axis always stays on: an empty plot
// is not a view of anything.
let ocAxes = { ra: true, dec: true };
let obsView = "table";     // observations pane: "table" (residuals) or "mpc80" (MPC 80-col astrometry)
let showAddl = false;      // table view: reveal the extra scroll-right ADES columns (obs ID/trkSub/σ/logSNR)

// ---- ephemeral flash banners --------------------------------------------
// Reusable transient notice: appears at full opacity, holds ~1.8s, then fades
// out over ~1.2s and removes itself (the CSS @keyframes drives the timing).
// kind: "info"  = light-blue slider colour with black text,
//       "alert" = white-on-red, like the systematic-misfit resid warning header.
function flashBanner(message, kind = "info") {
  let host = $("flashHost");
  if (!host) {
    host = document.createElement("div");
    host.id = "flashHost"; host.className = "flashhost";
    document.body.appendChild(host);
  }
  const el = document.createElement("div");
  el.className = `flash ${kind}`;
  const msg = document.createElement("span");
  msg.className = "flash-msg"; msg.textContent = message;
  const x = document.createElement("button");           // click to dismiss early
  x.type = "button"; x.className = "flash-x"; x.textContent = "×";
  x.setAttribute("aria-label", "dismiss");
  x.addEventListener("click", () => el.remove());
  el.append(msg, x);
  el.addEventListener("animationend", () => el.remove());  // else auto-fade removal
  host.appendChild(el);
}

// One-shot banners keyed off a fresh fit payload. De-duped so re-renders of the
// same fit (stage replay, obs edits) and interactive re-fits don't re-spam.
let _astFlashKey = null, _sysFlashKey = null, _ngFlashKey = null, _ngWarnKey = null;
let _ptFlashKey = null, _ptKernelNoted = false;
const _NG_LABEL = { yarkovsky: "Yarkovsky (A2)", srp: "solar radiation pressure (A1)",
  srp2: "1/r² model, A1+A2 (SRP + Yarkovsky-type)", srp3: "1/r² model, A1+A2+A3",
  comet1: "comet out-gassing (A1, radial)",
  comet2: "comet out-gassing (A1, A2)", comet3: "comet out-gassing (A1, A2, A3)" };
// A multi-designation fit is a proposed identification, and the submit button
// appearing is not self-explanatory -- say what the engine concluded, once per
// session, and repeat each caution it attached. A 'no' verdict is deliberately
// silent: the user fitted two designations together for their own reasons and
// does not need the UI arguing with them; it just doesn't offer to submit.
let _identFlashKey = null;
function flashIdentLinkage(p) {
  const itf = p.itf;
  if (!itf || itf.kind !== "ident" || !itf.found || !p.session) return;
  if (p.session === _identFlashKey) return;
  _identFlashKey = p.session;
  const who = (p.designations || []).join(" = ") || p.object;
  const kind = itf.neocp ? "NEOCP identification" : "identification";
  // A verdict that cannot be transmitted (a 'y2' -- the join rests on excepting a tracklet, and
  // the identification document has no way to say "except these observations") must not be told
  // to "review before submitting": there is no submit button for it. Its reason carries the whole
  // instruction, in ONE banner.
  const canSend = itf.submittable !== false;
  flashBanner(itf.clean
    ? `These fit as one object — a clean ${kind} (${who}). "Submit linkage" `
      + `sends it to the MPC.`
    // sentence(), not a bare join: a 'y2' / 'y-systematic' reason is written as a
    // whole paragraph and punctuates itself, but a 'y-examine' one is a PHRASE meant
    // to be read mid-sentence ("both arcs retain >=95% of their obs") and carries no
    // full stop. And y-examine is exactly the verdict that is still submittable, so
    // the two ran together in every banner that showed the tail — "...retain >=95% of
    // their obs Review before submitting the identification."
    : `These fit as one object (${who}), but not cleanly — ${sentence(itf.reason)}`
      + (canSend ? ` Review before submitting the ${kind}.` : ""),
    itf.clean ? "info" : "warn");
  (itf.warnings || []).forEach((w) => flashBanner("⚠ Linkage caution: " + w, "warn"));
}

function flashFitBanners(p) {
  // Blue: the 16 asteroid perturbers were auto-kept because they meaningfully
  // lowered the RMS (run_fit sets asteroids_auto; manual toggles never do).
  // Keyed on the fit, not the session: a payload off the disk cache (a ?object= deeplink,
  // usually) has no session until the viewer clicks, and still deserves the banner -- once.
  const astKey = p.restore || p.refit || p.session;
  if (p.asteroids_auto && astKey && astKey !== _astFlashKey) {
    _astFlashKey = astKey;
    const ex = p.asteroid_extras || [];
    const det = new Map((p.extra_perturbers || []).map((e) => [e.number, e]));
    const named = ex.map((n) => {
      const e = det.get(n);
      return e ? `(${n}) ${e.name || ""} — Δχ² ${fmt(e.dchi2, 0)}, passed ${fmtAu(e.closest_au)} au `
        + `on ${e.closest_date}` : `(${n})`;
    });
    const one = ex.length === 1;
    flashBanner(ex.length
      ? (one ? "An asteroid perturber beyond the 16 deflects this orbit: "
             : `${ex.length} asteroid perturbers beyond the 16 deflect this orbit: `)
        + named.join("; ")
        + `. The fit now includes ${16 + ex.length} asteroidal perturbers (the 16 plus `
        + (one ? "this one" : "these") + ") — compare by turning the perturbers off and on."
      : "The 16 asteroid perturbers improved this orbit — see the difference "
        + "by turning on and off the 16 asteroid perturbers.", "info");
  }
  // Amber: the asteroid-perturber screen could not run on this fit (an error, or no
  // extra-perturber kernel installed) -- said once, so a missing piece is never silent.
  const skip = p.perturbers && p.perturbers.skipped;
  if (skip && p.session && p.session !== _ptFlashKey) {
    _ptFlashKey = p.session;
    if (/failed/.test(skip)) {
      flashBanner("The asteroid-perturber screen did not run on this fit: " + skip, "warn");
    } else if (/kernel/.test(skip) && !_ptKernelNoted) {
      _ptKernelNoted = true;
      flashBanner("The asteroid-perturber screen is not installed — run "
        + "`pyroorb-fetch-data --perturbers` on the server to enable it.", "info");
    }
  }
  // Green: an automatic non-gravitational model was detected and applied because the
  // gravity-only orbit left a systematic trend this inner, multi-apparition object.
  if (p.nongrav_auto && p.session && p.session !== _ngFlashKey) {
    _ngFlashKey = p.session;
    flashBanner("Auto-detected a non-gravitational model — "
      + (_NG_LABEL[p.nongrav_auto] || p.nongrav_auto) + " — and applied it. The gravity-only "
      + "orbit left a systematic signal that a non-gravitational term explains better "
      + "(BIC margin). Use the model dropdown to compare.",
      "info");
  }
  // Amber (long-lived): the applied non-gravitational model has coefficients that may be
  // non-physical (e.g. A1 implies a sub-rock density). Advisory ONLY — the model was still fitted
  // (we want to surface rocket bodies, dark comets and other high-area/mass objects). Keyed on
  // object+model+text so switching to a flagged model fires once and switching away clears it.
  const ngWarn = p.nongrav && p.nongrav.warning;
  const wkey = ngWarn ? `${p.object}|${p.nongrav.model}|${ngWarn}` : null;
  if (wkey && wkey !== _ngWarnKey) flashBanner(ngWarn, "warn");
  _ngWarnKey = wkey;
  // Red: the engine flagged a systematic within-apparition residual trend.
  const sys = p.summary && p.summary.systematic;
  const key = sys ? `${p.object}|${sys}` : null;
  // The engine's phrase names the apparition, axis and the two scales; the
  // consequence and the next step are said here, once, in the same words the
  // CLI uses (cli.py's WARNING line), so the two surfaces agree.
  if (key && key !== _sysFlashKey) {
    flashBanner("Residuals trend instead of scatter: " + sys + ". Nights sitting " +
      "coherently off the orbit is the usual sign of a wrong linkage or a " +
      "contaminated night. Check the O−C plot night by night before trusting " +
      "this orbit.", "alert");
  }
  _sysFlashKey = key;     // (re)set so the same flag won't re-fire, but a fresh one will
}

// Google Analytics events (static/analytics.js). Absent -- an ad-blocker, or a jsdom
// test that never loads it -- is a no-op, so nothing here depends on analytics.
function _ga(name, params) { if (typeof gaEvent === "function") gaEvent(name, params); }

async function runFit() {
  const desig = $("desig").value.trim();
  if (!desig) return;
  // `force` (brute-force linkage) is an experimental engine option, NOT exposed in the UI --
  // it pulls arcs together for experiments even when they are NOT the same object (see
  // fit_object.fit_orbit). The interactive fit always runs the honest default path.
  const force = false;
  _ga("fit_object", { source: "designation" });
  $("fitBtn").disabled = true;
  $("msg").innerHTML = `<span class="loading">fitting ${escHtml(desig)}…</span>`;
  $("workspace").hidden = true;
  resetSkyView();                    // drop the prior object's on-sky marker / ephemeris
  try {
    // Queued, not inline: the server bounds how many fits run at once, so this
    // starts a job and polls it (FitJob, static/fitjob.js). The progress text
    // reports the caller's actual place in line -- under a crowd a fit may wait,
    // and saying so is better than a spinner that explains nothing.
    const p = await FitJob.run(desig, force, (text) => {
      $("msg").innerHTML = `<span class="loading">${escHtml(desig)} — ${escHtml(text)}</span>`;
    });
    // A coalesced fit (one fit serving many viewers) arrives WITHOUT an
    // interactive session, so claim a private one before anything can edit it.
    // It costs milliseconds -- rebuilding a fit's context is cheap; finding its
    // orbit is what was expensive, and that is what was shared.
    //
    // ONLY when a restore token is there, though. A payload served from the
    // ephemeral disk cache has no live template, so claiming would mean
    // re-fitting the object -- for a viewer who has not asked to interact and
    // most likely never will. That one is deferred to ensureLive(), below.
    if (p.ok && p.restore) await FitJob.ensureSession(p);
    if (!p.ok) {
      // Rate limited: the user's own pace, not a fault in the object or the
      // server. Say which, and when they can go again.
      if (p.rate_limited) {
        $("msg").innerHTML = `<b>Slow down a moment.</b> ${escHtml(p.error || "")}`;
        $("status").innerHTML = "";
        return;
      }
      // The server is at capacity -- not a problem with this object. Say that
      // plainly instead of "could not fit", which reads as the fit having failed.
      if (p.busy) {
        $("msg").innerHTML = `<b>The server is busy.</b> ${escHtml(p.error || "")}`;
        $("status").innerHTML = "";
        return;
      }
      // Pluto is a DE440 perturber in our force model -> banned. Pop the banner and say why.
      if (p.pluto_banned) {
        flashBanner(p.banner || p.error, "alert");
        $("msg").innerHTML = `<b>Pluto is banned.</b> ${escHtml(p.banner || p.error || "")}`;
        $("status").innerHTML = "";
        return;
      }
      $("msg").innerHTML = `<b>Could not fit “${escHtml(desig)}”.</b> ${escHtml(p.error || "")}` +
        (p.hint ? `<br><span style="color:var(--muted)">${escHtml(p.hint)}</span>` : "");
      $("status").innerHTML = "";
      return;
    }
    $("msg").innerHTML = "";
    obsAdded = false;                  // fresh fetch -> nothing pasted into this fit yet
    CURRENT = p;
    render(p);
    $("workspace").hidden = false;
    pushObjectUrl(desig);              // reflect into ?object= so the link is shareable
    if (p.unknown_obscodes && p.unknown_obscodes.length) {
      flashBanner("Unrecognized observatory code(s): " + p.unknown_obscodes.join(", ") +
        " — those observations were dropped. The server re-checks the MPC station table " +
        "periodically (PYROORB_OBSCODES_REFRESH_HOURS); if the code is new, re-fit " +
        "after the next check, or restart the server with PYROORB_OBSCODES_REFRESH=1.", "alert");
    }
  } catch (e) {
    $("msg").textContent = "request failed: " + e.message;
  } finally {
    $("fitBtn").disabled = false;
  }
}

// A live interactive session for whatever is on screen, claimed on demand.
// Usually instant (a restore token is redeemed server-side in milliseconds).
// For a payload served from the ephemeral disk cache there is no template to
// redeem, so this re-fits the object -- seconds, with the queue's own progress
// text -- which is why it is called on interaction and never on arrival.
async function ensureLive() {
  if (!CURRENT) return false;
  if (CURRENT.session) return true;
  const label = $("desig").value.trim() || CURRENT.refit || "this object";
  const ok = await FitJob.ensureSession(CURRENT, (text) => {
    $("msg").innerHTML = `<span class="loading">${escHtml(label)} — ${escHtml(text)}</span>`;
  });
  $("msg").innerHTML = ok ? "" :
    `<b>That fit has expired.</b> Re-run it to keep editing.`;
  if (!ok && typeof noteSessionExpired === "function") noteSessionExpired({ expired: true });
  return ok;
}

// ---- shareable URL state -------------------------------------------------
// Reflect the fitted object into the address bar (?object=…) so the link can be copied
// and sent to someone; on page load and on back/forward we re-fit whatever it names.
function pushObjectUrl(desig) {
  const params = new URLSearchParams(location.search);
  // Replace (don't stack a new history entry) when this is the first object of the
  // session or a re-search of the one already shown; push a real entry only when the
  // object actually changes, so Back/Forward step between distinct objects.
  const replace = !params.has("object") || params.get("object") === desig;
  params.set("object", desig);
  history[replace ? "replaceState" : "pushState"](
    { object: desig }, "", location.pathname + "?" + params.toString());
}

// Reflect the central pane into the URL too (&view=…), but only away from the default
// Observations pane -- so copying the address bar while looking at Orbarium/Ephemeris/
// Impact hands back a link that opens on that same view, while a plain ?object= link
// (old, or freshly copied while on Observations) keeps opening on Observations as before.
// A replace, not a push: switching panes is not a Back/Forward-worthy navigation, it just
// refines what "here" means for the object already on screen.
function pushViewUrl(which) {
  const params = new URLSearchParams(location.search);
  if (which === "obs") {
    if (!params.has("view")) return;
    params.delete("view");
  } else {
    if (params.get("view") === which) return;
    params.set("view", which);
  }
  history.replaceState(null, "", location.pathname + (params.toString() ? "?" + params.toString() : ""));
}

// A pasted-obs / add-obs fit can't be reproduced from a URL (the server doesn't have the
// astrometry), so drop any stale ?object= rather than mislabel the link.
function clearObjectUrl() {
  const params = new URLSearchParams(location.search);
  if (!params.has("object")) return;
  params.delete("object");
  const q = params.toString();
  history.replaceState(null, "", location.pathname + (q ? "?" + q : ""));
}

// Fit whatever object the current URL names (initial load + Back/Forward). Idempotent on
// the URL: runFit's pushObjectUrl sees the param already set to this value and replaces
// in place, so no extra history entry is created.
function fitFromUrl() {
  const params = new URLSearchParams(location.search);
  const obj = params.get("object");
  if (!obj) return;
  _pendingView = params.get("view");   // consumed once by render()'s newObj branch, below
  $("desig").value = obj;
  runFit();
}

// The user's CURRENT inclusion set (their edits if any, else the fit's own), by obs
// index -- sent with every interactive re-fit so toggles are never silently dropped.
function currentMask() {
  const n = CURRENT.residuals.length;
  const mask = new Array(n).fill(true);
  CURRENT.residuals.forEach((r) => { mask[r.i] = workingMask ? workingMask[r.i] : r.included; });
  return mask;
}

// "Submit linkage" must not fire on an orbit that no longer reflects the
// observation set on screen.  Toggling observations changes WHAT would be
// submitted (the document is built from the included set), but the ORBIT is
// still the one solved before the toggle -- so the linkage would be sent on
// evidence nobody has re-checked.  Disable until a Full step (or Multi-step)
// re-solves on the current set; render() resets the working mask from the fit,
// so a step clears this by itself.
function refreshSubmitGate() {
  const stale = !!(CURRENT && workingMask && (CURRENT.residuals || [])
    .some((r) => workingMask[r.i] !== r.included));
  for (const id of ["itfSubmitBtn", "itfDownloadBtn", "itfDirectBtn"]) {
    const b = $(id);
    if (!b || b.hidden) continue;
    if (b.dataset.title0 === undefined) b.dataset.title0 = b.title || "";
    b.disabled = stale;
    b.title = stale
      ? "Observations were switched on or off since the last solve — run a "
        + "Full step (or Multi-step) first, so the orbit matches the linkage "
        + "you would submit."
      : b.dataset.title0;
  }
}

// Differential correction on the current edited inclusion set. single=true is the
// "Full step" button -- ONE damped correction step (find_orb's `f`; click repeatedly, an
// unchanged orbit means converged). single=false is "Multi-step" -- re-solve to convergence.
async function runStep(single) {
  if (!CURRENT || !(CURRENT.session || CURRENT.restore || CURRENT.refit)) return;
  if (!(await ensureLive())) return;
  const mask = currentMask();
  const name = single ? "Full step" : "Multi-step";
  const btn = $(single ? "fullStepBtn" : "multiStepBtn"), label = btn.textContent;
  btn.disabled = true; btn.textContent = single ? "stepping…" : "fitting…";
  try {
    const r = await fetch("/api/fullstep", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: CURRENT.session, mask, single: !!single }),
    });
    // A step runs through the fit pool (server.py:_pooled), so past its inline
    // deadline it answers with a job stub to poll rather than the stepped fit.
    const p = await FitJob.settle(await jsonBody(r, name), (t) => {
      $("msg").innerHTML = `<span class="loading">${escHtml(t)}</span>`;
    });
    if (!p.ok) {
      $("msg").innerHTML = `<b>${escHtml(name)} failed.</b> ${escHtml(p.error || "")}` +
        (p.expired ? ` <a href="#" onclick="runFit();return false">re-fit</a>` : "");
      return;
    }
    $("msg").innerHTML = "";
    CURRENT = p; render(p); $("workspace").hidden = false;
  } catch (e) {
    $("msg").textContent = `${name} failed: ` + e.message;
  } finally {
    btn.disabled = false; btn.textContent = label;
  }
}

// Re-run the FULL automatic pipeline (start orbit, outlier rejection, model discovery) on
// the open fit's observation set -- as if freshly pasted. Offered only after observations
// were added via the paste window; discards manual include/exclude edits (the engine's own
// auto-rejection decides the inclusion set afresh) and returns a brand-new session.
async function runReAuto() {
  if (!CURRENT || !(CURRENT.session || CURRENT.fit_key)) return;
  const btn = $("reAutoBtn"), label = btn.textContent;
  btn.disabled = true; btn.textContent = "solving…";
  $("msg").innerHTML = `<span class="loading">re-running the automatic pipeline…</span>`;
  try {
    const r = await fetch("/api/reauto", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: CURRENT.session || null,
                             fit_key: CURRENT.fit_key || null }),
    });
    // Re-auto-solve is a full fit and runs through the same admission-controlled
    // pool as /api/fit (src/pyroorb/web/fitqueue.py:run_bounded); past its inline
    // deadline it hands back a job stub (ok:true, pending:true) rather than a
    // result. FitJob.settle polls it out -- without that the stub itself used
    // to be rendered as the re-solved fit.
    const p = await FitJob.settle(await jsonBody(r, "Re-auto-solve"), (t) => {
      $("msg").innerHTML = `<span class="loading">${escHtml(t)}</span>`;
    });
    if (!p.ok) {
      $("msg").innerHTML = `<b>Re-auto-solve failed.</b> ${escHtml(p.error || "")}` +
        (p.expired ? ` <a href="#" onclick="runFit();return false">re-fit</a>` : "");
      return;
    }
    $("msg").innerHTML = "";
    CURRENT = p; render(p); $("workspace").hidden = false;
  } catch (e) {
    $("msg").textContent = "re-auto-solve failed: " + e.message;
  } finally {
    btn.disabled = false; btn.textContent = label;
  }
}

// The ITF options modal shows the ENGINE's defaults (GET /api/itf_defaults ->
// ItfConfig), fetched once per page load and refreshed into any field the user
// has not touched. Carrying them as literals in index.html is what silently
// broke the search: the corridor multiplier moved 1.1 -> 2.5 in the engine, the
// input went on sending 1.1, and every web search ran a 2.3x too-narrow
// corridor -- so the UI missed tracklets the CLI found (2026 DR42 / P10OAL0)
// with no error anywhere. The HTML values are only the offline fallback now.
let _itfDefaultsDone = false;
const _itfTouched = new Set();
for (const id of ["itfMaxUnc", "itfUncMult", "itfMagLimit", "itfRankMode",
                  "itfCandidateSource", "itfTraceLevel"]) {
  const el = $(id);
  if (el) el.addEventListener("input", () => _itfTouched.add(id));
}
async function _itfLoadDefaults() {
  if (_itfDefaultsDone) return;
  _itfDefaultsDone = true;              // one attempt; the HTML values stand in
  try {
    const r = await fetch("/api/itf_defaults");
    const p = await r.json();
    if (!p || !p.ok) return;
    const set = (id, v) => {
      if (v === undefined || v === null || _itfTouched.has(id)) return;
      $(id).value = v;
    };
    set("itfMaxUnc", p.max_uncert_arcsec);
    set("itfUncMult", p.uncert_mult);
    // the faint cut stays EMPTY: an untyped field is what lets the engine
    // deepen its default per object, and a typed one (even the default value)
    // is applied exactly -- so the default is shown, never filled in
    if (p.mag_limit !== undefined && p.mag_limit !== null)
      $("itfMagLimit").placeholder = `default is ${p.mag_limit}`;
    set("itfRankMode", p.rank_mode);
    set("itfCandidateSource", p.candidate_source);
    set("itfTraceLevel", p.trace_level);
    if ($("itfTraceLevelLive") && !_itfTouched.has("itfTraceLevel"))
      $("itfTraceLevelLive").value = p.trace_level;
  } catch (e) { /* offline / old server: keep the HTML fallback values */ }
}

// Read a JSON response, or fail with a message a human can act on. A hosted
// deploy sits behind a reverse proxy that answers a request it gave up on with
// a PLAIN-TEXT body -- Railway sends "upstream error" -- and response.json()
// turned that into the useless "Unexpected token 'u'". Long work is polled
// rather than awaited now (see runItfSearch), so this is the backstop for the
// short requests: say who failed, and show what they actually sent.
async function jsonBody(r, what) {
  const text = await r.text();
  let p;
  try {
    p = JSON.parse(text);
  } catch (e) {
    const snip = text.trim().slice(0, 120) || `HTTP ${r.status}`;
    throw new Error(`${what}: the server did not answer with JSON ` +
                    `(HTTP ${r.status}: ${snip})`);
  }
  // A dead interactive session answers every step with {expired:true}; the sticky
  // banner (expirednote.js) is what the user sees, the #msg line being easy to miss.
  if (p && p.expired && typeof noteSessionExpired === "function") noteSessionExpired(p);
  return p;
}

// ITF search: hunt the MPC Isolated Tracklet File for unlinked observations of the
// open object (src/pyroorb/itf/). Long-running (~30-120 s live); on a found linkage the
// server returns a fresh full fit payload for the combined seed+ITF orbit, which
// replaces the view exactly like a new fit; otherwise a transient banner reports
// "no additional found". Never submits to the MPC.
async function runItfSearch() {
  // The same handles every other interactive control accepts: a fit served from
  // the disk cache carries only `refit`, a coalesced one only `restore`, and
  // ensureLive() turns either into a session. Gating on session || fit_key alone
  // made "Search ITF" a silent no-op on any disk-cached object (2020 BB167).
  // A pasted fit's `fit_key` is resolved by /api/itf_search itself.
  if (!CURRENT || !(CURRENT.session || CURRENT.fit_key ||
                    CURRENT.restore || CURRENT.refit)) return;
  _setHidden($("itfOverlay"), false);
  const btn = $("itfBtn"), label = btn.textContent;
  btn.disabled = true; btn.textContent = "searching…";
  if (!CURRENT.session && !CURRENT.fit_key && !(await ensureLive())) {
    btn.disabled = false; btn.textContent = label;
    return;
  }
  // every search gets a fresh trace id so "see trace" can stream the live
  // decision feed (GET /api/itf_trace) while the POST is still running
  const traceId = "tr" + Math.random().toString(36).slice(2, 10) +
                  Date.now().toString(36);
  // the trace stays reachable after the "see trace" link's msg line clears:
  // the toolbar "ITF trace" button reopens the most recent feed (and a page
  // reload restores it via sessionStorage while the server still holds it)
  _lastTraceId = traceId;
  // the viewer's live "Detail" control starts where this search starts, so it
  // shows the truth rather than a stale value from the last search
  if ($("itfTraceLevelLive")) $("itfTraceLevelLive").value = $("itfTraceLevel").value;
  try { sessionStorage.setItem("itfTraceId", traceId); } catch (e) {}
  _setHidden($("itfTraceBtn"), true);
  // the server's job handle IS the trace id; itfcancel.js stops it by this
  _itfActiveJob = traceId;
  $("msg").innerHTML =
    `<span class="loading" id="itfElapsed">searching the ITF ` +
    `(typically 30–120 s)… </span>` +
    `<a href="#" id="itfTraceLink">see trace</a> · ` +
    `<a href="#" id="itfStopLink" title="Stop this search and free its ` +
    `server resources">stop search</a>`;
  $("itfTraceLink").addEventListener("click",
    (e) => { e.preventDefault(); openItfTrace(traceId); });
  const itfT0 = Date.now();
  let itfOutcome = "error", itfFound = null;    // -> itf_search_complete, sent once in finally
  _ga("initiate_itf_search", { mag_limit: parseFloat($("itfMagLimit").value) || null,
                               rank_mode: $("itfRankMode").value });
  try {
    const r = await fetch("/api/itf_search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session: CURRENT.session || null,
        fit_key: CURRENT.fit_key || null,
        mask: currentMask(),
        // start it on a worker thread and poll: an ITF search runs longer than
        // a hosted proxy will hold one request open, and a proxy that gives up
        // answers with a non-JSON body ("upstream error") while the search
        // itself is still running happily. The job handle IS the trace id.
        async: true,
        options: {
          max_uncert_arcsec: parseFloat($("itfMaxUnc").value) || null,
          uncert_mult: parseFloat($("itfUncMult").value) || null,
          mag_limit: parseFloat($("itfMagLimit").value) || null,
          rank_mode: $("itfRankMode").value,
          candidate_source: $("itfCandidateSource").value,
          jitter: !!$("itfJitter").value,
          trace_id: traceId,
          trace_level: parseInt($("itfTraceLevel").value, 10) || null,
        },
      }),
    });
    const started = await jsonBody(r, "ITF search");
    // async:true -> {ok, job}: collect the payload by polling. An older server
    // (no job handle) answered the POST with the payload itself; take it as-is.
    let p = started && started.job
      ? await awaitItfResult(started.job) : started;
    if (!p.ok) {
      // A BUSY answer is not a failure: the server refused to start a seventh
      // simultaneous search so the six already running keep their memory. Say
      // that plainly -- "failed" would send the user hunting for a fault in
      // their object, and the remedy is simply to wait.
      itfOutcome = p.busy ? "busy" : p.cancelled ? "cancelled" : "failed";
      if (p.cancelled) {
        // the user's own Stop: not a failure, and nothing to re-fit
        $("msg").innerHTML = "<b>ITF search stopped.</b>";
        flashBanner("ITF search stopped — its server resources are freed.", "info");
        return;
      }
      if (p.busy) {
        $("msg").innerHTML = `<b>Server busy.</b> ${escHtml(p.error || "")}`;
        flashBanner(`ITF search queued out: ${p.error ||
          "too many searches running -- try again in a minute"}`, "warn");
        return;
      }
      $("msg").innerHTML = `<b>ITF search failed.</b> ${escHtml(p.error || "")}` +
        (p.expired ? ` <a href="#" onclick="runFit();return false">re-fit</a>` : "");
      flashBanner(`ITF search failed: ${p.error || "unknown error"}`, "alert");
      return;
    }
    $("msg").innerHTML = "";
    if (!p.itf || !p.itf.found) {
      const why = p.itf && p.itf.reason ? p.itf.reason : "";
      itfOutcome = p.itf && p.itf.suppressed ? "suppressed" : "not_found";
      if (p.itf && p.itf.suppressed) {
        // A tracklet DID link -- the search declined to show it, and the
        // reason says which check refused. Reporting that as "nothing found"
        // was the misleading part; it is a caution, not an all-clear.
        flashBanner(`ITF search: a candidate tracklet linked but is not being ` +
          `shown${why ? " \u2014 " + why.replace(/^nothing shown: /, "")
            .replace(/ -- /g, " \u2014 ") : ""}`,
          "warn");
      } else {
        flashBanner(`ITF search: no additional observations found` +
          `${why ? ` (${why})` : ""}`, "info");
      }
      announceIncompleteSweep(p.itf);
      return;
    }
    // two different results (a submittable linkage and a fuller unconfirmed
    // one): the user picks which is shown (itfchoice.js)
    if (p.itf.alternative && typeof chooseItfResult === "function") {
      p = await chooseItfResult(p);
    }
    itfOutcome = p.itf.possible ? "possible" : "linked";
    itfFound = p.itf;
    CURRENT = p; render(p); $("workspace").hidden = false;
    const trks = (p.itf.tracklets || [])
      .map((t) => `${t.trksub} (${t.date} ${t.stn}, ${t.n_obs} obs)`).join(", ");
    if (p.itf.possible) {
      // thin-profile candidate (more than one apparition, but one of them a
      // lone night): shown for review, never
      // submittable. Fold the FIRST warning (the possible-narrative caution:
      // single-best-fit / candidate-fits) into this headline banner (216f3fd's
      // dedup), but render any ADDITIONAL warnings as their OWN separate,
      // same-color banners -- e.g. the bounded-multiplicity "not necessarily the
      // only match" caveat, which the user should read and dismiss on its own.
      const ws = p.itf.warnings || [];
      const lead = ws.length ? " " + ws[0] : " Not submittable.";
      flashBanner(`ITF search found a POSSIBLE (unconfirmed) linkage — ` +
        `${p.itf.tracklets.length} tracklet(s), +${p.itf.added_obs} obs: ` +
        `${trks}.${lead}`, "warn");
      ws.slice(1).forEach((w) => flashBanner("⚠ ITF caution: " + w, "warn"));
    } else {
      flashBanner(`ITF search linked ${p.itf.tracklets.length} tracklet(s), ` +
        `+${p.itf.added_obs} obs: ${trks}`, "info");
      // caution banners for a flagged extension (fit degraded / mag off /
      // systematic offset) -- the same flags that gate the submit click-through
      (p.itf.warnings || []).forEach((w) =>
        flashBanner("⚠ ITF caution: " + w, "warn"));
    }
    announceIncompleteSweep(p.itf);
  } catch (e) {
    $("msg").textContent = "ITF search failed: " + e.message;
    flashBanner("ITF search failed: " + e.message, "alert");
  } finally {
    _ga("itf_search_complete", {
      outcome: itfOutcome,
      n_tracklets: itfFound ? (itfFound.tracklets || []).length : 0,
      added_obs: itfFound ? (itfFound.added_obs || 0) : 0,
      elapsed_sec: Math.round((Date.now() - itfT0) / 1000),
    });
    _itfActiveJob = null;
    if (typeof itfStopSync === "function") itfStopSync();
    btn.disabled = false; btn.textContent = label;
  }
}

// Part of the corridor was never queried -- announce it whatever the verdict.
// On a no-match this is the difference between "we looked and it is not there"
// and "we ran out of room before we finished looking", which is not a detail:
// the second is not evidence of absence. On a find it says the result may be
// incomplete rather than wrong. Never an "alert": nothing is broken, the
// server was full, and the user's remedy is to re-run when it is quieter.
function announceIncompleteSweep(itf) {
  // "the server was tight, but we finished" -- a different message from "we
  // stopped early", and the user needs it to decide whether re-running later
  // is worth it. Only one of the two ever fires.
  const con = itf && itf.constrained;
  if (con) {
    flashBanner(`\u26a0 The server is low on memory, so this search ran less ` +
      `deeply than usual (${con.reason}). It finished \u2014 but re-running ` +
      `when the server is quieter may search further.`, "warn");
  }
  // Advisory measurements about the linkage that are NOT doubts about it --
  // currently the residual trend, whose gate was withdrawn once the full
  // calibration showed it firing on correct tracklets about as often as on
  // wrong ones. Rendered "info", never "warn": it does not gate the submit
  // click-through, is never acknowledged, and must not look like the cautions
  // that do.
  for (const d of (itf && itf.diagnostics) || []) {
    // A caution the orbit fitted WITHOUT the flagged data still predicts
    // (itf/leaveout.py): it explains itself, and the trend disclaimer below
    // would misdescribe it.
    if (d.startsWith("confirmed by leave-out:")) {
      flashBanner(`\u2139 For information only \u2014 ${d}.`, "info");
      continue;
    }
    flashBanner(`\u2139 For information only: ${d}. The test misfires about as ` +
      `often as it catches a bad tracklet, so it does not count against the ` +
      `linkage. Worth a glance at that tracklet's O\u2212C column.`, "info");
  }
  const inc = itf && itf.incomplete;
  if (!inc) return;
  const lead = inc.verdict_affected
    ? "this is NOT a clean \u201cnot found\u201d"
    : "more may be there that was never checked";
  flashBanner(`\u26a0 Search was INCOMPLETE: ${inc.windows} sky window(s) were ` +
    `never searched because ${inc.why} \u2014 ${lead}. ` +
    `Re-running when the server is quieter may find more.`, "warn");
}

// Collect a backgrounded ITF search: poll GET /api/itf_result until the worker
// thread finishes and hand back the same payload the blocking POST used to
// return. Each poll is a short request, so no proxy request limit is in play
// however long the search takes -- that is the whole point of the split.
// A transient poll failure (a proxy hiccup, a dropped connection) is retried,
// NOT reported: the search is still running server-side. Only the server
// forgetting the job (a restart, or the TTL) ends the wait.
async function awaitItfResult(job) {
  // ramped: the first polls are quick so a search that fails in a second (an
  // expired session, an unreachable database) still answers like one, then
  // settle to one poll every 2 s for the long haul
  const EVERY = [200, 400, 800, 1500, 2000];
  for (let n = 0; ; n++) {
    await new Promise((res) =>
      setTimeout(res, EVERY[Math.min(n, EVERY.length - 1)]));
    let p = null;
    try {
      p = await jsonBody(await fetch(
        `/api/itf_result?id=${encodeURIComponent(job)}`), "ITF search");
    } catch (e) { continue; }              // transient -> keep waiting
    if (p.ok && p.done) return p.result;
    if (p.expired) {
      return { ok: false, error: "the search was lost server-side (the server " +
                                 "restarted, or the result expired)" };
    }
    if (p.ok && !p.done) {
      const el = $("itfElapsed");
      if (el) {
        el.textContent = p.cancelling
          ? `stopping the search… `
          : `searching the ITF… ${Math.round(p.elapsed || 0)} s ` +
            `elapsed (typically 30–120 s) `;
      }
    }
  }
}

// Live ITF-search trace: poll /api/itf_trace for the running search's decision
// feed and stream it into the trace overlay (opened from the "see trace" link).
// Polling runs only while the overlay is visible; closing it pauses the feed
// and re-opening resumes from the same cursor.
let _traceId = null, _traceNext = 0, _traceTimer = null, _lastTraceId = null;
// the running search's job handle (null when none is running) -- itfcancel.js
let _itfActiveJob = null;
function openItfTrace(id) {
  if (_traceId !== id) {              // a new search -> reset the scrollback
    _traceId = id; _traceNext = 0;
    $("itfTraceText").textContent = "";
    $("itfTraceStatus").textContent =
      "live — the search is narrating every decision as it makes it";
  }
  _setHidden($("itfTraceOverlay"), true);
  _itfTracePoll();
}
async function _itfTracePoll() {
  if (_traceTimer) { clearTimeout(_traceTimer); _traceTimer = null; }
  if (!_traceId || $("itfTraceOverlay").hidden) return;   // closed -> pause
  let done = false;
  try {
    // the viewer's Detail control rides along on the poll: it re-levels the
    // RUNNING search from here on (the level it STARTED at is picked in the ITF
    // search options -- this one cannot rewrite lines already emitted)
    const lvlEl = $("itfTraceLevelLive");
    const lvl = lvlEl ? parseInt(lvlEl.value, 10) : NaN;
    const r = await fetch(`/api/itf_trace?id=${encodeURIComponent(_traceId)}` +
                          `&from=${_traceNext}` +
                          (lvl ? `&level=${lvl}` : ""));
    const p = await r.json();
    if (p.ok) {
      if (p.lines && p.lines.length) {
        const el = $("itfTraceText");
        // autoscroll only while the user is already at the bottom
        const stick = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
        el.textContent += p.lines.join("\n") + "\n";
        _traceNext = p.next;
        if (stick) el.scrollTop = el.scrollHeight;
      }
      done = !!p.done;
    } else if (p.expired) {
      // the server evicted this feed -- give up rather than poll forever
      $("itfTraceStatus").textContent =
        "trace no longer available (expired on the server)";
      return;
    }
  } catch (e) { /* transient poll failure -> just retry */ }
  if (done) {
    $("itfTraceStatus").textContent = "search finished — full trace below";
    return;
  }
  _traceTimer = setTimeout(_itfTracePoll, 400);
}
// changing the level while watching: push it at once rather than waiting out
// the poll interval, so the feed visibly answers the control
if ($("itfTraceLevelLive"))
  $("itfTraceLevelLive").addEventListener("change", () => _itfTracePoll());

// The submitter identity for every linkage route (queue, MPC, file): the name and
// e-mail this browser remembers (localStorage itfLinkName / itfLinkEmail). There is
// NO built-in submitter -- the header is always the person's own. Asked only when
// nothing is remembered yet, or when `force` (the menu's "Change name / e-mail…").
// The e-mail rule is the server's own (itf.linkage.header_problem); storage may throw
// (private mode, blocked site data), which must not break submitting.
const ID_EMAIL = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
function idGet(k) { try { return localStorage.getItem(k) || ""; } catch (e) { return ""; } }
function idSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* not remembered */ } }
function askIdentity(force) {
  let name = idGet("itfLinkName");
  let email = idGet("itfLinkEmail");
  if (!force && name && ID_EMAIL.test(email)) return { name, email };
  const n = window.prompt(
    'Your name for the identification header -- first initial + last name\n' +
    '(e.g. "J. Smith"):', name);
  if (n === null) return null;
  if (!n.trim()) { flashBanner("A name is required to submit a linkage.", "alert"); return null; }
  const e = window.prompt(
    "Your e-mail address (the MPC credits and replies to it):", email);
  if (e === null) return null;
  if (!ID_EMAIL.test(e.trim())) {
    flashBanner("A valid e-mail address is required to submit a linkage.", "alert"); return null;
  }
  name = n.trim(); email = e.trim();
  idSet("itfLinkName", name);
  idSet("itfLinkEmail", email);
  return { name, email };
}

// Submit the linkage on screen -- an ITF-search find, or the identification
// implied by a multi-designation fit the user assembled. route "queue" (the
// main button) sends it to the PyRoOrb identification queue, which checks it,
// paces it and forwards it to the MPC; route "mpc" (menu, only where the server
// sets DIRECT_SUBMIT_ALLOWED) sends it straight to the MPC. This is the one
// outward, irreversible action in the UI, so it confirms first (naming every id
// being claimed as one object), then POSTs only the session + identity -- the
// server holds the linkage and rebuilds the document; the client never sends a
// submittable document.
async function runItfSubmit(route) {
  route = route === "mpc" ? "mpc" : "queue";
  if (!CURRENT || !CURRENT.session || !(CURRENT.itf && CURRENT.itf.found)) return;
  const me = askIdentity(false);
  if (!me) return;
  const who = CURRENT.itf.kind === "ident"
    ? (CURRENT.designations || [CURRENT.object]).join(" = ")
    : ((CURRENT.designations && CURRENT.designations[0]) || CURRENT.object);
  const warns = (CURRENT.itf && CURRENT.itf.warnings) || [];
  // Each quality caution is its own click-through BEFORE the final confirm, so a
  // flagged extension (fit degraded, magnitude off, coherent offset) cannot be
  // submitted on a single reflexive click.
  for (let i = 0; i < warns.length; i++) {
    if (!window.confirm(
        `⚠ Caution ${i + 1} of ${warns.length} before submitting ${who}:\n\n` +
        `${warns[i]}\n\n` +
        `This is a sign the linkage may be WRONG. Continue anyway?`)) return;
  }
  // Final confirmation (submitconfirm.js): names the object, keeps the
  // designations the document carries apart from the tracklets being attached
  // to them, and shows the JSON document itself verbatim.
  if (!(await confirmLinkageSubmit(CURRENT, warns, route, me))) return;
  const btn = $("itfSubmitBtn"), label = btn.textContent;
  btn.disabled = true; btn.textContent = "submitting…";
  $("msg").innerHTML = `<span class="loading">${route === "mpc"
    ? "submitting the linkage to the MPC…"
    : "sending the linkage to the PyRoOrb identification queue…"}</span>`;
  try {
    // echo back each caution the user clicked through -- the server refuses
    // a submission whose stored warnings were not all acknowledged
    // pooled server-side (a Submit may refit): past the inline deadline the reply
    // is a job stub, settled here like every other pooled call
    const post = async (acked) => FitJob.settle(await jsonBody(await fetch("/api/itf_submit", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: CURRENT.session, acked_warnings: acked, route,
                             name: me.name, email: me.email }),
    }), "Linkage submission"));
    let p = await post(warns);
    // A caution the search set aside because the orbit fitted WITHOUT that data
    // predicted it (itf/leaveout.py) is re-judged on the observations being
    // submitted when the user has switched some on or off since the search; one
    // that no longer stands comes back here as an ordinary click-through.
    if (!p.ok && p.rejudged && p.rejudged.length) {
      const back = p.rejudged;
      for (let i = 0; i < back.length; i++) {
        if (!window.confirm(
            `⚠ Re-checked caution ${i + 1} of ${back.length} before submitting ${who}:\n\n` +
            `${back[i]}\n\n` +
            `The search had set this aside because the orbit fitted without that data ` +
            `still predicted it. With the observations you changed since, it no longer ` +
            `does. This is a sign the linkage may be WRONG. Continue anyway?`)) {
          $("msg").innerHTML = "";
          btn.disabled = false; btn.textContent = label;
          return;
        }
      }
      p = await post(warns.concat(back));
    }
    $("msg").innerHTML = "";
    if (!p.ok) {
      flashBanner(`Linkage submission failed: ${p.error || "unknown error"}`, "alert");
      btn.disabled = false; btn.textContent = label;
      return;
    }
    _ga("submit_linkage", { accepted: !!p.accepted, route });
    if (p.accepted) {
      // one-shot: the server cleared the pending linkage; reflect it done
      if (CURRENT.itf) CURRENT.itf.submittable = false;
      _setHidden($("itfSubmitWrap"), false);
      if (p.queued) {
        flashBanner(`Linkage queued as submission ${p.queued.document} — it is checked, ` +
          `deduplicated and forwarded to the MPC from the PyRoOrb queue. Follow it on ` +
          `${p.queued.dashboard} (private status: ${p.queued.status_url}).`, "info");
      } else {
        // show the MPC's own acknowledgement text when it sent one ("Thank you
        // for submitting your identifications. ..."), not a bare "200 OK"
        flashBanner(p.message || `Linkage submitted to the MPC (${p.status}).`, "info");
      }
    } else {
      flashBanner(`MPC did not accept the submission (${p.status || "no 200"}).`, "alert");
      btn.disabled = false; btn.textContent = label;
    }
  } catch (e) {
    $("msg").innerHTML = "";
    flashBanner("Linkage submission failed: " + e.message, "alert");
    btn.disabled = false; btn.textContent = label;
  }
}

// Download the linkage on screen as an MPC identification JSON file (the submit
// menu's "Download linkage file"): the document the queue or the MPC would get,
// with the remembered name / e-mail in its header, saved client-side. Nothing is
// transmitted to anyone.
async function runItfDownload() {
  if (!CURRENT || !CURRENT.session || !(CURRENT.itf && CURRENT.itf.found)) return;
  const me = askIdentity(false);
  if (!me) return;
  const btn = $("itfDownloadBtn"), label = btn.textContent;
  btn.disabled = true; btn.textContent = "building…";
  try {
    const r = await fetch("/api/itf_download", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: CURRENT.session, name: me.name, email: me.email }),
    });
    const p = await r.json();
    if (!p.ok) {
      flashBanner(`Linkage download failed: ${p.error || "unknown error"}` +
        (p.expired ? " -- re-run the ITF search" : ""), "alert");
      return;
    }
    const blob = new Blob([JSON.stringify(p.document, null, 2) + "\n"],
                          { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = p.filename || "itf_linkage.json";
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(a.href);
    _ga("download_linkage_file", {});
    flashBanner(`Linkage file ${p.filename} downloaded.` +
      (p.note ? ` (${p.note})` : ""), "info");
  } catch (e) {
    flashBanner("Linkage download failed: " + e.message, "alert");
  } finally {
    btn.disabled = false; btn.textContent = label;
  }
}

// Re-anchor the osculating elements to a user-picked epoch (find_orb's "set epoch").
// Integrates the orbit to the chosen date and shows the new elements there; the
// residuals don't change until the next Full step, and that step runs from this epoch.
async function setEpoch() {
  if (!CURRENT || !(CURRENT.session || CURRENT.restore || CURRENT.refit)) return;
  if (!(await ensureLive())) return;
  const epoch = $("elemEpoch").value;            // "YYYY-MM-DD" from the date picker
  if (!epoch) return;
  const inp = $("elemEpoch");
  inp.disabled = true; inp.classList.add("busy");
  try {
    const r = await fetch("/api/set_epoch", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: CURRENT.session, epoch, mask: currentMask() }),
    });
    const p = await FitJob.settle(await r.json(), (t) => {
      $("msg").innerHTML = `<span class="loading">${escHtml(t)}</span>`;
    });
    if (!p.ok) {
      $("msg").innerHTML = `<b>Set epoch failed.</b> ${escHtml(p.error || "")}` +
        (p.expired ? ` <a href="#" onclick="runFit();return false">re-fit</a>` : "");
      return;
    }
    $("msg").innerHTML = "";
    CURRENT = p; render(p); $("workspace").hidden = false;
  } catch (e) {
    $("msg").textContent = "set epoch failed: " + e.message;
  } finally {
    inp.disabled = false; inp.classList.remove("busy");
  }
}

// ---- non-gravitational models (find_orb's "turn on A1/A2/...") -----------
let NG_MODELS = null;     // [{value,label}] from /api/nongrav_models (fetched once)

async function loadNongravModels() {
  try {
    const r = await fetch("/api/nongrav_models");
    NG_MODELS = (await r.json()).models || [];
  } catch (e) { NG_MODELS = [{ value: "none", label: "No non-gravs (gravity only)" }]; }
  const sel = $("ngSelect");
  sel.innerHTML = NG_MODELS.map((m) => `<option value="${escAttr(m.value)}">${escHtml(m.label)}</option>`).join("");
}

// Re-solve the current object with the chosen nongrav model (always from the gravity
// baseline server-side, so flipping models never compounds).
async function runNongrav(model) {
  if (!CURRENT || !(CURRENT.session || CURRENT.restore || CURRENT.refit)) return;
  if (!(await ensureLive())) return;
  const sel = $("ngSelect");
  sel.disabled = true;
  $("ngBody").innerHTML = `<div class="ngnote loading">solving ${escHtml(model === "none" ? "gravity" : model)}…</div>`;
  try {
    const r = await fetch("/api/nongrav", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: CURRENT.session, model, mask: currentMask() }),
    });
    const p = await FitJob.settle(await r.json(), (t) => {
      $("ngBody").innerHTML = `<div class="ngnote loading">${escHtml(t)}</div>`;
    });
    if (!p.ok) {
      $("ngBody").innerHTML = `<div class="ngnote bad">${escHtml(p.error || "non-gravitational fit failed")}` +
        (p.expired ? ` — <a href="#" onclick="runFit();return false">re-fit</a>` : "") + `</div>`;
      return;
    }
    CURRENT = p; render(p); $("workspace").hidden = false;
  } catch (e) {
    $("ngBody").innerHTML = `<div class="ngnote bad">request failed: ${escHtml(e.message)}</div>`;
  } finally {
    sel.disabled = false;
  }
}

const ngFmt = (v) => (v === null || v === undefined || Number.isNaN(v))
  ? "—" : v.toExponential(3);

// On-demand HONEST ± (the "compute uncertainties" button): the interactive fit serves the fast
// linear sigma, which we never display; this asks the server for the profile-likelihood interval
// (re-fits the orbit at trial values, a few seconds) and re-renders the panel with the true ±
// and any "unconstrained" axes. Stashes the result on CURRENT.nongrav so a re-render keeps it.
async function computeNongravSigmas() {
  if (!CURRENT || !CURRENT.nongrav || !(CURRENT.session || CURRENT.restore || CURRENT.refit)) return;
  if (!(await ensureLive())) return;
  const btn = document.querySelector(".ngsigbtn");
  if (btn) { btn.disabled = true; btn.textContent = "computing ±… (a few seconds)"; }
  try {
    const r = await fetch("/api/nongrav_sigmas", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: CURRENT.session }),
    });
    const p = await FitJob.settle(await r.json(), (t) => {
      if (btn) btn.textContent = t;
    });
    if (!p.ok) {
      if (btn) { btn.disabled = false; btn.textContent = (p.error || "failed") + " — retry ±"; }
      return;
    }
    CURRENT.nongrav._sigmas = p.sigmas || {};
    CURRENT.nongrav._intervals = p.intervals || {};   // {axis: [minus, plus]} honest asymmetric halves
    CURRENT.nongrav._unc = p.unconstrained || [];
    CURRENT.nongrav._jk = p.jackknife || {};          // {axis: {se, z, spans_zero, n, sampled, n_tracklets}}
    CURRENT.nongrav._notrobust = p.not_robust || [];  // axes whose leave-one-tracklet cloud straddles 0
    // AMR's ± is A1's, rescaled by 1/SRP1AU server-side (nothing extra was refit for it).
    CURRENT.nongrav._amrSigma = p.amr_sigma;
    CURRENT.nongrav._amrInterval = p.amr_interval;    // [minus, plus] on the same scale
    renderNongrav(CURRENT);
  } catch (e) {
    if (btn) { btn.disabled = false; btn.textContent = "request failed — retry ±"; }
  }
}

// A/B the force model: toggle the 16 asteroid perturbers and re-fit. The server re-seeds
// from the pure-gravity baseline and holds the CURRENT observation set fixed -- a pure force-
// model swap, NO redemption or re-rejection (inclusion only ever changes via a manual obs
// toggle), so flipping never compounds and never silently changes what's in or out.
async function setAsteroids(on) {
  if (!CURRENT || !(CURRENT.session || CURRENT.restore || CURRENT.refit)) return;
  if (!(await ensureLive())) return;
  const tog = $("astToggle");
  tog.classList.add("busy");
  $("ngBody").innerHTML = `<div class="ngnote loading">` +
    `${on === "all" ? "adding every massive asteroid (slower)"
       : on ? "adding 16 asteroid perturbers" : "reverting to gravity"}…</div>`;
  try {
    const r = await fetch("/api/asteroids", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: CURRENT.session, on, mask: currentMask() }),
    });
    const p = await FitJob.settle(await r.json(), (t) => {
      $("ngBody").innerHTML = `<div class="ngnote loading">${escHtml(t)}</div>`;
    });
    if (!p.ok) {
      $("ngBody").innerHTML = `<div class="ngnote bad">${escHtml(p.error || "toggle failed")}` +
        (p.expired ? ` — <a href="#" onclick="runFit();return false">re-fit</a>` : "") + `</div>`;
      return;
    }
    // The object IS one of the 16 perturbers and the server could not exclude it -> banner, gravity.
    if (p.self_perturber && p.banner) flashBanner(p.banner, "alert");
    CURRENT = p; render(p); $("workspace").hidden = false;
  } catch (e) {
    $("ngBody").innerHTML = `<div class="ngnote bad">request failed: ${escHtml(e.message)}</div>`;
  } finally {
    tog.classList.remove("busy");
  }
}

// AREA/MASS RATIO -- find_orb's "Epoch ... ; AMR 0.034 m^2/kg" line (elem_out.cpp), which it
// prints for its 1-parameter SRP force model ONLY: with A2/A3 also free the radial coefficient
// is not the whole photon push, so calling it an area/mass ratio would over-claim. The server
// applies that rule and does the arithmetic (payload nongrav.amr is null on every other model),
// so nothing here carries a copy of SRP1AU. The ± appears with the profile sigmas.
function ngAmrLine(ng) {
  const amr = ng.amr;
  if (amr === null || amr === undefined) return "";
  const iv = ng._amrInterval, s = ng._amrSigma;
  const loose = (ng._unc || []).indexOf("A1") >= 0;   // A1's profile never bounded it
  // 3 significant figures, trailing zeros dropped -- but only after a decimal point: a bare
  // /0+$/ turned an AMR of 100 m²/kg (a solar sail, a foil sheet) into 1.
  const g = (v) => {
    const t = Number(v).toPrecision(3);
    return (t.includes("e") || !t.includes(".")) ? t : t.replace(/0+$/, "").replace(/\.$/, "");
  };
  let err = "";
  if (loose && s !== null && s !== undefined) {
    err = ` <span class="ngsig weak" title="A1 is unconstrained, so this is a lower bound on ` +
      `the uncertainty -- the fit bounds the area/mass ratio no better than zero">&gt; ± ${g(s)}</span>`;
  } else if (iv && Math.abs(iv[1] - iv[0]) > 0.15 * Math.max(iv[0], iv[1], 1e-300)) {
    err = ` <span title="asymmetric Δχ²=1 interval">+${g(iv[1])} / −${g(iv[0])}</span>`;
  } else if (s !== null && s !== undefined) {
    err = ` ± ${g(s)}`;
  }
  return `<div class="ngamr" title="the area/mass ratio this A1 implies for a PERFECT ABSORBER ` +
    `(A1 = SRP1AU × AMR) -- a real body reflects and re-radiates, so its true area/mass is ` +
    `somewhat smaller. find_orb prints this for the same 1-parameter SRP model.">` +
    `AMR <b>${g(amr)}${err}</b> m²/kg</div>`;
}

function renderNongrav(p) {
  const panel = $("ngPanel"), sel = $("ngSelect"), body = $("ngBody");
  // The nongrav force lives in the ASSIST backend only; hide the panel otherwise.
  // Shown for any payload that CAN become live, not only one holding a session: a disk-cache
  // hit (what a ?object= deeplink usually gets) carries none until the viewer clicks, and
  // hiding the panel then left the field invisible -- 352870 read as gravity only while its
  // orbit ran in +16 ast +(471). Every control in here claims a session (ensureLive) first.
  if (!p.force_model || p.force_model.indexOf("ASSIST") === -1
      || !(p.session || p.restore || p.refit)) {
    panel.hidden = true; return;
  }
  panel.hidden = false;
  // Reflect the active force model in the gravity / +16 ast / +343 ast segmented toggle.
  const cur = astState(p);
  $("astToggle").querySelectorAll(".seg")
    .forEach((s) => s.classList.toggle("active", s.dataset.ast === cur));
  // "all": the whole extra-perturber kernel, counted on the button; hidden without the kernel
  const nAll = p.perturbers_all_count || 0;
  $("astToggle").querySelectorAll("button[data-ast='all']").forEach((s) => {
    s.hidden = !nAll;
    s.textContent = `+${nAll} ast`;
  });
  // If this object IS one of the 16 +16ast perturbers, the +16ast segment runs the OTHER 15:
  // it can't be perturbed by itself, so the engine excludes it from its own field.
  const sp = p.self_perturber;
  // Extra perturbers beyond the 16 (the perturber screen's picks): named on the toggle and
  // in a standing line under it, so the field the orbit lives in is never only in a tooltip.
  const ex = p.asteroid_extras || [];
  const det = p.extra_perturbers || [];
  $("astToggle").querySelectorAll("button[data-ast='1']").forEach((s) => {
    // numbered-asteroid style "+(324)": a WORD JOINER (U+2060, invisible) between "+" and "("
    // so the browser never breaks there
    s.textContent = "+16 ast" + ex.map((n) => ` +⁠(${n})`).join("");
    s.title = (sp ? `(${sp}) is one of the 16 perturbers — this uses the other 15` : "")
      + (ex.length ? `${sp ? "; " : ""}plus ${ex.map((n) => `(${n})`).join(", ")}` : "");
  });
  const exNote = $("astExtras");
  if (exNote && p.perturbers_all) {
    exNote.hidden = false;
    exNote.innerHTML = `<b>All ${nAll} massive asteroids</b> of the perturber kernel are in the ` +
      `field, as asked — not only the ones the screen judged to matter.`;
  } else if (exNote) {
    exNote.hidden = !det.length;
    exNote.innerHTML = det.length
      ? "<b>Beyond the 16:</b> " + det.map((e) =>
          `(${e.number}) ${escHtml(e.name || "")} — Δχ² ${fmt(e.dchi2, 0)}, passed ` +
          `${fmtAu(e.closest_au)} au on ${escHtml(e.closest_date || "")}` +
          (e.mass_kg ? `, mass ${Number(e.mass_kg).toExponential(2)} kg (${escHtml(e.mass_source || "")})`
                     : "")).join("; ") +
        `. Added by the perturber screen: these observations need ${det.length === 1 ? "it" : "them"}.`
      : "";
  }
  if (!sel.options.length && NG_MODELS) {
    sel.innerHTML = NG_MODELS.map((m) => `<option value="${escAttr(m.value)}">${escHtml(m.label)}</option>`).join("");
  }
  const ng = p.nongrav || {};
  const requested = p.nongrav_request || (ng.model || "none");
  sel.value = requested;

  if (!ng.model) {
    // gravity, or a requested model that found no improvement and was reverted
    if (requested && requested !== "none") {
      body.innerHTML = `<div class="ngnote">No improvement over gravity — kept the ` +
        `gravitational orbit (this arc shows no significant signal for that model).</div>`;
    } else {
      body.innerHTML = `<div class="ngnote">Pure gravity. Pick a model above to fit ` +
        `out-gassing (comet A1/A2/A3) or Yarkovsky/SRP on top of this orbit.</div>`;
    }
    return;
  }
  const params = ng.params || {}, sigmas = ng.sigmas || {};
  // The ± is deferred: the HONEST profile-likelihood interval costs seconds of orbit re-fits, so we
  // never show the (overconfident) linear one. The sigma cell is left BLANK until the user clicks
  // "compute uncertainties (±)" -- a placeholder dash there reads as a minus sign ("negative"), so
  // the empty cell + the button below is what signals "not computed yet".
  // `_sigmas` (profile result) + `_unc` (unconstrained axes) are set by computeNongravSigmas().
  const shown = ng._sigmas || null, unc = new Set(ng._unc || []);
  const intervals = ng._intervals || {};
  const jk = ng._jk || {};                       // delete-1-tracklet robustness sigma (button result)
  const AXES = { A1: "A1 · radial", A2: "A2 · transverse", A3: "A3 · normal" };
  const rows = Object.keys(AXES).filter((k) => k in params).map((k) => {
    const v = params[k];
    let sCell = "", dCell = "";
    if (shown) {
      const s = shown[k];
      if (unc.has(k)) {
        sCell = `&gt; ± ${ngFmt(s)}`;
        dCell = `<span class="ngsig weak" title="the fit constrains this no better than zero">unconstrained</span>`;
      } else {
        const sig = (s && s > 0) ? Math.abs(v) / s : null;   // significance off the PROFILE sigma
        // Show the honest ASYMMETRIC interval ("+plus / -minus") when the -/+ half-widths differ
        // meaningfully on the degenerate valley; otherwise a plain "± sig".
        const iv = intervals[k];
        if (iv && Math.abs(iv[1] - iv[0]) > 0.15 * Math.max(iv[0], iv[1], 1e-300)) {
          sCell = `<span title="asymmetric Δχ²=1 interval">+${ngFmt(iv[1])} / −${ngFmt(iv[0])}</span>`;
        } else {
          sCell = `± ${ngFmt(s)}`;
        }
        dCell = sig === null ? "" : `<span class="${sig >= 3 ? "ngsig strong" : "ngsig weak"}">${sig.toFixed(1)}σ</span>`;
      }
    }
    // Robustness on its OWN LINE (a full-width sub-row under the coefficient), not a column --
    // the delete-1-tracklet jackknife ±, and either its σ or a "not robust" badge when the
    // leave-one-tracklet cloud straddles zero (a systematic / a few high-leverage tracklets the
    // formal ± cannot see). Kept narrow so the panel does not widen.
    let jkRow = "";
    if (shown && jk[k]) {
      const d = jk[k];
      let body;
      if (d.assessed === false) {
        // `n` is CONVERGED refits, `attempted` is how many deletions were tried at all
        // (<= n_tracklets: the refit budget or an undeletable tracklet can leave it short),
        // and `reason` says WHY not-assessed -- "N of M refits converged" used to be the only
        // text shown, which reads as "we tried every tracklet in the arc" even when the
        // budget never reached most of them.
        const why = d.reason ? ` -- ${d.reason}` : "";
        body = `robustness (jackknife): not assessed ` +
          `<span class="ngsig weak">(${d.n || 0} of ${d.attempted ?? d.n_tracklets} attempted ` +
          `converged, arc has ${d.n_tracklets} tracklets${why})</span>`;
      } else {
        const flagged = d.spans_zero;
        // z can be absent: a cloud with no measurable spread is reported as no significance at
        // all rather than as an infinite one, and the API sends non-finite numbers as null. Calling
        // .toFixed on that threw, and the panel rendered a successful request as "request failed".
        const verdict = flagged
          ? `<span class="ngjkbad" title="the sign does not survive dropping a tracklet">not robust</span>`
          : (Number.isFinite(d.z)
             ? `<span class="ngsig ${d.z >= 5 ? "strong" : ""}" title="robust MAD-scaled significance">${d.z.toFixed(1)}σ</span>`
             : `<span class="ngsig weak" title="the delete-1 refits showed no measurable spread">—</span>`);
        // PARTIAL COVERAGE: some tracklet of the arc was never deleted -- the refit budget
        // skipped it, or deleting it left too little arc to refit. Either way "the sign
        // survives dropping a tracklet" is a claim about the ones that were tried, so say so
        // rather than letting a partial sweep read as an exhaustive one. `d.attempted` is how
        // many deletions were TRIED (<= n_tracklets) and `d.n` how many of those CONVERGED --
        // showing `n` alone as "of n_tracklets" conflated the two and understated attempted
        // coverage whenever a refit was tried but rejected.
        const cover = d.sampled
          ? ` <span class="ngsig weak" title="not every tracklet was deleted -- an untried ` +
            `one could still flip the sign">` +
            `(${d.attempted ?? d.n} of ${d.n_tracklets} tracklets tried, ${d.n} converged)</span>`
          : "";
        body = `robustness (jackknife): ± ${ngFmt(d.se)} · ${verdict}${cover}`;
      }
      const flagged = d.assessed !== false && d.spans_zero;
      const scope = d.sampled ? "a sampled tracklet" : "any one tracklet";
      jkRow = `<tr class="ngjkrow${flagged ? " bad" : ""}"><td></td>` +
        `<td colspan="3" class="ngjk mono" title="delete-1-tracklet jackknife (robust): does the ` +
        `coefficient's sign survive dropping ${scope}? A check the formal ± cannot make.">` +
        `${body}</td></tr>`;
    }
    return `<tr><td class="k" title="${AXES[k]}">${k}</td>` +
      `<td class="v mono">${ngFmt(v)}</td><td class="s mono">${sCell}</td>` +
      `<td class="d">${dCell}</td></tr>` + jkRow;
  }).join("");
  const sigCtl = shown
    ? `<div class="ngunit">± = profile-likelihood (Δχ²=1)</div>`
    : `<button class="ngsigbtn" onclick="computeNongravSigmas()" ` +
      `title="re-fits the orbit at trial values for the true profile ± AND the delete-1-tracklet robustness jackknife (some seconds of re-fits)">` +
      `compute uncertainties (± &amp; robustness)</button>`;
  const head = shown
    ? `<thead><tr><th></th><th></th><th class="s">formal ±</th><th class="d"></th></tr></thead>`
    : "";
  const jkNote = (shown && ng._notrobust && ng._notrobust.length)
    ? `<div class="ngnote bad">${ng._notrobust.join(", ")} not robust to dropping a tracklet — ` +
      `the formal ± is optimistic here (it cannot be told from a systematic or a few high-leverage tracklets).</div>`
    : "";
  body.innerHTML =
    `<table class="kv ngtab">${head}<tbody>${rows}</tbody></table>` +
    ngAmrLine(ng) + jkNote + sigCtl + `<div class="ngunit">A in au/day²</div>`;
}

// ---- constrained fits (find_orb's limited_orbit) ------------------------
let CON_ELEMENTS = null;   // [{value,label,unit}] from /api/constraint_elements

async function loadConstraintElements() {
  try {
    const r = await fetch("/api/constraint_elements");
    CON_ELEMENTS = (await r.json()).elements || [];
  } catch (e) {
    CON_ELEMENTS = [{ value: "a", label: "a — semi-major axis", unit: "au" },
      { value: "e", label: "e — eccentricity", unit: "" },
      { value: "q", label: "q — perihelion distance", unit: "au" },
      { value: "Q", label: "Q — aphelion distance", unit: "au" },
      { value: "i", label: "i — inclination", unit: "deg" }];
  }
}

// POST a {element: target} set (possibly empty = release all) and re-render.
async function postConstraints(cons) {
  if (!CURRENT || !(CURRENT.session || CURRENT.restore || CURRENT.refit)) return;
  if (!(await ensureLive())) return;
  const status = $("conStatus");
  if (status) status.innerHTML = `<span class="loading">` +
    (Object.keys(cons).length ? `holding ${Object.keys(cons).join(", ")} fixed…` : "releasing…") +
    `</span>`;
  try {
    const r = await fetch("/api/constrain", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: CURRENT.session, constraints: cons, mask: currentMask() }),
    });
    const p = await FitJob.settle(await r.json(), (t) => {
      if (status) status.innerHTML = `<span class="loading">${escHtml(t)}</span>`;
    });
    if (!p.ok) {
      if (status) status.innerHTML = `<span class="bad">${escHtml(p.error || "constrained fit failed")}` +
        (p.expired ? ` — <a href="#" onclick="runFit();return false">re-fit</a>` : "") + `</span>`;
      return;
    }
    CURRENT = p; render(p); $("workspace").hidden = false;
  } catch (e) {
    if (status) status.innerHTML = `<span class="bad">request failed: ${escHtml(e.message)}</span>`;
  }
}

// Gather the checked element rows into {element: target} and re-solve.
function applyConstraints() {
  const cons = {};
  (CON_ELEMENTS || []).forEach((el) => {
    const on = $(`con_on_${el.value}`), val = $(`con_val_${el.value}`);
    if (on && on.checked && val && val.value !== "") cons[el.value] = parseFloat(val.value);
  });
  postConstraints(cons);
}

function renderConstraints(p, resetToggle = true) {
  const panel = $("conPanel"), toggle = $("conToggle"), body = $("conBody");
  // Constraints are pure functions of the state -> available on EITHER backend.
  if (!p.session) { panel.hidden = true; return; }
  panel.hidden = false;
  const active = p.constraints_request || p.constraints || {};
  const anyActive = Object.keys(active).length > 0;
  // Keep the panel open across interactive updates; collapse on a fresh object fit
  // unless that fit already carries constraints. Only re-derive on a full render():
  // when the user CLICKS the toggle (resetToggle=false) we must honour the box they
  // just set. A fresh autofit payload isn't "interactive", so re-deriving here would
  // instantly un-check the box -- which is why it only stuck after a Full step.
  if (resetToggle) {
    toggle.checked = anyActive || (toggle.checked && p.meta && p.meta.interactive);
  }
  if (!toggle.checked) { body.hidden = true; body.innerHTML = ""; return; }
  body.hidden = false;
  const vals = {}; (p.elements || []).forEach((e) => { vals[e.name] = e.value; });
  const fmtVal = (v) => (typeof v !== "number") ? v
    : (Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(3) : +v.toFixed(6));
  const rows = (CON_ELEMENTS || []).map((el) => {
    const on = el.value in active;
    // vals[] is null for every element of a suppressed orbit (no nominal was sent) -- seed the
    // box empty there rather than printing "null" into it.
    const nom = vals[el.value];
    const cur = on ? active[el.value] : (nom === undefined || nom === null ? "" : nom);
    return `<div class="conrow">` +
      `<label class="conlab"><input type="checkbox" id="con_on_${el.value}" ${on ? "checked" : ""}>` +
      ` ${el.label}</label>` +
      `<input type="number" step="any" id="con_val_${el.value}" class="conval" value="${fmtVal(cur)}">` +
      `<span class="conunit">${el.unit || ""}</span></div>`;
  }).join("");
  body.innerHTML =
    `<div class="conpresets">presets: ` +
      `<button type="button" class="conpreset" data-p="1">parabolic (e=1)</button> ` +
      `<button type="button" class="conpreset" data-p="0">circular (e=0)</button></div>` +
    rows +
    `<div class="conactions"><button type="button" id="conApply" class="conapply">Apply</button>` +
    `<span id="conStatus" class="ngnote"></span></div>` +
    `<div class="conhint">Held elements are pinned exactly; the rest re-solve to fit the ` +
    `astrometry. Expect the RMS to rise when a constraint fights the data ` +
    `(find_orb limited_orbit).</div>`;
  $("conApply").addEventListener("click", applyConstraints);
  body.querySelectorAll(".conpreset").forEach((b) => b.addEventListener("click", () => {
    const on = $("con_on_e"), val = $("con_val_e");
    if (on && val) { on.checked = true; val.value = b.dataset.p; }
  }));
}

// ---- render orchestration ------------------------------------------------
function render(p) {
  viewStage = null;                 // a fresh fit always lands on the final orbit
  workingMask = [];                 // start editing from the fit's own inclusion set
  p.residuals.forEach((r) => { workingMask[r.i] = r.included; });
  $("msg").innerHTML = p.warning
    ? `<span style="color:var(--warn)">⚠ ${p.warning}</span>` : "";
  renderStatus(p);
  renderElements(p);
  renderPhysical(p);
  renderNongrav(p);
  renderConstraints(p);
  renderHistory(p);
  renderObs(p);
  renderObservatories(p);
  renderOCPlot(p);
  if (orbit3d) orbit3d.setData(p.orbit3d);
  else orbit3d = new Orbit3D($("orbitCanvas"), p.orbit3d, $("legend"));
  // central pane: a brand-new object lands on Observations -- unless this fit came from
  // a deep link naming a &view= (e.g. Orbarium), which lands there instead, once; a re-fit
  // of the same object (toggle/model/epoch change) keeps the current pane and refreshes it.
  const newObj = (p.object !== _lastRenderedObj);
  _lastRenderedObj = p.object;
  if (newObj) {
    const v = _pendingView; _pendingView = null;
    setCentralPane(v || "obs");
  }
  else if (centralPane === "ephem") loadEphem();
  else if (centralPane === "impact") loadImpact();
  else if (centralPane === "orbarium" && orbarium) orbarium.setData(p);
  $("exportBtn").disabled = false;     // an orbit is loaded -> export is available
  // ...except the ORBIT products when the fit carries no determined orbit (orbitsuppress.js).
  if (window.applyOrbitSuppression) window.applyOrbitSuppression(p);
  $("itfBtn").disabled = false;        // ... and so is the ITF search
  // "Submit linkage" (or, when the server has MPC submission disabled, the
  // offline "Download linkage file" alternative) appears whenever the payload
  // carries a submittable linkage: an ITF-search find, or the identification a
  // multi-designation fit implies (kind "ident" -- fit two designations together
  // and the MPC document that claim needs is already built). An ITF find's
  // button vanishes on any later re-fit/edit that drops .itf, since the pending
  // linkage is stale then; an ident linkage is re-derived from the fit each
  // payload, so it simply follows the observations on screen.
  const linked = !!(p.itf && p.itf.found && p.itf.submittable && p.session);
  // one split button: "Submit linkage" -> the PyRoOrb queue; its ▾ menu -> the
  // file, the identity, and (DIRECT_SUBMIT_ALLOWED only) straight to the MPC
  _setHidden($("itfSubmitWrap"), linked);
  _setHidden($("itfSubmitMenu"), false);
  $("itfSubmitBtn").disabled = false;
  $("itfSubmitBtn").textContent = "Submit linkage";
  _setHidden($("itfDirectBtn"), linked && !!p.itf.direct_submit_allowed);
  $("itfDirectBtn").disabled = false;
  $("itfDownloadBtn").disabled = false;
  $("itfDownloadBtn").textContent = "Download linkage file (JSON)";
  refreshSubmitGate();          // ...and never on an unsolved observation set
  flashIdentLinkage(p);         // say WHY the submit button just appeared
  flashFitBanners(p);                  // one-shot asteroid-polish / systematics notices
  // Sibling modules (convergence.js and any later one) render off the payload
  // without being wired into this function: this is the hook they listen on, so
  // reworking any render* above cannot silently strand them.
  document.dispatchEvent(new CustomEvent("pyroorb:render", { detail: p }));
}

function renderStatus(p) {
  const s = p.summary;
  const good = s.rms_residual < 0.6 && s.n_obs >= Math.max(6, s.n_obs_total * 0.55);
  const link = good ? '<span class="chip ok">ok</span>' : '<span class="chip suspect">suspect</span>';
  const pha = p.physical.pha ? '<span class="chip pha">PHA</span>' : "";
  // Natural-satellite badge: the object orbits a giant planet (auto-detected or --center'd).
  const sp = p.planetocentric;
  const sat = sp
    ? `<span class="chip sat" title="Detected as a natural satellite — its orbit about ${sp.center_label} is shown in the Orbital elements panel (toggle heliocentric ⇄ ${sp.center_label}-centric).">${sp.class}</span>`
    : "";
  // Prefer the published "(279) Thule" number+name form (several bodies -> show all);
  // fall back to the working designation when nothing resolved (e.g. an unnumbered
  // NEOCP object). A resolved name keeps the raw designation as a hover tooltip.
  const names = p.mp_names || [];
  const objLabel = names.length ? names.map((n) => n.label).join(" · ") : p.object;
  const objTitle = names.length ? ` title="${p.object}"` : "";
  // Radar (delay/Doppler) folded into the fit — same measurements the CLI reports. Compact
  // stat + median-|O−C| tooltip; the full per-row O−C block rides p.radar.rows for a panel.
  const rad = p.radar;
  const radar = rad
    ? (rad.applied
        ? `<div class="stat" title="${rad.n_used}/${rad.n} delay/Doppler radar measurements folded into the fit (median |O−C| ${fmt(rad.post_median_z, 2)}σ)"><span>radar 📡</span><b>${rad.n_used}</b>/${rad.n}</div>`
        : `<div class="stat" title="${rad.n} radar measurements present but not fit (${rad.note || "unavailable"})"><span>radar 📡</span><b>—</b></div>`)
    : "";
  $("status").innerHTML =
    `<div class="stat"><span>obj</span><b${objTitle}>${objLabel}</b>${p.combined ? " ⛓" : ""}</div>` +
    `<div class="stat"><span>RMS</span><b>${fmt(s.rms_residual, 3)}″</b> (${fmt(s.rms_normalized, 2)}σ)</div>` +
    `<div class="stat"><span>obs</span><b>${s.n_obs}</b>/${s.n_obs_total}</div>` +
    `<div class="stat"><span>arc</span><b>${fmt(s.arc_days, 0)}</b>d · ${s.n_oppositions}opp</div>` +
    radar + link + pha + sat;
}

// Greek/short symbols so the row is compact; the full name is a hover tooltip
// (native title appears after ~1s) instead of eating the width the ±σ needs.
const ELEM_SYM = { asc_node: "Ω", arg_per: "ω" };
// Essentially-unconstrained heads-up: a short arc that fits cleanly but leaves the orbit's
// scale almost free. p.constraint is null for a determined orbit. The text is brief and
// generic by design (headline + one neutral sentence) -- it never ranks individual elements.
function renderConstraintWarn(p) {
  const box = $("constraintBanner");
  const c = p.constraint;
  if (!c) { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
  box.classList.toggle("severe", c.level === "unconstrained");
  // A dof-thin fit says so here, as one sentence, rather than in a second banner (dof.js).
  const dof = window.dofClause ? window.dofClause(p) : null;
  const text = [c.detail, dof].filter(Boolean).join(" ");
  const detail = text ? `<div class="cw-detail">${escHtml(text)}</div>` : "";
  box.innerHTML = `<div class="cw-head">⚠ ${escHtml(c.headline)}</div>${detail}`;
}

// Which element set the panel shows for a satellite: "helio" (osculating heliocentric) or
// "planeto" (osculating about the planet). Persists while toggling the same object; resets to
// "planeto" when a new satellite loads (that's the meaningful orbit). "helio" for non-satellites.
let elemView = "helio";
let _lastElemObj = null;

function renderElements(p) {
  renderConstraintWarn(p);
  const sat = p.planetocentric;                 // null unless this object is a satellite
  if (p.object !== _lastElemObj) {              // a freshly loaded object picks the default view
    elemView = sat ? "planeto" : "helio";
    _lastElemObj = p.object;
  }
  if (!sat) elemView = "helio";
  // Satellite view toggle: heliocentric ⇄ <Planet>-centric osculating elements.
  const tog = $("satToggle");
  if (sat) {
    tog.hidden = false;
    tog.innerHTML =
      `<button class="segbtn ${elemView === "helio" ? "active" : ""}" data-ev="helio" ` +
      `title="Osculating heliocentric elements (the satellite's Sun-relative orbit)">heliocentric</button>` +
      `<button class="segbtn ${elemView === "planeto" ? "active" : ""}" data-ev="planeto" ` +
      `title="Osculating elements about ${sat.center_label} — the satellite's real orbit">${sat.center_label}-centric</button>`;
    tog.querySelectorAll("button").forEach((b) => {
      b.onclick = () => { elemView = b.dataset.ev; renderElements(CURRENT); };
    });
  } else {
    tog.hidden = true; tog.innerHTML = "";
  }
  const usingSat = sat && elemView === "planeto";
  const rows = usingSat ? sat.elements : p.elements;
  $("frameLbl").textContent = usingSat ? `· ${sat.frame}` : "";
  // Editable osculating epoch: only for the heliocentric interactive session (the planetocentric
  // re-anchor isn't wired) -- hide it in the satellite view.
  const ep = p.epoch || {};
  const ctl = $("epochCtl");
  if (p.session && ep.iso && !usingSat) {
    $("elemEpoch").value = ep.iso.slice(0, 10);
    ctl.hidden = false;
  } else {
    ctl.hidden = true;
  }
  // An orbit the astrometry does not determine is shown as per-element ranges and no values
  // at all (orbitsuppress.js; the server has already blanked them). Null = ordinary fit.
  const suppressed = window.suppressedElementRows ? window.suppressedElementRows(p, rows) : null;
  $("elemBody").innerHTML = suppressed !== null ? suppressed : rows.map((e) => {
    const sym = ELEM_SYM[e.name] || e.name;
    const u = e.unit ? ` ${e.unit}` : "";
    const sig = e.sigma !== null ? `± ${e.sigma < 1e-3 ? e.sigma.toExponential(1) : fmt(e.sigma, 4)}` : "";
    const sigTitle = e.sigma !== null
      ? ` title="1-σ (~68% confidence) formal uncertainty in ${sym}, from the linearised fit covariance"` : "";
    const tpTitle = e.name === "Tp" && e.value !== null && e.value !== undefined && !Number.isNaN(e.value)
      ? ` title="${jdToDate(e.value).slice(0, 10)}"` : "";
    return `<tr><td class="k" title="${e.label}">${sym}</td>` +
      `<td class="v"${tpTitle}>${fmt(e.value, e.name === "Tp" ? 3 : 6)}${u}</td><td class="s"${sigTitle}>${sig}</td></tr>`;
  }).join("");
}

function renderPhysical(p) {
  const ph = p.physical;            // epoch now lives (editable) in the Orbital elements panel
  const cell = (lbl, val) => `<div class="cell"><div class="lbl">${lbl}</div><div class="val">${val}</div></div>`;
  const moid = ph.moid_earth_au !== null
    ? `${fmt(ph.moid_earth_au, 4)} au` : "—";
  $("physBody").innerHTML =
    cell("class", ph.class || "—") +
    cell("Earth MOID", moid) +
    cell("T<sub>J</sub>", fmt(ph.tisserand_jupiter, 2)) +
    cell("U (uncert.)", ph.U !== null ? fmt(ph.U, 1) : "—") +
    cell("H (abs. mag)", ph.H !== null ? `${fmt(ph.H, 2)} ± ${fmt(ph.H_sigma, 2)}` : "—") +
    cell("force model", p.force_model) +
    cell("frame", ph.frame.replace(" (heliocentric)", ""));
}

function renderHistory(p) {
  $("iodLbl").textContent = "";   // IOD now appears as the first row of the trace
  if (!p.history.length) { $("histBody").innerHTML = '<li class="empty">no stage trace</li>'; return; }
  const snaps = p.stage_residuals || [];
  $("histBody").innerHTML = p.history.map((h, idx) => {
    const dropped = h.kept === false ? " dropped" : "";
    const can = snaps[idx] ? " clickable" : "";        // only stages with a snapshot replay
    const active = idx === viewStage ? " active" : "";
    return `<li class="${dropped}${can}${active}" data-stage="${idx}"><span class="stage">${h.stage}</span>` +
      `<span class="meta">${h.n_obs} obs · ${fmt(h.rms, 3)}″` +
      (h.score !== null ? ` · ${fmt(h.score, 2)}σ` : "") + `</span></li>`;
  }).join("");
}

// ---- observations table --------------------------------------------------
// Above this many rows the table VIRTUALIZES: only the rows in (and just around) the
// viewport are kept in the DOM, re-rendered on scroll. A mega-object's 8000+ obs would
// otherwise be ~8000×13 cells rebuilt on every sort/toggle -- so at/below this count we
// render the whole table at once (exactly as before, zero behaviour change for normal
// objects), and only above it do we window. Nothing is hidden either way: the full list
// is reachable by scrolling, and the count shows the true total.
const VIRTUAL_ABOVE = 1500;
const VIRTUAL_BUFFER = 12;   // rows of overscan rendered above/below the viewport

// Virtualization state for the obs table; null when the table is rendered in full
// (small object, MPC-80 view, or none loaded). `_lastObsP` lets a re-render of the SAME
// payload (a live include/exclude toggle, a sort flip) preserve the scroll position,
// while loading a different object resets to the top.
let _virt = null;
let _lastObsP = null;
let _virtRaf = 0;

// The six extra scroll-right ADES columns, revealed by the "＋ addl info" toggle
// (showAddl). Each is present only for ADES observations that populated the field;
// OBS80 and unpopulated submissions render "—". Kept off-screen by default so the
// common view stays the compact 14-column table. ADDL_COLS must equal the count of
// <td>s here (and the <th>s below) so the virtual-spacer colspan stays correct.
const ADDL_COLS = 6;
function addlCells(r) {
  const num = (v, d) => (v === null || v === undefined) ? "—" : fmt(v, d);
  const str = (v) => (v === null || v === undefined || v === "") ? "—" : escHtml(String(v));
  return `<td class="l addl asep idcell" title="${escAttr(r.obs_id || "")}">${str(r.obs_id)}</td>` +
    `<td class="l addl">${str(r.trksub)}</td>` +
    `<td class="addl">${num(r.ades_rms_ra, 3)}</td>` +
    `<td class="addl">${num(r.ades_rms_dec, 3)}</td>` +
    `<td class="addl">${num(r.mag_sigma, 2)}</td>` +
    `<td class="addl">${num(r.log_snr, 2)}</td>`;
}

// One observations-table row. `next` is the following row in display order (for the
// apparition-gap separator); factored out so the full render and the virtual window
// share one definition. 14 columns (+ ADDL_COLS when showAddl) -- keep in sync with
// the <thead> and the vspace colspan.
const SIGBAR_MAX = 6;   // residual-bar full-scale, in sigmas -- anything beyond is clipped
function obsRowHtml(r, next, maxSig, chrono) {
  const appbreak = chrono && next && (next.jd_utc - r.jd_utc) > APPARITION_GAP_DAYS;
  const cls = (r.included ? "" : "excluded") + (appbreak ? " appbreak" : "");
  const sig = r.resid_sigmas || 0;
  const w = Math.min(40, (sig / maxSig) * 40);
  const col = sig > 3 ? "var(--bad)" : sig > 1.5 ? "var(--warn)" : "var(--good)";
  // A single wild excluded obs (thousands of sigmas) would otherwise squash every real
  // bar to nothing, so the scale tops out at SIGBAR_MAX and anything past it is clipped.
  const bar = `<span class="resbar${sig > SIGBAR_MAX ? " clipped" : ""}" ` +
    `style="width:${w}px;background:${col}" title="${fmt(sig, 1)}σ"></span>`;
  return `<tr class="${cls.trim()}" data-i="${r.i}">
    <td class="l" style="color:var(--muted)">${r.i + 1}</td>
    <td class="l">${jdToDate(r.jd_utc)}</td>
    <td class="l code">${r.obscode}</td>
    <td>${r.ra_deg !== null ? raHMS(r.ra_deg) : "—"}</td>
    <td>${r.dec_deg !== null ? decDMS(r.dec_deg) : "—"}</td>
    <td>${r.mag !== null ? fmt(r.mag, 1) : "—"}</td>
    <td class="l">${bandCell(r.band)}</td>
    <td>${dmagCell(r.dmag)}</td>
    <td style="color:var(--muted)">${r.r_au != null ? fmt(r.r_au, 3) : "—"}</td>
    <td style="color:var(--muted)">${r.delta_au != null ? fmt(r.delta_au, 3) : "—"}</td>
    <td>${r.d_ra_arcsec >= 0 ? "+" : ""}${fmt(r.d_ra_arcsec, 2)}</td>
    <td>${r.d_dec_arcsec >= 0 ? "+" : ""}${fmt(r.d_dec_arcsec, 2)}</td>
    <td>${fmt(r.resid_sigmas, 1)}</td><td class="l">${bar}</td>${showAddl ? addlCells(r) : ""}</tr>`;
}

// Render only the viewport slice of a virtualized obs table into the existing <tbody>,
// padding above/below with a single zero-border spacer row each so the scrollbar geometry
// matches the full list. Cheap enough to call on every scroll frame.
function renderObsWindow() {
  if (!_virt) return;
  const sc = $("obsScroll");
  const { rows, rowH, maxSig, chrono, tbody } = _virt;
  const total = rows.length;
  const vh = sc.clientHeight || 500;
  const first = Math.max(0, Math.floor(sc.scrollTop / rowH) - VIRTUAL_BUFFER);
  const last = Math.min(total, first + Math.ceil(vh / rowH) + 2 * VIRTUAL_BUFFER);
  const spacer = (n) =>
    `<tr class="vspace"><td colspan="${14 + (showAddl ? ADDL_COLS : 0)}" style="height:${n * rowH}px;padding:0;border:0"></td></tr>`;
  let html = first > 0 ? spacer(first) : "";
  for (let k = first; k < last; k++) html += obsRowHtml(rows[k], rows[k + 1], maxSig, chrono);
  if (last < total) html += spacer(total - last);
  tbody.innerHTML = html;
}

// The rows to render: the final fit, OR a history stage's snapshot (residuals +
// inclusion recomputed at that stage's orbit, index-aligned with p.residuals).
function rowsForView(p) {
  const snap = (viewStage !== null && p.stage_residuals) ? p.stage_residuals[viewStage] : null;
  if (snap) {
    return p.residuals.map((r) => ({
      ...r,
      included: snap.included[r.i],
      d_ra_arcsec: snap.d_ra_arcsec[r.i],
      d_dec_arcsec: snap.d_dec_arcsec[r.i],
      resid_sigmas: snap.resid_sigmas[r.i],
    }));
  }
  // final view: reflect any pending include/exclude edits (orbit unchanged till Full step)
  if (workingMask) return p.residuals.map((r) => ({ ...r, included: workingMask[r.i] }));
  return p.residuals;
}

// the active display order (date, or worst-first) — shared by the table + the
// jump-to-worst and right-click bulk-toggle handlers so "above/below" matches what's shown.
function sortRows(rows) {
  if ($("worstFirst").checked) rows.sort((a, b) => (b.resid_sigmas || 0) - (a.resid_sigmas || 0));
  else rows.sort((a, b) => a.jd_utc - b.jd_utc);
  return rows;
}

function flashRow(tr) { tr.classList.add("flash"); setTimeout(() => tr.classList.remove("flash"), 1600); }

// Scroll to + flash the worst INCLUDED residual (find_orb `w`). Falls back to
// worst-first sort if that row is past the render cap.
function jumpToWorst() {
  if (!CURRENT) return;
  let worst = null;
  for (const r of rowsForView(CURRENT)) {
    if (r.included && (worst === null || (r.resid_sigmas || 0) > (worst.resid_sigmas || 0))) worst = r;
  }
  if (!worst) return;
  let tr = $("obsScroll").querySelector(`tr[data-i="${worst.i}"]`);
  if (!tr && _virt) {
    // Virtualized: the worst row may be outside the rendered window. Scroll to its position
    // in display order and render that window, so the row exists before we centre on it.
    const idx = _virt.rows.findIndex((r) => r.i === worst.i);
    if (idx >= 0) {
      const sc = $("obsScroll");
      sc.scrollTop = Math.max(0, idx * _virt.rowH - sc.clientHeight / 2);
      renderObsWindow();
      tr = sc.querySelector(`tr[data-i="${worst.i}"]`);
    }
  }
  if (!tr) { $("worstFirst").checked = true; renderObs(CURRENT); tr = $("obsScroll").querySelector(`tr[data-i="${worst.i}"]`); }
  if (tr) { tr.scrollIntoView({ block: "center", behavior: "smooth" }); flashRow(tr); }
}

// Right-click bulk toggle: set EVERY obs above/below the pivot (inclusive, in display
// order) to the OPPOSITE of that group's current majority — most-on -> all off, most-off
// -> all on. Edits the working mask only (Full step refits).
function bulkToggle(dir, pivotIdx) {
  if (viewStage !== null || !CURRENT || !workingMask) return;
  const rows = sortRows(rowsForView(CURRENT).slice());
  const pos = rows.findIndex((r) => r.i === pivotIdx);
  if (pos < 0) return;
  const group = dir === "up" ? rows.slice(0, pos + 1) : rows.slice(pos);
  const on = group.filter((r) => workingMask[r.i]).length;
  const target = (on * 2 >= group.length) ? false : true;   // majority on -> exclude all; else include all
  group.forEach((r) => { workingMask[r.i] = target; });
  renderObs(CURRENT);
}

function hideRowMenu() { $("rowMenu").hidden = true; }

// Photometric O−C cell. dmag = observed − model magnitude. We never reject on
// photometry (survey mags are noisy), so this is advisory: muted normally, warn
// past ~1 mag, bad past ~2 mag. Null (no mag / bad geometry) renders as "—".
// Photometric band cell. The ADES band the magnitude was reported in -- one or two
// characters ("V", "G", "r", an LSST "Lu"), and the key photometry.BAND_SHIFT uses to
// reduce that magnitude to V, so it is what every Δmag and the implied H hang on.
// Blank for an obs that carried no magnitude at all -> "—".
function bandCell(band) {
  const b = (band === null || band === undefined) ? "" : String(band).trim();
  if (!b) return `<span style="color:var(--muted)">—</span>`;
  return escHtml(b);
}

function dmagCell(dmag) {
  if (dmag === null || dmag === undefined) return `<span style="color:var(--muted)">—</span>`;
  const a = Math.abs(dmag);
  const col = a > 2 ? "var(--bad)" : a > 1 ? "var(--warn)" : "var(--muted)";
  return `<span style="color:${col}">${dmag >= 0 ? "+" : ""}${fmt(dmag, 1)}</span>`;
}

function renderObs(p) {
  const base = rowsForView(p);
  const stageView = viewStage !== null && p.stage_residuals && p.stage_residuals[viewStage];
  // "not final orbit" replay banner
  const banner = $("obsBanner");
  if (stageView) {
    const h = p.history[viewStage];
    banner.innerHTML = `⚠ <b>NOT FINAL ORBIT</b> — showing progress as of the ` +
      `<b>${h.stage}</b> stage (${h.n_obs} obs, RMS ${fmt(h.rms, 3)}″). ` +
      `<a href="#" data-final="1">Return to current/final fit →</a>`;
  }
  paneStateShow("obsBanner", !!stageView);
  // live edit state (final view only): is the working set different from the fit?
  const stale = !stageView && workingMask &&
    p.residuals.some((r) => workingMask[r.i] !== r.included);
  for (const id of ["fullStepBtn", "multiStepBtn"]) {
    const b = $(id);
    b.hidden = !!stageView;                 // no live editing while replaying a stage
    b.classList.toggle("stale", !!stale);
  }
  refreshSubmitGate();                      // toggles land here, so re-gate submit
  // Re-auto-solve appears only once obs were added via the paste window (and never
  // during a stage replay) -- it re-runs the whole auto pipeline on the merged set.
  $("reAutoBtn").hidden = !!stageView || !obsAdded;

  let rows = sortRows(base.slice());
  const maxSig = Math.min(SIGBAR_MAX, Math.max(2, ...base.map((r) => r.resid_sigmas || 0)));
  const nInc = base.filter((r) => r.included).length;
  let count = `(${rows.length})`;   // virtualization keeps every row reachable -- show the true total
  if (stale) {
    const inc = base.filter((r) => r.included);
    const liveRms = Math.sqrt(inc.reduce((s, r) =>
      s + (r.d_ra_arcsec ** 2 + r.d_dec_arcsec ** 2), 0) / Math.max(1, inc.length));
    count += ` · edited: ${nInc} in, RMS ≈ ${liveRms.toFixed(3)}″ — Full/Multi-step to refit`;
  }
  $("obsCount").textContent = count;

  // MPC 80-col view toggle: highlight the button + reveal the copy button. The
  // "addl info" toggle only affects the residuals table, so its button highlights
  // but its extra columns simply don't apply while the MPC-80 text view is active.
  $("mpc80Btn").classList.toggle("active", obsView === "mpc80");
  $("addlBtn").classList.toggle("active", showAddl);
  $("copyObsBtn").hidden = obsView !== "mpc80";
  if (obsView === "mpc80") { _virt = null; renderObsMpc80(p, base); return; }

  // Extra ADES columns append to the RIGHT (after the residual bar) so the default
  // view is untouched and the additions are reachable by scrolling right; the `wide`
  // class lets the table exceed the pane width instead of squeezing every column.
  const addlHead = showAddl ? `<th class="l asep" title="ADES observation identifier (obsID)">obs ID</th>` +
    `<th class="l" title="ADES tracklet-submission id (trkSub)">trkSub</th>` +
    `<th title="ADES-reported RA uncertainty (arcsec); blank if the submitter didn't report one">σRA″</th>` +
    `<th title="ADES-reported Dec uncertainty (arcsec)">σDec″</th>` +
    `<th title="ADES-reported magnitude uncertainty (rmsMag)">σmag</th>` +
    `<th title="ADES log₁₀(SNR) of the detection">log SNR</th>` : "";
  const head = `<table class="obs${showAddl ? " wide" : ""}"><thead><tr>
    <th class="l">#</th><th class="l">date (UTC)</th><th class="l">code</th>
    <th>RA</th><th>Dec</th><th>mag</th>
    <th class="l" title="ADES photometric band the magnitude was reported in (V, G, r, Lu, …); the band→V shift is what Δmag and H are computed through">band</th>
    <th title="photometric O−C: observed − model magnitude (mag)">Δmag</th>
    <th title="heliocentric (Sun→object) distance, au">r</th><th title="topocentric (Earth→object) distance, au">Δ</th>
    <th>ΔRA″</th><th>ΔDec″</th><th>σ</th><th></th>${addlHead}</tr></thead><tbody>`;
  // Only meaningful in chronological order (worst-first sort scrambles the time axis):
  // whiten the row separator wherever consecutive obs straddle an apparition-sized gap,
  // the same > APPARITION_GAP_DAYS threshold the O−C plot uses for its solid divider.
  const chrono = !$("worstFirst").checked;

  // Small object: render the whole table at once -- byte-identical to the pre-virtualization
  // behaviour (and `_virt = null` disengages the scroll handler).
  if (rows.length <= VIRTUAL_ABOVE) {
    _virt = null; _lastObsP = p;
    const body = rows.map((r, idx) => obsRowHtml(r, rows[idx + 1], maxSig, chrono)).join("");
    $("obsScroll").innerHTML = head + body + "</tbody></table>";
    return;
  }

  // Mega-object: virtualize. Build thead + empty tbody, measure one row's height, then render
  // the viewport window; the scroll handler re-renders the window as the user scrolls. A
  // re-render of the SAME object (toggle / sort flip) keeps the scroll position; a new object
  // starts at the top.
  const sc = $("obsScroll");
  const keep = (p === _lastObsP) ? sc.scrollTop : 0;
  _lastObsP = p;
  sc.innerHTML = head + "</tbody></table>";
  const tbody = sc.querySelector("table.obs tbody");
  tbody.innerHTML = obsRowHtml(rows[0], rows[1], maxSig, chrono);   // one row, to measure
  const rowH = tbody.firstChild.offsetHeight || 21;
  _virt = { rows, rowH, maxSig, chrono, tbody };
  renderObsWindow();              // lay out full height so scrollTop can be restored...
  sc.scrollTop = keep;
  renderObsWindow();              // ...then render the window at the restored position
}

const _esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

// MPC 80-column astrometry view. One canonical OBS80 line per obs (from the payload),
// in chronological order; an EXCLUDED obs gets 'x' in column 65 — find_orb's marker for
// "this observation is turned off" (mpc_obs.cpp: is_included = buff[64] != 'x'). The text
// is plain + selectable, so the user can highlight a subset and copy, or hit ⧉ copy for all.
function renderObsMpc80(p, base) {
  // obs80 lines live on the final-fit residuals; fall back to that map when replaying a
  // stage (whose snapshot rows carry residuals only). Keyed by obs index.
  const byI = new Map(p.residuals.map((r) => [r.i, r.obs80]));
  const rows = base.slice().sort((a, b) => (a.jd_utc || 0) - (b.jd_utc || 0) || a.i - b.i);
  const lines = rows.map((r) => {
    // A space-based / roving obs is a TWO-line record (RA/Dec line + 's'/'v' companion),
    // joined by a newline in the payload. The 'x' exclusion marker rides the FIRST line only.
    const parts = (r.obs80 || byI.get(r.i) || "").split("\n");
    let first = parts[0].padEnd(80, " ");
    if (!r.included) first = first.slice(0, 64) + "x" + first.slice(65);  // col 65 (0-idx 64)
    const block = [first, ...parts.slice(1).map((l) => l.padEnd(80, " "))]
      .map((l) => l.replace(/\s+$/, "")).join("\n");
    return r.included ? _esc(block) : `<span class="exline">${_esc(block)}</span>`;
  });
  const nExcl = rows.filter((r) => !r.included).length;
  $("obsCount").textContent = `(${rows.length} obs` + (nExcl ? `, ${nExcl} excluded — 'x' in col 65` : "") + ")";
  $("obsScroll").innerHTML = `<pre class="mpc80">${lines.join("\n")}</pre>`;
}

function renderObservatories(p) {
  const obs = p.observatories.slice().sort((a, b) => b.n - a.n);
  $("obsCodeCount").textContent = `(${obs.length} codes)`;
  const head = `<table class="obs" style="font-size:11px"><thead><tr>
    <th class="l">code</th><th class="l">name</th><th>obs</th><th>used</th><th>median resid″</th></tr></thead><tbody>`;
  $("codeBody").innerHTML = head + obs.map((c) => {
    const m = c.median_resid;
    const col = m === null ? "var(--muted)" : m > 1.0 ? "var(--warn)" : "var(--fg)";
    const nm = c.name || "";
    // Name truncates with an ellipsis (CSS: td.name) rather than wrapping; the
    // full name is on the title tooltip so nothing is lost when it doesn't fit.
    return `<tr><td class="l">${c.obscode}</td>` +
      `<td class="l name" title="${escAttr(nm)}">${escHtml(nm)}</td>` +
      `<td>${c.n}</td><td>${c.n_used}</td>` +
      `<td style="color:${col}">${m === null ? "—" : fmt(m, 2)}</td></tr>`;
  }).join("") + "</tbody></table>";
}

// ---- O−C residual plot (SVG) --------------------------------------------
// Squeeze the time axis: real spacing is kept inside each observing arc, but any
// empty span longer than OC_GAP_CAP is collapsed to OC_GAP_CAP. Otherwise a
// two-apparition object's months-long opposition gap swallows the whole pane and
// each arc shrinks to a sliver. Each squeeze is flagged with a dashed break.
const OC_GAP_CAP = 4 / 24;   // days (4 hours)
// A void longer than this (≈ a full opposition gap, not mere inter-night) is an
// apparition boundary: in the O−C plot it gets the solid whiter divider, and in the
// observations table the row separator across it is whitened to match. Shared so the
// two panes stay in lock-step — change the threshold here and both follow.
const APPARITION_GAP_DAYS = 90;   // days

function renderOCPlot(p) {
  const box = $("ocBody"), W = box.clientWidth || 400, H = box.clientHeight || 200;
  const m = { l: 34, r: 8, t: 8, b: 20 };
  // Flag the pane title red when the engine reports a systematic within-apparition
  // residual trend past its flagging threshold (the tell of a bad linkage).
  const sys = p.summary && p.summary.systematic;
  const head = $("ocHead");
  if (head) { head.classList.toggle("sysflag", !!sys); head.title = sys || ""; }
  if (!p.residuals.length) { box.innerHTML = ""; return; }
  // residual value per component in the active unit: raw O−C arcsec, or normalised
  // by the per-obs σ (so 1 = a 1σ miss — surfaces obs that are big only because their
  // σ is tight, and vice-versa). σ_ra/σ_dec ride on every residual in the payload.
  const sig = ocUnit === "sigma";
  const vRA = (r) => sig ? (r.sigma_ra ? r.d_ra_arcsec / r.sigma_ra : 0) : r.d_ra_arcsec;
  const vDec = (r) => sig ? (r.sigma_dec ? r.d_dec_arcsec / r.sigma_dec : 0) : r.d_dec_arcsec;
  const cap = sig ? 5 : 6, floor = sig ? 2 : 0.5;
  // Scale on the INCLUDED observations only -- an excluded/toggled-off obs is by
  // definition off the fit, and letting it set the range flattens everything the
  // orbit actually used. Excluded points are still drawn, clamped to the edge.
  let scaled = p.residuals.filter((r) => r.included);
  if (!scaled.length) scaled = p.residuals;   // nothing included: fall back to all
  let amax = floor;
  // Scale on the SHOWN axes only -- with Dec hidden, a big Dec miss must not keep
  // squashing the RA points the user asked to look at.
  for (const r of scaled) {
    if (ocAxes.ra) amax = Math.max(amax, Math.abs(vRA(r) || 0));
    if (ocAxes.dec) amax = Math.max(amax, Math.abs(vDec(r) || 0));
  }
  amax = Math.min(amax, cap);  // clamp so a few wild outliers don't flatten the rest

  // Piecewise jd -> axis-day map, with long voids compressed to OC_GAP_CAP.
  const uniq = [...new Set(p.residuals.map((r) => r.jd_utc))].sort((a, b) => a - b);
  const axOf = new Map([[uniq[0], 0]]);
  const breaks = [];   // {ax, days} — a compressed void and how long it really was
  let cum = 0;
  for (let i = 1; i < uniq.length; i++) {
    let dt = uniq[i] - uniq[i - 1];
    if (dt > OC_GAP_CAP) { breaks.push({ ax: cum + OC_GAP_CAP / 2, days: dt }); dt = OC_GAP_CAP; }
    cum += dt;
    axOf.set(uniq[i], cum);
  }
  const axSpan = cum || 1;
  const x = (jd) => m.l + (axOf.get(jd) / axSpan) * (W - m.l - m.r);
  const y = (a) => m.t + (1 - (a + amax) / (2 * amax)) * (H - m.t - m.b);

  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  // axes
  svg += `<line x1="${m.l}" y1="${y(0)}" x2="${W - m.r}" y2="${y(0)}" stroke="#30363d"/>`;
  svg += `<line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H - m.b}" stroke="#30363d"/>`;
  const unit = sig ? "σ" : "″", prec = sig ? 0 : 1;
  for (const t of [-amax, 0, amax]) {
    const lbl = t === 0 ? "0" : t.toFixed(prec) + unit;
    svg += `<text x="2" y="${y(t) + 3}" fill="#8b949e" font-size="9">${lbl}</text>`;
  }
  // compressed-gap break markers. A long void (> APPARITION_GAP_DAYS, i.e. a separate
  // apparition rather than just inter-night) gets a solid, whiter line so opposition
  // breaks read at a glance; shorter gaps stay a faint dashed grey.
  for (const b of breaks) {
    const bx = (m.l + (b.ax / axSpan) * (W - m.l - m.r)).toFixed(1);
    const lbl = b.days >= 300 ? (b.days / 365.25).toFixed(1) + "y"
      : b.days >= 1 ? Math.round(b.days) + "d" : Math.round(b.days * 24) + "h";
    const major = b.days > APPARITION_GAP_DAYS;
    const stroke = major ? `stroke="#adbac7"` : `stroke="#444c56" stroke-dasharray="2 3"`;
    svg += `<line x1="${bx}" y1="${m.t}" x2="${bx}" y2="${H - m.b}" ${stroke}/>`;
    svg += `<text x="${bx}" y="${m.t + 8}" fill="${major ? "#adbac7" : "#6e7681"}" font-size="8" text-anchor="middle">⇥${lbl}</text>`;
  }
  // points
  for (const r of p.residuals) {
    const op = r.included ? 0.85 : 0.25, rad = r.included ? 1.7 : 1.4;
    if (ocAxes.ra)
      svg += `<circle cx="${x(r.jd_utc).toFixed(1)}" cy="${y(Math.max(-amax, Math.min(amax, vRA(r)))).toFixed(1)}" r="${rad}" fill="var(--ra)" opacity="${op}"/>`;
    if (ocAxes.dec)
      svg += `<circle cx="${x(r.jd_utc).toFixed(1)}" cy="${y(Math.max(-amax, Math.min(amax, vDec(r)))).toFixed(1)}" r="${rad}" fill="var(--dec)" opacity="${op}"/>`;
  }
  svg += `<text x="${m.l}" y="${H - 6}" fill="#8b949e" font-size="9">${jdToDate(uniq[0]).slice(0,10)}</text>`;
  svg += `<text x="${W - m.r}" y="${H - 6}" fill="#8b949e" font-size="9" text-anchor="end">${jdToDate(uniq[uniq.length-1]).slice(0,10)}</text>`;
  svg += `</svg>`;
  box.innerHTML = svg;
}

// ---- orbital geometry shared by the asteroid and the planets -------------
// JD <-> JS Date (Unix epoch is JD 2440587.5).
const jdToJsDate = (jd) => new Date((jd - 2440587.5) * 86400000);
const jsDateToJd = (d) => d.getTime() / 86400000 + 2440587.5;
const SCRUB_HALF = 730.5;   // slider half-width in days (±2 years around the anchor)
// Camera-elevation clamp shared by both 3D views. el=0 is top-down (looking
// straight down the ecliptic pole, 90° above the plane); el=-90° would be edge-on
// (0° above the plane). We forbid the last 3°: the floor is 3° above the ecliptic,
// so the orbits never collapse to a degenerate flat line. el ∈ [EL_MIN, 0].
const EL_MIN = -(90 - 3) * Math.PI / 180;   // 3° above the ecliptic plane

// Classical elements -> heliocentric ecliptic xyz (mirrors engine_api._rotate_to_ecliptic).
function rotEcl(xo, yo, OmegaDeg, omegaDeg, iDeg) {
  const O = OmegaDeg * Math.PI / 180, w = omegaDeg * Math.PI / 180, i = iDeg * Math.PI / 180;
  const cO = Math.cos(O), sO = Math.sin(O), cw = Math.cos(w), sw = Math.sin(w),
        ci = Math.cos(i), si = Math.sin(i);
  const px = cO * cw - sO * sw * ci, py = sO * cw + cO * sw * ci, pz = sw * si;
  const qx = -cO * sw - sO * cw * ci, qy = -sO * sw + cO * cw * ci, qz = cw * si;
  return [px * xo + qx * yo, py * xo + qy * yo, pz * xo + qz * yo];
}
function keplerXYZ(a, e, iDeg, nodeDeg, periDeg, Mdeg) {
  if (e > 1) {
    // Unbound (1I, 2I, 3I): M = e sinh H - H is signed and never wrapped (negative before
    // perihelion), a < 0. Same solve as engine_api._orbit_path's hyperbolic branch, so the
    // dot the slider moves stays on the path the server sampled.
    const Mr = Mdeg * Math.PI / 180, s = -a;
    let H = Math.asinh(Mr / e);
    for (let k = 0; k < 60; k++) {
      const dH = (e * Math.sinh(H) - H - Mr) / (e * Math.cosh(H) - 1);
      H -= dH; if (Math.abs(dH) < 1e-11) break;
    }
    return rotEcl(s * (e - Math.cosh(H)), s * Math.sqrt(e * e - 1) * Math.sinh(H),
      nodeDeg, periDeg, iDeg);
  }
  let M = (((Mdeg % 360) + 360) % 360) * Math.PI / 180;
  if (M > Math.PI) M -= 2 * Math.PI;
  let E = M;
  for (let k = 0; k < 60; k++) {
    const dE = (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    E -= dE; if (Math.abs(dE) < 1e-11) break;
  }
  return rotEcl(a * (Math.cos(E) - e), a * Math.sqrt(1 - e * e) * Math.sin(E),
    nodeDeg, periDeg, iDeg);
}

const ORBIT_COLOR = "#56d4dd";                 // the asteroid orbit's "above-plane" colour
const ORBIT_COLOR_BELOW = darkenHex(ORBIT_COLOR, 0.31);   // below-ecliptic half: 25% toward black

// Darken a #rrggbb colour toward black by `frac` (0..1). Used to shade the part of
// the orbit that dips below the ecliptic plane a touch darker than the lit half.
function darkenHex(hex, frac) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.round(((n >> 16) & 255) * (1 - frac));
  const g = Math.round(((n >> 8) & 255) * (1 - frac));
  const b = Math.round((n & 255) * (1 - frac));
  return `rgb(${r},${g},${b})`;
}

// Split a sampled orbit polyline (heliocentric ecliptic xyz, z = height above the
// ecliptic) into runs that are above vs below the plane, inserting the exact z=0
// crossing point at each boundary, and report those crossings as the orbit's nodes.
// Pure geometry (no canvas) so it can be unit-tested. Returns:
//   { segments: [{pts:[[x,y,z]…], above:bool}], nodes: [{pos:[x,y,0], ascending}],
//     maxAbsZ }   (ascending = body climbs south→north through the plane).
function splitOrbitByEcliptic(path, closed = true) {
  const segments = [], nodes = [];
  let maxAbsZ = 0;
  for (const p of path) maxAbsZ = Math.max(maxAbsZ, Math.abs(p[2]));
  const n = path.length;
  if (n < 2) return { segments, nodes, maxAbsZ };
  // A CLOSED ellipse (engine samples E over a full 0..2π) is walked over its wrap edge
  // too (i==n -> back to path[0]), since a node can land on the seam. An OPEN curve --
  // the hyperbolic branch of an unbound orbit -- must NOT be: walking the seam there
  // drew a straight chord from the inbound end to the outbound end of the hyperbola.
  const last = closed ? n : n - 1;
  let above = path[0][2] >= 0;
  let cur = [path[0]];
  for (let i = 1; i <= last; i++) {
    const prev = path[i - 1], p = path[i % n];
    const pAbove = p[2] >= 0;
    if (pAbove === above) { if (i < n) cur.push(p); continue; }
    // sign flip between prev and p -> interpolate the z=0 crossing (a node)
    const t = prev[2] / (prev[2] - p[2]);
    const m = [prev[0] + (p[0] - prev[0]) * t, prev[1] + (p[1] - prev[1]) * t, 0];
    cur.push(m);
    segments.push({ pts: cur, above });
    nodes.push({ pos: m, ascending: prev[2] < 0 });   // climbing up through plane = ascending
    cur = [m]; if (i < n) cur.push(p);
    above = pAbove;
  }
  if (cur.length > 1) segments.push({ pts: cur, above });
  return { segments, nodes, maxAbsZ };
}

// JPL "Keplerian elements and rates" (J2000, valid 1800–2050). Fields: a[au], e,
// I[deg], L[deg] (mean longitude), w=ϖ (longitude of perihelion), O=Ω (node),
// each with its per-century rate. https://ssd.jpl.nasa.gov/planets/approx_pos.html
const PLANETS = [
  { name: "Mercury", color: "#9c8a7a", core: false,
    a: [0.38709927, 0.00000037], e: [0.20563593, 0.00001906], I: [7.00497902, -0.00594749],
    L: [252.25032350, 149472.67411175], w: [77.45779628, 0.16047689], O: [48.33076593, -0.12534081] },
  { name: "Venus", color: "#d8b36a", core: false,
    a: [0.72333566, 0.00000390], e: [0.00677672, -0.00004107], I: [3.39467605, -0.00078890],
    L: [181.97909950, 58517.81538729], w: [131.60246718, 0.00268329], O: [76.67984255, -0.27769418] },
  { name: "Earth", color: "#3b82c4", core: true,
    a: [1.00000261, 0.00000562], e: [0.01671123, -0.00004392], I: [-0.00001531, -0.01294668],
    L: [100.46457166, 35999.37244981], w: [102.93768193, 0.32327364], O: [0.0, 0.0] },
  { name: "Mars", color: "#c1502e", core: true,
    a: [1.52371034, 0.00001847], e: [0.09339410, 0.00007882], I: [1.84969142, -0.00813131],
    L: [-4.55343205, 19140.30268499], w: [-23.94362959, 0.44441088], O: [49.55953891, -0.29257343] },
  { name: "Jupiter", color: "#c2a06a", core: true,
    a: [5.20288700, -0.00011607], e: [0.04838624, -0.00013253], I: [1.30439695, -0.00183714],
    L: [34.39644051, 3034.74612775], w: [14.72847983, 0.21252668], O: [100.47390909, 0.20469106] },
  { name: "Saturn", color: "#d9c179", core: false,
    a: [9.53667594, -0.00125060], e: [0.05386179, -0.00050991], I: [2.48599187, 0.00193609],
    L: [49.95424423, 1222.49362201], w: [92.59887831, -0.41897216], O: [113.66242448, -0.28867794] },
  { name: "Uranus", color: "#7fd4d4", core: false,
    a: [19.18916464, -0.00196176], e: [0.04725744, -0.00004397], I: [0.77263783, -0.00242939],
    L: [313.23810451, 428.48202785], w: [170.95427630, 0.40805281], O: [74.01692503, 0.04240589] },
  { name: "Neptune", color: "#4f6fd4", core: false,
    a: [30.06992276, 0.00026291], e: [0.00859048, 0.00005105], I: [1.77004347, 0.00035372],
    L: [-55.12002969, 218.45945325], w: [44.96476227, -0.32241464], O: [131.78422574, -0.00508664] },
];
function planetElems(p, jd) {
  const T = (jd - 2451545.0) / 36525.0, at = (f) => f[0] + f[1] * T;
  const a = at(p.a), e = at(p.e), I = at(p.I), L = at(p.L), w = at(p.w), O = at(p.O);
  return { a, e, I, O, argPeri: w - O, M: L - w };
}
function planetXYZ(p, jd) { const k = planetElems(p, jd); return keplerXYZ(k.a, k.e, k.I, k.O, k.argPeri, k.M); }
// 288 segments (was 96): at Orbarium's tight NEO-encounter zoom levels (span down to
// ~0.05 au against Earth's ~1 au orbit) the straight chords between sample points were
// visibly off the true ellipse -- the drawn dot (computed exactly, not sampled) looked
// off its own orbit line. 3x the segments cuts the polyline's sagitta ~9x (it scales as
// the square of the per-segment angle), invisible even at the closest zoom; imperceptible
// extra cost since this is 4 planets x ~a few hundred points per redraw.
const PLANET_ORBIT_SEGMENTS = 288;
function planetPath(p, jd) {
  const k = planetElems(p, jd), pts = [];
  const n = PLANET_ORBIT_SEGMENTS;
  for (let j = 0; j <= n; j++) pts.push(keplerXYZ(k.a, k.e, k.I, k.O, k.argPeri, j * 360 / n));
  return pts;
}

// ---- shared orthographic 3-D camera ---------------------------------------
// Base for the two canvas scenes that draw the same heliocentric space:
//   • Orbit3D  — the orbit itself (Sun-centred ellipses, body scrubbed to a date)
//   • Orbarium — that same space drawn as Earth→object observation sightlines
// It owns the camera (azimuth/elevation/zoom about an optional `center`) and all
// pointer/touch input; subclasses only implement draw(). Keeping the input wiring,
// the orthographic projection and the per-frame setup here means a camera fix lands
// in one place for both views (they previously diverged by copy-paste).
class OrthoView {
  constructor(canvas, { zoomMin = 0.2, zoomMax = 8 } = {}) {
    this.c = canvas; this.ctx = canvas.getContext("2d");
    this.az = -0.6; this.el = -0.5; this.zoom = 1;
    this.center = [0, 0, 0];          // world point the camera rotates about (Orbit3D leaves it at origin)
    this.span = 1;                    // half-extent (au) the frame is scaled to fit
    this._zoomMin = zoomMin; this._zoomMax = zoomMax;
    this.dragging = false; this.lx = 0; this.ly = 0;
    canvas.addEventListener("mousedown", (e) => { this.dragging = true; this.lx = e.clientX; this.ly = e.clientY; });
    window.addEventListener("mouseup", () => { this.dragging = false; });
    window.addEventListener("mousemove", (e) => {
      if (!this.dragging) return;
      this._rotateBy(e.clientX - this.lx, e.clientY - this.ly);
      this.lx = e.clientX; this.ly = e.clientY; this.draw();
    });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault(); this._zoomBy(e.deltaY < 0 ? 1.12 : 0.89); this.draw();
    }, { passive: false });
    // touch: one finger rotates, two fingers pinch-zoom (mouse events don't fire on touch)
    const tdist = (e) => Math.hypot(e.touches[0].clientX - e.touches[1].clientX,
      e.touches[0].clientY - e.touches[1].clientY);
    let pinch = 0;
    canvas.addEventListener("touchstart", (e) => {
      if (e.touches.length === 1) { this.dragging = true; this.lx = e.touches[0].clientX; this.ly = e.touches[0].clientY; }
      else if (e.touches.length === 2) { this.dragging = false; pinch = tdist(e); }
    }, { passive: true });
    canvas.addEventListener("touchmove", (e) => {
      if (e.touches.length === 1 && this.dragging) {
        this._rotateBy(e.touches[0].clientX - this.lx, e.touches[0].clientY - this.ly);
        this.lx = e.touches[0].clientX; this.ly = e.touches[0].clientY; this.draw(); e.preventDefault();
      } else if (e.touches.length === 2 && pinch) {
        const d = tdist(e); this._zoomBy(d / pinch); pinch = d; this.draw(); e.preventDefault();
      }
    }, { passive: false });
    canvas.addEventListener("touchend", () => { this.dragging = false; pinch = 0; });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  // drag → rotate; elevation clamped between 3° above the ecliptic (EL_MIN) and top-down (0).
  _rotateBy(dxPx, dyPx) {
    this.az += dxPx * 0.01; this.el += dyPx * 0.01;
    this.el = Math.max(EL_MIN, Math.min(0, this.el));
  }
  _zoomBy(f) { this.zoom = Math.max(this._zoomMin, Math.min(this._zoomMax, this.zoom * f)); }

  // orthographic: translate by −center, rotate (az then el), drop depth.
  project(p) {
    const x = p[0] - this.center[0], y = p[1] - this.center[1], z = p[2] - this.center[2];
    const ca = Math.cos(this.az), sa = Math.sin(this.az);
    const x1 = x * ca + y * sa, y1 = -x * sa + y * ca;
    const ce = Math.cos(this.el), se = Math.sin(this.el);
    return [x1, y1 * ce - z * se];
  }

  // Size the canvas (HiDPI), clear it, and return the screen-space helpers both
  // scenes draw through — or null if the canvas has no box yet.
  _frame() {
    const c = this.c, dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return null;
    c.width = w * dpr; c.height = h * dpr;
    const ctx = this.ctx; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2;
    const scale = (Math.min(w, h) / 2 / this.span) * this.zoom * 0.9;
    const S = (p) => { const q = this.project(p); return [cx + q[0] * scale, cy - q[1] * scale]; };
    const stroke = (pts) => { ctx.beginPath(); pts.forEach((p, k) => { const q = S(p); k ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }); ctx.stroke(); };
    return { ctx, w, h, cx, cy, scale, S, stroke };
  }

  // Each planet at `jd`: faint orbit ellipse + a filled dot, optionally labelled.
  _drawPlanets(f, jd, { planets = PLANETS, alpha = 0.3, dotR = 2.5, label = false } = {}) {
    const { ctx, S, stroke } = f;
    ctx.lineWidth = 1; ctx.font = "10px sans-serif";
    for (const p of planets) {
      ctx.strokeStyle = p.color; ctx.globalAlpha = alpha; stroke(planetPath(p, jd));
      ctx.globalAlpha = 1;
      const q = S(planetXYZ(p, jd));
      ctx.fillStyle = p.color; ctx.beginPath(); ctx.arc(q[0], q[1], dotR, 0, 7); ctx.fill();
      if (label) { ctx.fillStyle = "#8b949e"; ctx.fillText(p.name, q[0] + 4, q[1] - 3); }
    }
  }

  draw() { /* subclasses implement */ }
}

// ---- Orbit3D: the heliocentric orbit (orthographic, draggable, date-scrubbable).
// Sun-centred; the body is advanced to a scrubbable date. (The Orbarium view further
// down shows the SAME space from the observer's side, as Earth→object sightlines.)
class Orbit3D extends OrthoView {
  constructor(canvas, data, legendEl) {
    super(canvas, { zoomMin: 0.2, zoomMax: 8 });
    this.legend = legendEl;
    this.full = false; this.jd = null;
    this._wireControls();
    this.setData(data);
  }

  // --- time scrubber + fullscreen wiring ---
  // The slider is a ±2yr window CENTRED on an anchor date; the calendar box shows
  // the current date, and the Fit-epoch/Now buttons light up when selected.
  // Buttons / date-pick recentre the window (slider back to middle); dragging the
  // slider moves within the window without moving the anchor.
  _wireControls() {
    this.slider = $("tSlider"); this.dateIn = $("tDate");
    this.slider.addEventListener("input", () => {
      this.jd = this.anchorJd + ((this.slider.value - 500) / 500) * SCRUB_HALF;
      this._refresh();
    });
    this.dateIn.addEventListener("change", () => {
      if (!this.dateIn.value) return;
      const [y, m, d] = this.dateIn.value.split("-").map(Number);
      this._anchor(jsDateToJd(new Date(Date.UTC(y, m - 1, d, 12))));
    });
    $("tEpoch").addEventListener("click", () => { if (this.data) this._anchor(this.data.epoch_jd); });
    $("tNow").addEventListener("click", () => this._anchor(jsDateToJd(new Date())));
    $("fsBtn").addEventListener("click", () => {
      const el = $("viz3d");
      if (!document.fullscreenElement) {
        if (el.requestFullscreen) el.requestFullscreen();
        _ga("fullscreen_orbit", {});
      }
      else document.exitFullscreen();
    });
    document.addEventListener("fullscreenchange", () => {
      this.full = document.fullscreenElement === $("viz3d");
      requestAnimationFrame(() => this.draw());
    });
  }

  setData(data) {
    this.data = data;
    // Frame the aphelion; an unbound orbit has none, so frame the sampled branch's far end.
    let reach = data ? (data.Q || 0) : 0;
    if (data && !reach && data.path) for (const p of data.path) reach = Math.max(reach, Math.hypot(p[0], p[1], p[2]));
    this.span = data ? Math.max(reach, 1.6) * 1.15 : 3;
    this._anchor(data ? data.epoch_jd : jsDateToJd(new Date()));   // default: fit epoch, centred
  }

  // recentre the ±2yr scrubber on `jd` (buttons / date pick / new fit)
  _anchor(jd) {
    this.anchorJd = jd; this.jd = jd;
    if (this.slider) this.slider.value = 500;
    this._refresh();
  }

  // reflect this.jd into the calendar box + button highlights, then redraw
  _refresh() {
    if (this.dateIn) this.dateIn.value = jdToJsDate(this.jd).toISOString().slice(0, 10);
    const e = this.data ? this.data.epoch_jd : null, now = jsDateToJd(new Date());
    const ep = $("tEpoch"), nw = $("tNow");
    if (ep) ep.classList.toggle("active", e !== null && Math.abs(this.jd - e) < 0.5);
    if (nw) nw.classList.toggle("active", Math.abs(this.jd - now) < 0.5);
    this.draw();
  }

  // asteroid position at the selected date (two-body advance of M from epoch; the
  // hyperbolic solve for an unbound orbit -- see objPosAt)
  asteroidPos() {
    const d = this.data;
    if (!d || !d.path) return null;
    return objPosAt(d, this.jd) || d.position;
  }

  // Draw the asteroid's orbit, shading the half BELOW the ecliptic plane darker.
  // (The ecliptic crossings are still computed to split the colour, but the node
  // markers/labels themselves are intentionally not drawn.)
  _drawOrbitWithNodes(f) {
    const { ctx, S } = f;
    const path = this.data.path;
    const { segments, maxAbsZ } = splitOrbitByEcliptic(path, this.data.closed !== false);
    ctx.lineWidth = 1.8; ctx.globalAlpha = 1;
    // Near-coplanar orbit (i≈0): z is all numerical noise — draw flat.
    if (maxAbsZ < this.span * 0.002) {
      ctx.strokeStyle = ORBIT_COLOR; f.stroke(path); return;
    }
    for (const seg of segments) {
      ctx.strokeStyle = seg.above ? ORBIT_COLOR : ORBIT_COLOR_BELOW;
      ctx.beginPath();
      seg.pts.forEach((p, k) => { const q = S(p); k ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
      ctx.stroke();
    }
  }

  draw() {
    const f = this._frame();
    if (!f) return;
    const { ctx, S, stroke } = f;

    // XYZ ecliptic axes (fullscreen only). +X points to the vernal equinox, i.e.
    // the First Point of Aries ♈ — the zero point of ecliptic J2000; Y/Z annotated
    // so the triad reads as an orientation legend.
    if (this.full) {
      const L = this.span * 0.95;
      const axes = [
        [[L, 0, 0], "#f0596b", "X", "♈ First Point of Aries"],
        [[0, L, 0], "#5fbf6b", "Y", ""],
        [[0, 0, L], "#5f8bff", "Z", "ecliptic north pole"],
      ];
      ctx.lineWidth = 1;
      for (const [v, col, lab, desc] of axes) {
        ctx.strokeStyle = col; ctx.globalAlpha = 0.55; stroke([[0, 0, 0], v]);
        const e = S(v);
        ctx.globalAlpha = 0.95; ctx.fillStyle = col;
        ctx.font = "bold 12px monospace"; ctx.fillText(lab, e[0] + 4, e[1]);
        ctx.globalAlpha = 0.75; ctx.fillStyle = "#8b949e";
        ctx.font = "10px sans-serif"; ctx.fillText(desc, e[0] + 4, e[1] + 12);
      }
      ctx.globalAlpha = 1;
    }

    // planets: their orbits (faint ellipses) + a dot at the selected date.
    const shown = PLANETS.filter((p) => (this.full || p.core) && p.a[0] < this.span * 1.5);
    this._drawPlanets(f, this.jd, { planets: shown, alpha: 0.32, dotR: 2.5, label: this.full });

    // Sun
    const sun = S([0, 0, 0]);
    ctx.fillStyle = "#ffd33d"; ctx.beginPath(); ctx.arc(sun[0], sun[1], 4, 0, 7); ctx.fill();

    // asteroid orbit + dot (larger than the planet dots). The half of the ellipse
    // BELOW the ecliptic is drawn a touch darker, and the two ecliptic crossings
    // (the ascending ☊ / descending ☋ nodes) are marked.
    if (this.data && this.data.path) {
      this._drawOrbitWithNodes(f);
      const ap = this.asteroidPos();
      if (ap) {
        const bp = S(ap);
        ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.arc(bp[0], bp[1], 3.5, 0, 7); ctx.fill();
        ctx.strokeStyle = ORBIT_COLOR; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(bp[0], bp[1], 6, 0, 7); ctx.stroke();
      }
    }

    if (this.legend) {
      this.legend.innerHTML =
        `<div><i style="background:#56d4dd"></i> ${CURRENT ? CURRENT.object : "asteroid"}</div>` +
        shown.map((p) => `<div><i style="background:${p.color}"></i> ${p.name}</div>`).join("") +
        (this.full ? "" : `<div style="opacity:.6">⛶ for all planets</div>`);
    }
  }
}

// ---- Orbarium: observations as green Earth→object sightlines in 3D --------
const EARTH_PLANET = PLANETS.find((p) => p.name === "Earth");

// Object's heliocentric ecliptic xyz at an arbitrary date (two-body advance of M
// from the fit epoch). Serves Orbit3D's scrubbed dot and Orbarium's sightline tips.
// An unbound orbit advances the same way -- its M is signed and unwrapped, its mean
// motion sqrt(mu/(-a)^3) -- and keplerXYZ branches to the hyperbolic solve on e > 1,
// so 1I/2I/3I move under the slider like anything else. Without a mean motion the
// only honest answer is the epoch position.
function objPosAt(d, jd) {
  if (!d || d.a == null) return null;
  if (d.n_deg_per_day != null && d.e != null && d.M != null) {
    const Mt = d.M + d.n_deg_per_day * (jd - d.epoch_jd);
    return keplerXYZ(d.a, d.e, d.i, d.node, d.peri, Mt);
  }
  return d.position || null;          // degenerate: best we have
}

// Choose up to `target` observations to draw. Below `target` total, show every one of
// them -- a two-night arc covered by many observatories used to collapse to one
// sightline per NIGHT (underwhelming: 2 arrows for 20 real observations) even though
// 20 is well within a legible count. Above `target`, sample by TIME across the whole
// arc rather than by night: each of `target` equally-spaced instants across
// [first jd, last jd] claims its nearest not-yet-used observation. A short, dense arc
// (few nights, many observatories each) keeps several picks per night in proportion to
// how much of the arc that night covers, instead of flattening to one; a long,
// sparsely-sampled arc still spreads picks across its full time baseline as before.
function selectOrbariumObs(residuals, target = 15) {
  const obs = (residuals || [])
    .filter((r) => r.included && r.jd_utc != null)
    .map((r) => ({ jd: r.jd_utc, obscode: r.obscode }))
    .sort((a, b) => a.jd - b.jd);
  if (obs.length <= target) return obs;
  const t0 = obs[0].jd, t1 = obs[obs.length - 1].jd;
  const used = new Set();
  const picks = [];
  for (let k = 0; k < target; k++) {
    const tTarget = target === 1 ? t0 : t0 + (t1 - t0) * k / (target - 1);
    let best = -1, bestD = Infinity;
    for (let i = 0; i < obs.length; i++) {
      if (used.has(i)) continue;
      const d = Math.abs(obs[i].jd - tTarget);
      if (d < bestD) { bestD = d; best = i; }
    }
    used.add(best); picks.push(obs[best]);
  }
  return picks.sort((a, b) => a.jd - b.jd);
}

// distance (screen px) from point (px,py) to the segment (ax,ay)-(bx,by).
function segDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  let t = l2 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

// Orbarium: the observation-geometry counterpart to Orbit3D — the SAME heliocentric
// space and orthographic camera, but drawn from the observer's side, each observation
// a green Earth→object sightline. Allows a much wider zoom range to frame a single
// short apparition.
class Orbarium extends OrthoView {
  constructor(canvas, legendEl) {
    super(canvas, { zoomMin: 0.15, zoomMax: 40 });
    this.legend = legendEl;
    this.arrows = []; this.objPath = null;
    this.note = null; this.midJd = null;
    this.hover = null;                        // sightline under the cursor -> tooltip
    canvas.addEventListener("mousemove", (e) => this._onHover(e));
    canvas.addEventListener("mouseleave", () => { if (this.hover) { this.hover = null; this.draw(); } });
  }

  // Hover a green sightline -> show "OBSCODE YYYY-MM-DD" to the right of the cursor.
  _onHover(ev) {
    if (this.dragging) return;                // don't fight the rotate-drag
    const rect = this.c.getBoundingClientRect();
    const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    let best = null, bestD = 7;               // px pick radius
    for (const a of this.arrows) {
      if (!a.se || !a.so) continue;
      const d = segDist(mx, my, a.se[0], a.se[1], a.so[0], a.so[1]);
      if (d < bestD) { bestD = d; best = a; }
    }
    if (best) {
      this.hover = { arrow: best, x: mx, y: my,
        label: `${best.obscode || "???"} ${jdToDate(best.jd).slice(0, 10)}` };
      this.draw();
    } else if (this.hover) { this.hover = null; this.draw(); }
  }

  // Build the sightlines from a fit payload; recentre + reset the view to frame them.
  setData(payload) {
    this.az = -0.6; this.el = -0.5; this.zoom = 1;
    this.arrows = []; this.objPath = null; this.note = null; this.hover = null;
    const d = payload && payload.orbit3d;
    const picks = selectOrbariumObs(payload && payload.residuals);
    if (!picks.length) { this.note = "No observations to show."; this.draw(); return; }
    if (payload && payload.orbit_suppressed) {
      // Not "unconverged" -- the fit converged fine; the arc just does not say WHERE on the
      // sightlines the object is, which is the one thing this view exists to draw.
      this.note = "The arc does not determine an orbit, so there is nowhere to place the object.";
      this.draw(); return;
    }
    // An unbound orbit is admitted: objPosAt advances it by its own mean motion through the
    // hyperbolic solve, so its sightline tips land on the branch the server sampled.
    if (!d || d.a == null || d.e == null || d.M == null || d.n_deg_per_day == null) {
      this.note = "This view needs a converged orbit."; this.draw(); return;
    }
    this.objPath = d.path || null;
    const pts = [];
    for (const o of picks) {
      const earth = planetXYZ(EARTH_PLANET, o.jd);
      const obj = objPosAt(d, o.jd);
      if (!earth || !obj) continue;
      this.arrows.push({ earth, obj, jd: o.jd, obscode: o.obscode });
      pts.push(earth, obj);
    }
    if (!this.arrows.length) { this.note = "No observations to show."; this.draw(); return; }
    // observational centre = mean of every start and end point; span frames them all.
    const c = [0, 0, 0];
    for (const p of pts) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
    this.center = c.map((v) => v / pts.length);
    let r = 0;
    for (const p of pts) r = Math.max(r, Math.hypot(p[0] - this.center[0], p[1] - this.center[1], p[2] - this.center[2]));
    this.span = Math.max(r * 1.25, 0.05);
    this.midJd = this.arrows[Math.floor(this.arrows.length / 2)].jd;
    this.draw();
  }

  draw() {
    const f = this._frame();
    if (!f) return;
    const { ctx, w, h, S, stroke } = f;
    if (this.note) {
      ctx.fillStyle = "#8b949e"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
      ctx.fillText(this.note, w / 2, h / 2); ctx.textAlign = "left";
      if (this.legend) this.legend.innerHTML = "";
      return;
    }

    // context: all planet orbits + a dot at the obs epoch, then the object's orbit.
    this._drawPlanets(f, this.midJd, { alpha: 0.28, dotR: 2.2, label: true });
    if (this.objPath) { ctx.strokeStyle = "#56d4dd"; ctx.globalAlpha = 0.28; stroke(this.objPath); }
    ctx.globalAlpha = 1;

    // Sun (often off-canvas when zoomed onto a single apparition — that's fine).
    const sun = S([0, 0, 0]);
    ctx.fillStyle = "#ffd33d"; ctx.beginPath(); ctx.arc(sun[0], sun[1], 3.5, 0, 7); ctx.fill();

    // the sightlines: Earth dot -> green arrow -> object, with an arrowhead at the object.
    for (const a of this.arrows) {
      const e = S(a.earth), o = S(a.obj);
      a.se = e; a.so = o;                     // cache for hover hit-testing
      const hot = this.hover && this.hover.arrow === a;
      ctx.strokeStyle = hot ? "#7bed9f" : "#34d058"; ctx.lineWidth = hot ? 2.4 : 1.4; ctx.globalAlpha = 0.92;
      ctx.beginPath(); ctx.moveTo(e[0], e[1]); ctx.lineTo(o[0], o[1]); ctx.stroke();
      const dx = o[0] - e[0], dy = o[1] - e[1], L = Math.hypot(dx, dy) || 1;
      const ux = dx / L, uy = dy / L, hl = 7, hw = 3.5;
      ctx.fillStyle = "#34d058"; ctx.beginPath();
      ctx.moveTo(o[0], o[1]);
      ctx.lineTo(o[0] - ux * hl - uy * hw, o[1] - uy * hl + ux * hw);
      ctx.lineTo(o[0] - ux * hl + uy * hw, o[1] - uy * hl - ux * hw);
      ctx.closePath(); ctx.fill();
      ctx.globalAlpha = 1;
      ctx.fillStyle = EARTH_PLANET.color; ctx.beginPath(); ctx.arc(e[0], e[1], 2.2, 0, 7); ctx.fill();
    }
    // the object's positions (arrow tips) over the arc, on top.
    ctx.fillStyle = "#ffffff";
    for (const a of this.arrows) { const o = S(a.obj); ctx.beginPath(); ctx.arc(o[0], o[1], 1.8, 0, 7); ctx.fill(); }

    // hover tooltip: "OBSCODE YYYY-MM-DD" to the right of the cursor.
    if (this.hover && this.hover.label) {
      ctx.font = "11px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
      const tw = ctx.measureText(this.hover.label).width;
      let tx = this.hover.x + 12;
      if (tx + tw + 6 > w) tx = this.hover.x - 12 - tw - 6;   // flip left near the right edge
      const ty = Math.max(11, Math.min(h - 4, this.hover.y));
      ctx.fillStyle = "rgba(0,0,0,0.72)"; ctx.fillRect(tx - 3, ty - 11, tw + 6, 15);
      ctx.fillStyle = "#e6edf3"; ctx.fillText(this.hover.label, tx, ty);
    }

    if (this.legend) {
      this.legend.innerHTML =
        `<div><i style="background:#34d058"></i> sightline (${this.arrows.length})</div>` +
        `<div><i style="background:#56d4dd"></i> ${CURRENT ? CURRENT.object : "object"}</div>` +
        PLANETS.map((pl) => `<div><i style="background:${pl.color}"></i> ${pl.name}</div>`).join("") +
        `<div style="opacity:.6">drag: rotate · scroll: zoom</div>`;
    }
  }
}

let orbarium = null;

// ---- central pane picker: Observations | Orbarium | View Ephemeris -------
let centralPane = "obs";                 // "obs" | "orbarium" | "ephem" | "impact"
let ephemRows = [];                      // last fetched ephemeris rows
let ephemSel = -1;                       // selected row index (drives the sky plot)
let ephemDialogMode = "export";          // the shared ephem dialog: "export" tab | "pane" refresh
// startJD null -> today. faint: V-mag cut-off (null = show every epoch); unobs: drop the
// epochs the observatory can't shoot (find_orb's "suppress unobservables").
const ephemParams = { startJD: null, step: 1, n: 30, site: "500", faint: null, unobs: false };
let ephemHiddenNote = "";                // what the last fetch's display cuts hid, if anything

const _DAYMS = 86400000, _JD_UNIX = 2440587.5;
function _todayJD() {
  return Date.parse(new Date().toISOString().slice(0, 10) + "T00:00:00Z") / _DAYMS + _JD_UNIX;
}
function _jdToISODate(jd) {
  return new Date((jd - _JD_UNIX) * _DAYMS).toISOString().slice(0, 10);
}

// ---- central-pane registry ------------------------------------------------
// One entry per <option> in #paneSelect. A pane is just two elements in the DOM
// carrying its key as data-pane — a control strip in the panel header
// (.panectl) and a body (.panebody) — plus, optionally, an `onShow` hook and
// side panels it owns elsewhere in the layout (`aux`). setCentralPane() shows
// the matching pair and hides EVERY other one by that same rule, so adding a
// pane means adding a row here and the two tagged elements, and a button can
// never leak onto a pane it doesn't belong to. Individual buttons inside a
// strip still hide themselves on app state; that is a private matter for the
// pane that owns them.
const CENTRAL_PANES = {
  obs: {},
  orbarium: {
    onShow: () => {
      if (!CURRENT) return;
      if (!orbarium) orbarium = new Orbarium($("orbariumCanvas"), $("orbariumLegend"));
      orbarium.setData(CURRENT);
      requestAnimationFrame(() => orbarium.draw());   // canvas now has a real box
    },
  },
  ephem: { aux: ["skyview"], onShow: () => loadEphem() },   // on-sky pane lives with the ephemeris
  impact: { onShow: () => loadImpact() },
};

// Some pane-owned elements (the replay banner) are ALSO state-driven: their
// renderer decides whether they have anything to say. The pane framework owns
// the other half of that answer. Renderers call paneStateShow() instead of
// setting .hidden, and setCentralPane re-applies every remembered state, so
// switching panes can never strand one over the wrong content.
const _paneOwnedState = {};                 // element id -> renderer's answer
function _applyPaneOwned(id) {
  const el = $(id);
  el.hidden = !_paneOwnedState[id] || el.dataset.pane !== centralPane;
}
function paneStateShow(id, on) {
  _paneOwnedState[id] = !!on;
  _applyPaneOwned(id);
}

function setCentralPane(which) {
  if (!(which in CENTRAL_PANES)) which = "obs";
  centralPane = which;
  if ($("paneSelect").value !== which) $("paneSelect").value = which;
  for (const el of document.querySelectorAll(".panectl, .panebody"))
    el.hidden = el.dataset.pane !== which;
  for (const id of Object.keys(_paneOwnedState)) _applyPaneOwned(id);
  for (const [key, pane] of Object.entries(CENTRAL_PANES))
    for (const id of pane.aux || []) $(id).hidden = key !== which;
  $("obsCount").style.visibility = which === "obs" ? "visible" : "hidden";
  if (CENTRAL_PANES[which].onShow) CENTRAL_PANES[which].onShow();
  pushViewUrl(which);
}

function _ephemHandle() {
  // Prefer the live interactive session (reflects current toggles/model/epoch); fall
  // back to a pasted fit's export key, then to the designation(s).
  if (CURRENT.session) return `session=${encodeURIComponent(CURRENT.session)}`;
  if (CURRENT.fit_key) return `key=${encodeURIComponent(CURRENT.fit_key)}`;
  const ds = (CURRENT.designations && CURRENT.designations.length)
    ? CURRENT.designations : [CURRENT.object];
  return `desig=${encodeURIComponent(ds.join(","))}`;
}

// The two find_orb display cuts as query params, shared by the pane fetch and the text
// export so both hide exactly the same epochs. Blank faint limit = no limit.
function _ephemFilterQS(p) {
  return (p.faint == null ? "" : `&faint=${p.faint}`) + (p.unobs ? "&unobs=1" : "");
}

function updateEphemRangeLabel() {
  const p = ephemParams, jd = p.startJD == null ? _todayJD() : p.startJD;
  const cuts = (p.faint == null ? "" : ` · V≤${p.faint}`) + (p.unobs ? " · observable only" : "");
  $("ephemRangeLabel").textContent =
    `from ${_jdToISODate(jd)} · ${p.n}×${p.step}d · obs ${p.site}${cuts}`;
}

async function loadEphem() {
  if (!CURRENT) return;
  updateEphemRangeLabel();
  const p = ephemParams, startJD = p.startJD == null ? _todayJD() : p.startJD;
  const url = `/api/ephem?${_ephemHandle()}&start=${startJD}&step=${p.step}&n=${p.n}` +
              `&site=${encodeURIComponent(p.site)}` + _ephemFilterQS(p);
  $("ephemWrap").innerHTML = `<div id="ephemMsg"><span class="loading">computing ephemeris…</span></div>`;
  try {
    const r = await fetch(url);
    const data = await r.json();
    if (!data.ok) {
      ephemRows = [];
      $("ephemWrap").innerHTML = `<div id="ephemMsg">ephemeris failed: ${escHtml(data.error || "")}</div>`;
      return;
    }
    ephemRows = data.rows || [];
    ephemSel = -1;
    // Say what the cuts hid, so a short table never reads as a computation that failed.
    ephemHiddenNote = data.hidden_note || "";
    if (p.unobs && data.site_has_horizon === false) {
      ephemHiddenNote = `observatory ${p.site} has no horizon (geocentric or space-based) —` +
                        ` nothing to suppress`;
    }
    renderEphemTable();
  } catch (e) {
    $("ephemWrap").innerHTML = `<div id="ephemMsg">ephemeris request failed: ${escHtml(e.message)}</div>`;
  }
}

function renderEphemTable() {
  if (!ephemRows.length) {
    const why = ephemHiddenNote ? ` (${_esc(ephemHiddenNote)})` : "";
    $("ephemWrap").innerHTML =
      `<div id="ephemMsg">no ephemeris rows${why}.</div>`;
    return;
  }
  const note = ephemHiddenNote
    ? `<div class="eph-hidden">${_esc(ephemHiddenNote)}</div>` : "";
  // Altitude columns only exist when the fetch ran the topocentric observer pass (i.e.
  // suppression is on): show them then, so a surviving row visibly IS observable.
  const showAlt = ephemRows.some((x) => x.alt_deg != null);
  const altHead = showAlt
    ? `<th title="target altitude at the observatory (°)">alt</th>
       <th title="solar altitude at the observatory (°)">sun</th>` : "";
  const head = note + `<table class="obs"><thead><tr>
    <th class="l">date (UTC)</th><th>RA</th><th>Dec</th>
    <th title="distance from the observer (au)">Δ au</th>
    <th title="heliocentric distance (au)">r au</th>
    <th title="solar elongation (°)">elong</th>
    <th title="apparent V magnitude">V</th>${altHead}
    <th title="sky motion (arcsec / min)">″/min</th>
    <th title="motion position angle (° N→E)">mPA</th>
    <th title="1σ error-ellipse semi-major axis (arcsec)">σa″</th>
    <th title="1σ error-ellipse semi-minor axis (arcsec)">σb″</th>
    <th title="error-ellipse position angle (° N→E)">ePA</th>
  </tr></thead><tbody>`;
  const nCols = 12 + (showAlt ? 2 : 0);
  // A run of epochs the display cuts removed leaves a gap in the time series; mark it
  // with a break row (find_orb prints a row of dots) so consecutive table rows are never
  // mistaken for consecutive epochs.
  // (the reason wording comes from the server, so it can't drift from the text table)
  const breakRow = (b) =>
    `<tr class="eph-break"><td colspan="${nCols}">${b.n} hidden — ${_esc(b.why || "")}</td></tr>`;
  const body = ephemRows.map((x, i) => {
    const date = (x.date_iso || "").slice(0, 16);
    return (x.hidden_before ? breakRow(x.hidden_before) : "") +
      `<tr data-i="${i}"${i === ephemSel ? ' class="sel"' : ""}>
      <td class="l">${date}</td>
      <td>${x.ra_hms || (x.ra_deg != null ? raHMS(x.ra_deg) : "—")}</td>
      <td>${x.dec_dms || (x.dec_deg != null ? decDMS(x.dec_deg) : "—")}</td>
      <td>${fmt(x.delta_au, 4)}</td>
      <td>${fmt(x.r_au, 3)}</td>
      <td>${fmt(x.elong_deg, 1)}</td>
      <td>${x.v_mag == null ? "—" : fmt(x.v_mag, 1)}</td>${showAlt
        ? `<td>${fmt(x.alt_deg, 0)}</td><td>${fmt(x.sun_alt_deg, 0)}</td>` : ""}
      <td>${fmt(x.motion_arcsec_min, 2)}</td>
      <td>${fmt(x.motion_pa_deg, 1)}</td>
      <td>${fmt(x.unc_smaj_arcsec, 3)}</td>
      <td>${fmt(x.unc_smin_arcsec, 3)}</td>
      <td>${fmt(x.unc_pa_deg, 0)}</td></tr>` +
      (x.hidden_after ? breakRow(x.hidden_after) : "");
  }).join("");
  $("ephemWrap").innerHTML = head + body + "</tbody></table>";
  $("ephemWrap").querySelector("tbody").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-i]");
    if (tr) selectEphemRow(parseInt(tr.dataset.i, 10));
  });
}

function selectEphemRow(i) {
  if (i < 0 || i >= ephemRows.length) return;
  ephemSel = i;
  const tb = $("ephemWrap").querySelector("tbody");
  // select by data-i, NOT by position: break rows sit between the data rows.
  if (tb) tb.querySelectorAll("tr[data-i]").forEach(
    (tr) => tr.classList.toggle("sel", parseInt(tr.dataset.i, 10) === i));
  plotSky(ephemRows[i]);
}

// ---- impact-location pane (Phase 1: where/when it reaches the ground) -----
async function loadImpact() {
  if (!CURRENT) return;
  $("impactWrap").innerHTML = `<div class="impact-msg"><span class="loading">propagating forward — locating any Earth impact…</span></div>`;
  try {
    const r = await fetch(`/api/impact?${_ephemHandle()}`);
    const data = await r.json();
    renderImpact(data);
  } catch (e) {
    $("impactWrap").innerHTML = `<div class="impact-msg">impact request failed: ${escHtml(e.message)}</div>`;
  }
}

function _distPretty(km) {
  if (km == null) return "—";
  const LD = 384400, au = 149597870.7;
  if (km < 1e5) return `${fmt(km, 0)} km`;
  if (km < 0.05 * au) return `${fmt(km / LD, 2)} lunar distances (${fmt(km, 0)} km)`;
  return `${fmt(km / au, 4)} au (${fmt(km / LD, 1)} LD)`;
}

function renderImpact(data) {
  const W = $("impactWrap");
  if (!data || !data.ok) {
    W.innerHTML = `<div class="impact-msg">impact assessment failed: ${escHtml((data && data.error) || "unknown")}</div>`;
    return;
  }
  const windowNote = `searched ${fmt(data.window_days, 0)} days forward from the orbit epoch (${data.epoch_iso} TDB) · ${data.force_model || ""}`;
  let loc;

  if (!data.impact) {
    // Honest non-impact: report the closest approach, never a fabricated hit.
    const caT = data.ca_time_iso ? data.ca_time_iso.slice(0, 19) + " UTC" : "—";
    const miss = data.ca_geocentric_km != null ? data.ca_geocentric_km : data.min_geocentric_km;
    const alt = data.ca_min_altitude_km;
    loc =
      `<div class="impact-card noimpact">
         <div class="impact-headline ok">No earth impact found within 30 days of orbit epoch</div>
         <div class="impact-sub">${data.reason === "perigee above ground (flyby)"
            ? "Its closest approach stays above the atmosphere — no surface impact."
            : "No close approach to Earth in the searched window."}</div>
         <table class="impact-kv">
           <tr><td>Closest approach</td><td>${caT}</td></tr>
           <tr><td>Miss distance</td><td>${_distPretty(miss)}</td></tr>
           ${alt != null ? `<tr><td>Min. altitude</td><td>${fmt(alt, 0)} km above ground</td></tr>` : ""}
         </table>
       </div>
       <div class="impact-foot">${windowNote}</div>`;
  } else {
    // Predicted impact: show the atmospheric-entry (100 km) and ground crossings.
    const rows = (data.crossings || []).map((c) => {
      const t = (c.time_iso || "").slice(0, 23) + " UTC";
      const lat = c.lat_deg, lon = c.lon_deg;
      const osm = `https://www.openstreetmap.org/?mlat=${lat.toFixed(4)}&mlon=${lon.toFixed(4)}#map=6/${lat.toFixed(2)}/${lon.toFixed(2)}`;
      const label = c.altitude_km >= 99 ? "Atmospheric entry (100 km)"
                  : c.altitude_km <= 1 ? "Ground point (0 km)"
                  : `${fmt(c.altitude_km, 0)} km`;
      return `<tr>
        <td class="l">${label}</td>
        <td>${t}</td>
        <td>${fmt(lat, 4)}°, ${fmt(lon, 4)}° <a href="${osm}" target="_blank" rel="noopener" title="open in OpenStreetMap">map ↗</a></td>
        <td>${fmt(c.geo_speed_km_s, 2)} km/s</td>
        <td>${fmt(c.entry_slope_deg, 1)}°</td></tr>`;
    }).join("");
    loc =
      `<div class="impact-card impact">
         <div class="impact-headline warn">⚠ Predicted Earth impact</div>
         <div class="impact-sub">Close approach ${(data.ca_time_iso || "").slice(0, 19)} UTC ·
           perigee ${fmt(data.ca_min_altitude_km, 0)} km (below ground → surface impact)</div>
         <table class="obs impact-cross"><thead><tr>
           <th class="l">altitude</th><th>time (UTC)</th>
           <th>geodetic lat / lon</th>
           <th title="geocentric speed at the crossing">entry speed</th>
           <th title="angle of the trajectory below the local horizontal">slope</th>
         </tr></thead><tbody>${rows}</tbody></table>
         ${_footprintBlock(data.footprint)}
       </div>
       <div class="impact-note">Nominal corridor validated to ~km on the known impactor 2008 TC3;
         the 1-σ footprint is the semilinear image of the orbit covariance (Milani-Valsecchi /
         Dimare-Del Vigna), not a Monte-Carlo cloud.</div>
       <div class="impact-foot">${windowNote}</div>`;
  }

  const locHd =
    `<div class="impact-sec-hd">Where — impact location
       <span class="impact-sec-sub">if it hits, where on the ground · near-term: ${fmt(data.window_days, 0)} days
       after the observations (orbit epoch ${data.epoch_iso})</span></div>`;
  W.innerHTML = locHd + loc;
}

// ---- 1-σ impact footprint (Phase 1b): semilinear uncertainty ellipse -------
function _footprintBlock(fp) {
  if (!fp || !fp.ok) return "";
  const sl = fp.semilinear, n = fp.nominal;
  // Full-axis lengths (2× the semi-axes) read more naturally as a "region size".
  const major = 2 * sl.semi_major_km, minor = 2 * sl.semi_minor_km;
  const ctr = `${fmt(sl.center_lat_deg, 4)}°, ${fmt(sl.center_lon_deg, 4)}°`;
  return `<div class="impact-footprint">
      <div class="fp-hd">1-σ uncertainty footprint at ${fmt(fp.altitude_km, 0)} km entry</div>
      <table class="impact-kv">
        <tr><td title="long × short full axes of the 1-σ ground ellipse">Region size (1-σ)</td>
            <td>${_kmPretty(major)} × ${_kmPretty(minor)}</td></tr>
        <tr><td title="bearing of the long axis, clockwise from north">Long-axis bearing</td>
            <td>${fmt(sl.azimuth_deg, 0)}°</td></tr>
        <tr><td>Ellipse centre</td><td>${ctr}</td></tr>
      </table>
    </div>`;
}

function _kmPretty(km) {
  return km < 1 ? `${fmt(km * 1000, 0)} m` : `${fmt(km, 2)} km`;
}

// ---- on-sky location pane (Aladin Lite) ----------------------------------
let aladin = null, aladinReady = false;

// Aladin Lite (CDS) is loaded LAZILY -- only when the sky pane is first used -- and never at
// page load. The CDS host (aladin.cds.unistra.fr) is occasionally down and slow-fails (a long
// TCP timeout, not a fast refusal); loading its CSS/JS in <head> would hang the whole page on
// that. Injecting on demand with a timeout keeps page load instant and degrades the sky view to
// an "unavailable" message when CDS can't be reached.
const ALADIN_CSS = "https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.css";
const ALADIN_JS  = "https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js";
const ALADIN_LOAD_TIMEOUT_MS = 8000;
let aladinInjected = false;

// Resolve once the global `A` (Aladin's API object) exists; reject if CDS hasn't responded
// within the timeout, so a slow-fail degrades instead of hanging forever. Injects the script +
// stylesheet at most once; later calls re-poll the SAME single <script>, so a CDS that recovers
// (or finishes a slow load) works on a retry with no duplicate tags. A hard error (DNS/refused)
// clears the guard so the next attempt can re-inject cleanly.
function loadAladin() {
  return new Promise((resolve, reject) => {
    if (typeof A !== "undefined") { resolve(); return; }
    if (!aladinInjected) {
      aladinInjected = true;
      const link = document.createElement("link");
      link.rel = "stylesheet"; link.href = ALADIN_CSS;
      document.head.appendChild(link);
      const script = document.createElement("script");
      script.src = ALADIN_JS; script.charset = "utf-8"; script.async = true;
      script.onerror = () => { aladinInjected = false; };   // allow a clean retry on hard failure
      document.head.appendChild(script);
    }
    const t0 = Date.now();
    (function poll() {
      if (typeof A !== "undefined") { resolve(); return; }
      if (Date.now() - t0 > ALADIN_LOAD_TIMEOUT_MS) {
        reject(new Error("Aladin Lite did not load (CDS unreachable or slow)"));
        return;
      }
      setTimeout(poll, 150);
    })();
  });
}

function ensureAladin(cb) {
  if (aladinReady && aladin) { cb(); return; }
  loadAladin()
    .then(() => A.init)
    .then(() => {
      if (!aladin) {
        aladin = A.aladin("#aladin-lite-div", {
          // Pin the survey to the CDS alasky mirror by full HiPS URL: it sends
          // Access-Control-Allow-Origin:* so the tiles load cross-origin. Passing the
          // bare id "P/DSS2/color" lets Aladin route to the IRSA mirror, which omits the
          // CORS header and gets blocked by the browser (any origin, localhost included).
          survey: "https://alasky.cds.unistra.fr/DSS/DSSColor", fov: 0.5, cooFrame: "ICRS",
          showReticle: false, showZoomControl: true, showFullscreenControl: false,
          showLayersControl: true, showGotoControl: false, showCooGrid: false,
        });
      }
      aladinReady = true;
      cb();
      // nudge Aladin to re-measure in case the WebGL canvas latched onto a stale size
      requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
    })
    .catch(() => {
      $("skyPlaceholder").textContent =
        "Sky view unavailable — Aladin Lite could not load (CDS unreachable or slow).";
      $("skyPlaceholder").style.display = "flex";
    });
}

// 1σ error ellipse as an RA/Dec polyline (arcsec params -> a closed loop on the sphere).
function ellipsePolyline(ra0, dec0, smajAsec, sminAsec, paDeg) {
  if (smajAsec == null || sminAsec == null || paDeg == null) return null;
  const a = smajAsec / 3600, b = sminAsec / 3600;          // -> degrees
  if (!(a > 0)) return null;
  const pa = paDeg * Math.PI / 180;
  const cosd = Math.cos(dec0 * Math.PI / 180) || 1e-6;
  const pts = [];
  for (let k = 0; k <= 48; k++) {
    const t = 2 * Math.PI * k / 48;
    // major axis points along PA (N through E); minor axis is +90° from it.
    const east = a * Math.cos(t) * Math.sin(pa) + b * Math.sin(t) * Math.cos(pa);
    const north = a * Math.cos(t) * Math.cos(pa) - b * Math.sin(t) * Math.sin(pa);
    pts.push([ra0 + east / cosd, dec0 + north]);
  }
  return pts;
}

// Default placeholder shown when no row is selected. "ephemeris row" is a link that
// jumps the central pane to the ephemeris table (see boot handler below). The prose is
// wrapped in a <span> so it's a single flex item — bare text + an inline <a> as direct
// children of the flex-centered .sky-ph would have their surrounding spaces trimmed.
const SKY_PH_DEFAULT =
  '<span>Click an <a href="#" id="skyPhEphemLink">ephemeris row</a> to plot its sky position.</span>';

// Wipe the on-sky pane back to its empty state (called when a new object is fitted so
// the prior object's marker / title / ephemeris rows don't linger).
function resetSkyView() {
  ephemRows = [];
  ephemSel = -1;
  $("skyTitle").textContent = "";
  $("skyHint").textContent = "";
  const ph = $("skyPlaceholder");
  ph.innerHTML = SKY_PH_DEFAULT;
  ph.style.display = "flex";
  if (aladin) { try { aladin.removeLayers(); } catch (e) { /* aladin not ready */ } }
}

function plotSky(row) {
  if (!row || row.ra_deg == null || row.dec_deg == null) return;
  $("skyTitle").textContent = CURRENT ? (CURRENT.object || "") : "";
  $("skyHint").textContent =
    `${(row.date_iso || "").slice(0, 16)} · 1σ ${fmt(row.unc_smaj_arcsec, 3)}″×${fmt(row.unc_smin_arcsec, 3)}″`;
  ensureAladin(() => {
    $("skyPlaceholder").style.display = "none";
    aladin.gotoRaDec(row.ra_deg, row.dec_deg);   // recenter; keep the user's current zoom
    aladin.removeLayers();                        // clear the previous row's marker + ellipse
    const overlay = A.graphicOverlay({ color: "#ff4d4f", lineWidth: 2 });
    aladin.addOverlay(overlay);
    overlay.add(A.circle(row.ra_deg, row.dec_deg, 0.002, { color: "#ffd166" }));  // ~7″ reticle
    const poly = ellipsePolyline(row.ra_deg, row.dec_deg,
                                 row.unc_smaj_arcsec, row.unc_smin_arcsec, row.unc_pa_deg);
    if (poly) overlay.add(A.polyline(poly, { color: "#ff4d4f" }));
  });
}

// ---- boot ----------------------------------------------------------------
// a USER switching panes is the analytics event; setCentralPane itself also runs on deep
// links and re-fits, which are not a choice anyone made
function userSetCentralPane(which) {
  setCentralPane(which);
  _ga(`switch_to_${centralPane}_mode`, { pane: centralPane });
}
$("paneSelect").addEventListener("change", (e) => userSetCentralPane(e.target.value));
// "ephemeris row" in the on-sky placeholder jumps the central pane to the ephemeris
// table (delegated, so it survives placeholder re-renders).
$("skyPlaceholder").addEventListener("click", (e) => {
  if (e.target.closest("#skyPhEphemLink")) { e.preventDefault(); userSetCentralPane("ephem"); }
});
$("ephemParamsBtn").addEventListener("click", () => {
  ephemDialogMode = "pane";
  $("ephStep").value = ephemParams.step;
  $("ephN").value = ephemParams.n;
  $("ephSite").value = ephemParams.site;
  $("ephStart").value = _jdToISODate(ephemParams.startJD == null ? _todayJD() : ephemParams.startJD);
  openEphemDialog();
});

// ---- ephemeris display cuts (faint limit / suppress unobservables) -------
// The faint box carries NO default: empty means "no limit", which is what the pane
// starts on. (find_orb's own 22.0 lives in ephemeris.FAINT_LIMIT_DEFAULT and is what a
// bare CLI --ephem-faint takes; the UI deliberately doesn't quote or preset it, so
// there is no engine number here to go stale.)

// "Suppress unobservables" only means something for a site with a horizon; for the
// geocentre and space-based codes find_orb doesn't even offer it, so grey it out.
async function syncEphemUnobs() {
  const site = ($("ephSite").value || "500").trim() || "500";
  const row = $("ephUnobs").closest(".eph-chk");
  let ok = site !== "500";
  if (ok) {
    try {
      const d = await (await fetch(`/api/ephem_site?site=${encodeURIComponent(site)}`)).json();
      ok = !!(d && d.site_has_horizon);
    } catch (e) { /* offline: trust the code and leave it enabled */ }
  }
  $("ephUnobs").disabled = !ok;
  row.classList.toggle("disabled", !ok);
  $("ephUnobsHint").textContent = ok
    ? "below the horizon, in daylight, or behind the Moon"
    : `observatory ${site} has no horizon (geocentric or space-based)`;
}
$("ephSite").addEventListener("change", syncEphemUnobs);

// Fill the shared dialog's cut controls from the pane's current settings and open it.
function openEphemDialog() {
  $("ephFaint").value = ephemParams.faint == null ? "" : ephemParams.faint;
  $("ephUnobs").checked = !!ephemParams.unobs;
  syncEphemUnobs();
  _setHidden($("ephemOverlay"), true);
}
$("fitBtn").addEventListener("click", runFit);
$("desig").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.target.blur(); runFit(); } });
// Cap the search box at 50 chars: on any over-length entry (typing or paste), clear it
// and warn. A designation / short linkage list is always well under this.
$("desig").addEventListener("input", (e) => {
  if (e.target.value.length > 50) {
    e.target.value = "";
    flashBanner("(max 50 characters allowed)", "alert");
  }
});
$("worstFirst").addEventListener("change", () => { if (CURRENT) renderObs(CURRENT); });

// click a history stage -> replay that stage's residuals (delegation; #histBody persists)
$("histBody").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-stage]");
  if (!li || !CURRENT) return;
  const idx = Number(li.dataset.stage);
  if (!(CURRENT.stage_residuals && CURRENT.stage_residuals[idx])) return;  // no snapshot
  viewStage = (viewStage === idx) ? null : idx;     // click again to toggle back
  renderObs(CURRENT); renderHistory(CURRENT);
});
// "Return to final fit" link in the replay banner
$("obsBanner").addEventListener("click", (e) => {
  if (e.target.closest("a[data-final]")) {
    e.preventDefault(); viewStage = null;
    renderObs(CURRENT); renderHistory(CURRENT);
  }
});
// click an obs row -> toggle include/exclude (final view only; not during replay)
$("obsScroll").addEventListener("click", (e) => {
  if (viewStage !== null || !CURRENT || !workingMask) return;
  const tr = e.target.closest("tr[data-i]");
  if (!tr) return;
  const i = Number(tr.dataset.i);
  if (Number.isNaN(i)) return;
  workingMask[i] = !workingMask[i];
  renderObs(CURRENT);
});
// Virtualized table: re-render the visible window as the user scrolls (rAF-throttled to one
// rebuild per frame). No-op when the table is rendered in full (_virt === null).
$("obsScroll").addEventListener("scroll", () => {
  if (!_virt || _virtRaf) return;
  _virtRaf = requestAnimationFrame(() => { _virtRaf = 0; renderObsWindow(); });
});
$("fullStepBtn").addEventListener("click", () => runStep(true));
$("multiStepBtn").addEventListener("click", () => runStep(false));
$("reAutoBtn").addEventListener("click", runReAuto);
// <input type=date> fires "change" on every keystroke that happens to leave the field
// syntactically complete (e.g. Chrome pads a lone first digit into year 0001) -- not just
// on commit -- so typing "1999" tries to fit epoch year 1 mid-keystroke and errors out
// (planetary ephemerides don't reach back that far). Debounce so a burst of keystrokes
// collapses to one fetch after typing pauses; blur (click-out) and Enter fire right away.
let _epochDebounce = null;
function _epochCommit() {
  if (_epochDebounce) { clearTimeout(_epochDebounce); _epochDebounce = null; }
  setEpoch();
}
$("elemEpoch").addEventListener("change", () => {
  if (_epochDebounce) clearTimeout(_epochDebounce);
  _epochDebounce = setTimeout(_epochCommit, 700);
});
$("elemEpoch").addEventListener("blur", () => { if (_epochDebounce) _epochCommit(); });
$("elemEpoch").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { _epochCommit(); $("elemEpoch").blur(); }
});
$("worstBtn").addEventListener("click", jumpToWorst);

// MPC 80-col astrometry: toggle the view, and copy the whole block to the clipboard.
$("mpc80Btn").addEventListener("click", () => {
  obsView = obsView === "mpc80" ? "table" : "mpc80";
  if (CURRENT) renderObs(CURRENT);
});
// "＋ addl info": reveal/hide the extra scroll-right ADES columns in the table view.
$("addlBtn").addEventListener("click", () => {
  showAddl = !showAddl;
  if (CURRENT) renderObs(CURRENT);
});
$("copyObsBtn").addEventListener("click", async () => {
  const pre = $("obsScroll").querySelector("pre.mpc80");
  if (!pre) return;
  const text = pre.textContent;
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {                                   // clipboard API blocked (non-https) -> legacy path
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select();
    document.execCommand("copy"); ta.remove();
  }
  const b = $("copyObsBtn"), was = b.textContent;
  b.textContent = "✓ copied"; setTimeout(() => { b.textContent = was; }, 1200);
});
$("ngSelect").addEventListener("change", (e) => {
  _ga("change_force_model", { control: "nongrav", model: e.target.value });
  runNongrav(e.target.value);
});
$("conToggle").addEventListener("change", () => {
  // Sync handler: it only shows/hides the panel. The calls that actually need a
  // session (postConstraints) claim one through their own guard.
  if (!CURRENT || !(CURRENT.session || CURRENT.restore || CURRENT.refit)) return;
  if ($("conToggle").checked) {
    renderConstraints(CURRENT, false);          // reveal inputs; honour the just-checked box
  } else {
    $("conBody").hidden = true; $("conBody").innerHTML = "";
    if (CURRENT.constraints && Object.keys(CURRENT.constraints).length) postConstraints({});
  }
});

// right-click an obs row -> bulk-toggle menu (above/below). Final view only.
$("obsScroll").addEventListener("contextmenu", (e) => {
  if (viewStage !== null || !CURRENT || !workingMask) return;
  const tr = e.target.closest("tr[data-i]");
  if (!tr) return;
  e.preventDefault();
  const menu = $("rowMenu");
  menu.dataset.pivot = tr.dataset.i;
  menu.hidden = false;
  menu.style.left = Math.min(e.clientX, window.innerWidth - 190) + "px";
  menu.style.top = Math.min(e.clientY, window.innerHeight - 80) + "px";
});
$("rowMenu").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-dir]");
  if (b) { e.stopPropagation(); bulkToggle(b.dataset.dir, Number($("rowMenu").dataset.pivot)); }
  hideRowMenu();
});
document.addEventListener("click", hideRowMenu);
document.addEventListener("keydown", (e) => {
  const typing = /^(INPUT|TEXTAREA)$/.test((document.activeElement || {}).tagName || "");
  if (e.key === "Escape") hideRowMenu();
  if (typing) return;
  // Letter shortcuts fire ONLY on the bare key: a Ctrl/Cmd/Alt chord belongs to the
  // browser (Ctrl-F is find-in-page, Cmd-W closes the tab, Alt-F opens a menu) and
  // used to be swallowed as a Full step / jump-to-worst.
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const key = e.key.toLowerCase();
  if (key === "w" && CURRENT) jumpToWorst();
  // f = ONE Full step (find_orb `f`), m = Multi-step (solve to convergence) --
  // but not while replaying a history stage
  if (key === "f" && CURRENT && CURRENT.session && viewStage === null) {
    e.preventDefault(); runStep(true);
  }
  if (key === "m" && CURRENT && CURRENT.session && viewStage === null) {
    e.preventDefault(); runStep(false);
  }
});
// Re-render the plot whenever its pane actually changes size — this also fires
// once the pane is first laid out, so the SVG is sized against a real box height
// instead of a 0-height (pre-layout) read. Mirrors the orbit canvas's observer.
new ResizeObserver(() => { if (CURRENT) renderOCPlot(CURRENT); }).observe($("ocBody"));

// O−C unit switch: arcsec ↔ σ (re-renders only the plot; the table is unchanged)
$("ocUnitToggle").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-unit]");
  if (!b || b.dataset.unit === ocUnit) return;
  ocUnit = b.dataset.unit;
  $("ocUnitToggle").querySelectorAll(".seg")
    .forEach((s) => s.classList.toggle("active", s.dataset.unit === ocUnit));
  if (CURRENT) renderOCPlot(CURRENT);
});

// O−C axis switch: the colour key itself toggles RA·cosδ / Dec. A click that would
// hide BOTH is read as "show only the other one" rather than ignored -- clicking the
// axis you want gone, twice, is the natural way to ask for a single-axis plot.
$("ocAxisToggle").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-axis]");
  if (!b) return;
  const k = b.dataset.axis, other = k === "ra" ? "dec" : "ra";
  if (ocAxes[k] && !ocAxes[other]) { ocAxes[k] = false; ocAxes[other] = true; }
  else ocAxes[k] = !ocAxes[k];
  $("ocAxisToggle").querySelectorAll(".axkey")
    .forEach((s) => s.classList.toggle("off", !ocAxes[s.dataset.axis]));
  if (CURRENT) renderOCPlot(CURRENT);
});

// The toggle segment a payload's force model is: "0" gravity, "1" the 16 (+ any screen
// picks), "all" the whole extra-perturber kernel.
function astState(p) {
  return !p.asteroids ? "0" : p.perturbers_all ? "all" : "1";
}

// gravity / +16-asteroid / all-asteroid segmented toggle (re-fits in the chosen field)
$("astToggle").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-ast]");
  if (!b) return;
  const want = b.dataset.ast;                               // "0" | "1" | "all"
  if (CURRENT && want === astState(CURRENT)) return;        // already in that state
  _ga("change_force_model", { control: "asteroids",
    model: want === "all" ? "allast" : want === "1" ? "16ast" : "gravity" });
  setAsteroids(want === "all" ? "all" : want === "1");
});

// ---- paste loader + export (WEB_UI_PLAN §2.7) ----------------------------
const _setHidden = (el, on) => { el.hidden = !on; };

// Export handle: prefer the live interactive session so the export is exactly the
// orbit on screen (current include/exclude mask, nongrav model, epoch, any Full/Multi
// steps) rather than a fresh from-scratch autofit. Falls back to a pasted fit's cached
// key, then to the designation list (which re-fits cache-first). Same precedence as
// _ephemHandle so every export product reflects the same displayed orbit.
function exportUrl(kind, extra = "") {
  if (!CURRENT) return null;
  return `/api/export/${kind}?${_ephemHandle()}${extra}`;
}

$("exportBtn").addEventListener("click", (e) => {
  e.stopPropagation();
  if ($("exportBtn").disabled) return;
  _setHidden($("exportMenu"), $("exportMenu").hidden);
});
document.addEventListener("click", () => _setHidden($("exportMenu"), false));
$("exportMenu").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-export]");
  if (!b) return;
  _setHidden($("exportMenu"), false);
  const kind = b.dataset.export;
  _ga("export", { export_type: kind });
  if (kind === "ephem") {                     // ephemeris -> collect params first
    ephemDialogMode = "export";
    $("ephStart").value = new Date().toISOString().slice(0, 10);   // default: today (YYYY-MM-DD)
    openEphemDialog();
  } else if (kind === "mpec") {               // pseudo-MPEC is a rendered HTML doc -> new tab
    const u = exportUrl(kind);
    if (u) window.open(u, "_blank");
  } else {                                    // elements / mpcorb / sof -> copy/paste box
    const u = exportUrl(kind);
    if (u) openExportText(EXPORT_TITLES[kind] || "Export", u);
  }
});

// Copy/paste export box: fetch the text product for the CURRENT displayed orbit and
// show it inline in a selectable, copyable overlay (no new tab/window). The GET carries
// the session handle (see exportUrl), so the server returns exactly the on-screen fit.
const EXPORT_TITLES = { elements: "Orbital elements (8-line)", mpcorb: "MPCORB line",
                        sof: "SOF record", ephem: "Ephemeris",
                        state: "State vector + covariance (JSON)",
                        covar: "find_orb covar.json" };
function openExportText(title, url) {
  $("exportResultTitle").textContent = title;
  const ta = $("exportResultText");
  ta.value = "Loading…";
  $("exportCopyBtn").textContent = "Copy";
  $("exportCopyBtn").disabled = true;
  _setHidden($("exportResultOverlay"), true);
  fetch(url).then((r) => r.text()).then((t) => {
    ta.value = t;
    ta.scrollTop = 0;
    $("exportCopyBtn").disabled = false;
    ta.focus(); ta.select();
  }).catch((ex) => { ta.value = "export failed: " + ex; });
}
$("exportCopyBtn").addEventListener("click", async () => {
  const ta = $("exportResultText");
  try {
    await navigator.clipboard.writeText(ta.value);
  } catch (ex) {                              // clipboard API unavailable (insecure ctx) -> fall back
    ta.focus(); ta.select(); document.execCommand && document.execCommand("copy");
  }
  $("exportCopyBtn").textContent = "Copied ✓";
  setTimeout(() => { $("exportCopyBtn").textContent = "Copy"; }, 1200);
});
$("exportResultClose").addEventListener("click", () => _setHidden($("exportResultOverlay"), false));
$("exportResultOverlay").addEventListener("click", (e) => {
  if (e.target === $("exportResultOverlay")) _setHidden($("exportResultOverlay"), false);
});

$("ephCancel").addEventListener("click", () => _setHidden($("ephemOverlay"), false));
$("ephemOverlay").addEventListener("click", (e) => {
  if (e.target === $("ephemOverlay")) _setHidden($("ephemOverlay"), false);
});
$("ephStart").addEventListener("input", () => $("ephStart").setCustomValidity(""));
$("ephGo").addEventListener("click", () => {
  const step = parseFloat($("ephStep").value) || 1;
  const n = parseInt($("ephN").value, 10) || 30;
  const site = ($("ephSite").value || "500").trim() || "500";
  // A text field, not <input type=date>: the native picker renders in the
  // browser's locale (mm/dd/yyyy in the US), and the date must read YYYY-MM-DD.
  const dateStr = ($("ephStart").value || "").trim();
  const dm = dateStr.match(/^(\d{4})[-\/. ]+(\d{1,2})[-\/. ]+(\d{1,2})$/);
  const dms = dm && Date.UTC(+dm[1], +dm[2] - 1, +dm[3]);
  const valid = dm && new Date(dms).getUTCDate() === +dm[3] && +dm[2] >= 1 && +dm[2] <= 12;
  if (dateStr && !valid) {
    $("ephStart").setCustomValidity("Use YYYY-MM-DD, e.g. 2026-09-30");
    $("ephStart").reportValidity();
    return;
  }
  $("ephStart").setCustomValidity("");
  const startJD = dateStr ? (dms / 86400000 + 2440587.5) : null;
  const faintRaw = ($("ephFaint").value || "").trim();       // blank -> no faint limit
  const faint = faintRaw === "" || isNaN(parseFloat(faintRaw)) ? null : parseFloat(faintRaw);
  const unobs = $("ephUnobs").checked && !$("ephUnobs").disabled;
  _setHidden($("ephemOverlay"), false);
  if (ephemDialogMode === "pane") {          // refresh the in-pane View Ephemeris table
    ephemParams.step = step; ephemParams.n = n; ephemParams.site = site; ephemParams.startJD = startJD;
    ephemParams.faint = faint; ephemParams.unobs = unobs;
    loadEphem();
    return;
  }
  let extra = `&step=${step}&n=${n}&site=${encodeURIComponent(site)}`;   // export to a copy/paste box
  extra += _ephemFilterQS({ faint, unobs });
  if (startJD != null) extra += `&start=${startJD}`;
  const u = exportUrl("ephem", extra);
  if (u) openExportText("Ephemeris", u);
});

// About / credits modal
function openAbout() { _setHidden($("aboutOverlay"), true); }
$("aboutLink").addEventListener("click", openAbout);
$("aboutLink").addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openAbout(); }
});
$("aboutClose").addEventListener("click", () => _setHidden($("aboutOverlay"), false));
$("aboutOverlay").addEventListener("click", (e) => {
  if (e.target === $("aboutOverlay")) _setHidden($("aboutOverlay"), false);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    ["aboutOverlay", "pasteOverlay", "ephemOverlay", "settingsOverlay", "exportResultOverlay",
     "itfOverlay", "itfTraceOverlay"]
      .forEach((id) => _setHidden($(id), false));
  }
});

// Settings: read-only inventory of what this host resolved (data files /
// ephemerides / env overrides). A missing optional file silently changes the
// fit, so this is the panel to compare across hosts. Inspection only -- no PUTs.
function _settingsRow(it) {
  const li = document.createElement("li");
  li.className = "set-row";
  // Tri-state mark: ✓ present, ✗ absent (red, louder when critical), · neutral (env vars).
  const mark = document.createElement("span");
  if (it.present === true) { mark.className = "set-mark ok"; mark.textContent = "✓"; }
  else if (it.present === false) {
    mark.className = "set-mark bad" + (it.critical ? " crit" : "");
    mark.textContent = "✗";
  } else { mark.className = "set-mark neutral"; mark.textContent = "·"; }
  const main = document.createElement("div");
  main.className = "set-main";
  const name = document.createElement("div");
  name.className = "set-name";
  name.textContent = it.name;
  const det = document.createElement("span");
  det.className = "set-detail";
  det.textContent = it.detail || "";
  name.appendChild(det);
  main.appendChild(name);
  if (it.path) {
    const path = document.createElement("div");
    path.className = "set-path mono";
    path.textContent = it.path;
    main.appendChild(path);
  }
  if (it.note) {
    const note = document.createElement("div");
    note.className = "set-note";
    note.textContent = it.note;
    main.appendChild(note);
  }
  li.appendChild(mark);
  li.appendChild(main);
  return li;
}

function renderSettings(data) {
  const body = $("settingsBody");
  body.innerHTML = "";
  if (!data || data.ok === false) {
    body.innerHTML = `<div class="settings-loading">Could not read host status` +
      (data && data.error ? `: ${data.error}` : "") + `.</div>`;
    return;
  }
  const p = data.platform || {};
  const plat = document.createElement("div");
  plat.className = "set-platform mono";
  plat.textContent = [p.system, p.release, p.machine, "· Python " + p.python]
    .filter(Boolean).join(" ");
  body.appendChild(plat);
  (data.groups || []).forEach((g) => {
    const h = document.createElement("div");
    h.className = "set-group-hd";
    h.textContent = g.title;
    body.appendChild(h);
    const ul = document.createElement("ul");
    ul.className = "set-list";
    (g.items || []).forEach((it) => ul.appendChild(_settingsRow(it)));
    body.appendChild(ul);
  });
}

async function openSettings() {
  _setHidden($("settingsOverlay"), true);
  $("settingsBody").innerHTML = `<div class="settings-loading">Loading host status…</div>`;
  try {
    const r = await fetch("/api/environment");
    renderSettings(await r.json());
  } catch (ex) {
    renderSettings({ ok: false, error: String(ex) });
  }
}
$("settingsBtn").addEventListener("click", openSettings);
$("settingsClose").addEventListener("click", () => _setHidden($("settingsOverlay"), false));
$("settingsOverlay").addEventListener("click", (e) => {
  if (e.target === $("settingsOverlay")) _setHidden($("settingsOverlay"), false);
});

// Paste-an-obs loader. "Add to current fit" is offered whenever a fit is open and
// reachable -- a live interactive session, a pasted fit's export key (the two
// handles /api/add_obs resolves the current arc from), OR a handle ensureLive()
// can turn into a session on click: a coalesced fit's `restore` token, or a
// disk-cache hit's `refit` designation. Those last two are the same gate every
// other interactive control uses; leaving them out hid the button on any fit
// served from the queue's disk tier or whose session was never claimed.
function _canAddToFit() {
  return !!(CURRENT && (CURRENT.session || CURRENT.fit_key ||
                        CURRENT.restore || CURRENT.refit));
}
$("pasteBtn").addEventListener("click", () => {
  _setHidden($("pasteAdd"), _canAddToFit());
  _setHidden($("pasteOverlay"), true); $("pasteText").focus();
});
$("pasteCancel").addEventListener("click", () => _setHidden($("pasteOverlay"), false));
$("itfBtn").addEventListener("click", () => {
  if ($("itfBtn").disabled) return;
  _itfLoadDefaults();
  _itfShowServerState();          // live: capacity and queue change by the minute
  _setHidden($("itfOverlay"), true);
});

// Tell the user what the SERVER can do before they commit to minutes of
// waiting. Capacity and the queue both change while a page is open, so this is
// re-read on every modal open rather than taken from the once-per-load
// defaults fetch -- a stale "plenty of room" is worse than no notice at all.
// The notice element is created on demand so no index.html change is needed.
async function _itfShowServerState() {
  const host = $("itfGo") && $("itfGo").parentNode;
  if (!host) return;
  let el = $("itfServerState");
  if (!el) {
    el = document.createElement("div");
    el.id = "itfServerState";
    el.style.cssText = "margin:.5em 0;padding:.5em .7em;border-radius:4px;" +
                       "font-size:.9em;line-height:1.35";
    host.parentNode.insertBefore(el, host);
  }
  el.hidden = true;
  try {
    const p = await (await fetch("/api/itf_defaults")).json();
    if (!p || !p.ok) return;
    const bits = [];
    const cap = p.capacity;
    if (cap && cap.verdict === "unusable") {
      bits.push(`<b>The server is out of memory for ITF searches right now.</b> ` +
        `A search started now would stop before finding anything, so it will ` +
        `be refused \u2014 please try again in a minute.`);
    } else if (cap && cap.verdict === "shallow") {
      bits.push(`<b>The server is low on memory.</b> Your search will run, but ` +
        `less deeply than usual \u2014 a weakly constrained orbit may be cut ` +
        `short. Re-running when it is quieter may search further.`);
    }
    const sl = p.slots;
    if (sl && sl.running >= sl.limit) {
      bits.push(`${sl.running} of ${sl.limit} search slots are busy; a new ` +
        `search will be refused until one frees up.`);
    } else if (sl && sl.running > 0) {
      bits.push(`${sl.running} of ${sl.limit} search slots in use.`);
    }
    if (!bits.length) return;
    const bad = cap && cap.verdict === "unusable";
    el.style.background = bad ? "#5a1e1e" : "#4a3c14";
    el.style.color = "#f2e9d8";
    el.innerHTML = bits.join("<br>");
    el.hidden = false;
  } catch (e) { /* old server / offline: no notice, same as before */ }
}
$("itfCancel").addEventListener("click", () => _setHidden($("itfOverlay"), false));
$("itfOverlay").addEventListener("click", (e) => {
  if (e.target === $("itfOverlay")) _setHidden($("itfOverlay"), false);
});
$("itfGo").addEventListener("click", runItfSearch);
$("itfSubmitBtn").addEventListener("click", () => runItfSubmit("queue"));
$("itfSubmitMore").addEventListener("click", (e) => {
  e.stopPropagation();
  _setHidden($("itfSubmitMenu"), $("itfSubmitMenu").hidden);
});
document.addEventListener("click", () => _setHidden($("itfSubmitMenu"), false));
$("itfDownloadBtn").addEventListener("click", () => {
  _setHidden($("itfSubmitMenu"), false); runItfDownload();
});
$("itfDirectBtn").addEventListener("click", () => {
  _setHidden($("itfSubmitMenu"), false); runItfSubmit("mpc");
});
$("itfIdentityBtn").addEventListener("click", () => {
  _setHidden($("itfSubmitMenu"), false);
  const me = askIdentity(true);
  if (me) flashBanner(`Linkages will be submitted as ${me.name} <${me.email}>.`, "info");
});
$("itfTraceClose").addEventListener("click", () => _setHidden($("itfTraceOverlay"), false));
$("itfTraceOverlay").addEventListener("click", (e) => {
  if (e.target === $("itfTraceOverlay")) _setHidden($("itfTraceOverlay"), false);
});
$("itfTraceBtn").addEventListener("click", () => {
  if (_lastTraceId) openItfTrace(_lastTraceId);
});
// After a page reload, restore access to the last search's trace if the
// server still holds that feed (it keeps the most recent handful in memory).
(async () => {
  let id = null;
  try { id = sessionStorage.getItem("itfTraceId"); } catch (e) {}
  if (!id) return;
  try {
    const r = await fetch(`/api/itf_trace?id=${encodeURIComponent(id)}&from=0`);
    const p = await r.json();
    if (p.ok) { _lastTraceId = id; _setHidden($("itfTraceBtn"), true); }
  } catch (e) { /* server restarted or feed evicted -> button stays hidden */ }
})();
$("pasteOverlay").addEventListener("click", (e) => {
  if (e.target === $("pasteOverlay")) _setHidden($("pasteOverlay"), false);
});
$("pasteFit").addEventListener("click", () => runPaste("/api/fit_text", $("pasteFit"), "fitting pasted observations…"));
$("pasteAdd").addEventListener("click", async () => {
  // No session and no pasted-fit key: claim one first (restore token, or a
  // fresh fit for a disk-cache hit), exactly as the step/model controls do.
  if (!CURRENT.session && !CURRENT.fit_key && !(await ensureLive())) return;
  const handle = CURRENT.session ? `session=${encodeURIComponent(CURRENT.session)}`
    : `fit_key=${encodeURIComponent(CURRENT.fit_key)}`;
  runPaste(`/api/add_obs?${handle}`, $("pasteAdd"), "adding observations and re-fitting…");
});

// Shared paste-submit: POST the textarea to `url`, then render the returned fit the
// same way a fetched object renders (both /api/fit_text and /api/add_obs return the
// run_fit payload shape). `btn` is the clicked button, disabled with a spinner.
async function runPaste(url, btn, busy) {
  const text = $("pasteText").value;
  if (!text.trim()) return;
  const force = false;   // experimental engine option, not exposed in the UI (see runFit)
  _ga("paste_observations", {
    action: url.startsWith("/api/add_obs") ? "add"
          : btn === $("pasteFit") ? "fit" : "neocp_fit",   // neocpfit.js pastes for the user
    n_lines: text.split("\n").filter((l) => l.trim()).length,
  });
  const sep = url.indexOf("?") === -1 ? "?" : "&";
  const label = btn.textContent;
  btn.disabled = true; btn.textContent = "working…";
  $("msg").innerHTML = `<span class="loading">${busy}</span>`;
  try {
    const r = await fetch(`${url}${sep}force=${force ? 1 : 0}`, {
      method: "POST", headers: { "Content-Type": "text/plain" }, body: text,
    });
    let p = await r.json();
    // A paste runs through the same admission-controlled worker pool as every
    // other fit (src/pyroorb/web/fitqueue.py:run_bounded); past its inline deadline it
    // answers with a job stub (ok:true, pending:true) instead of a result, and
    // this poll finishes it out exactly as FitJob.run does for /api/fit_start.
    // Without this the stub itself used to be rendered as the fitted object.
    if (FitJob.isPending(p)) {
      p = await FitJob.pollUntilDone(p.job, (t) => {
        $("msg").innerHTML = `<span class="loading">${escHtml(busy)} — ${escHtml(t)}</span>`;
      }, p);
    }
    if (!p.ok) {
      // Pasted Pluto astrometry is banned (DE440 perturber) -- pop the banner and explain.
      if (p.pluto_banned) {
        flashBanner(p.banner || p.error, "alert");
        $("msg").innerHTML = `<b>Pluto is banned.</b> ${escHtml(p.banner || p.error || "")}`;
        return;
      }
      $("msg").innerHTML = `<b>Could not process pasted observations.</b> ${escHtml(p.error || "")}` +
        (p.hint ? `<br><span style="color:var(--muted)">${escHtml(p.hint)}</span>` : "");
      return;
    }
    _setHidden($("pasteOverlay"), false);
    $("pasteText").value = "";
    $("msg").innerHTML = "";
    $("desig").value = p.object || "(pasted)";
    // "Add to current fit" merged new obs into the open fit -> offer Re-auto-solve from
    // now on; a fresh whole-object paste (/api/fit_text) starts over with nothing added.
    obsAdded = url.startsWith("/api/add_obs");
    CURRENT = p; render(p); $("workspace").hidden = false;
    clearObjectUrl();                  // pasted obs aren't reproducible from a link
  } catch (e) {
    $("msg").textContent = "paste failed: " + e.message;
  } finally {
    btn.disabled = false; btn.textContent = label;
  }
}
// ---- collapsible visuals (right) column -----------------------------------
// The right column (3D orbit + on-sky + O−C plot) folds away to hand the middle pane
// the full width. It AUTO-shows at >=1500px and AUTO-hides below; a manual toggle (the
// topbar "Visuals" button, plus a floating edge tab when it's hidden) flips it in
// between, and the next time the viewport crosses 1500px the width rule re-asserts.
// Repaint is automatic: the orbit canvas and #ocBody each have a ResizeObserver that
// fires when the column regains a real width.
const RIGHTCOL_BREAKPOINT = 1500;
let rightColShown = true;
function applyRightCol(show) {
  rightColShown = show;
  $("workspace").classList.toggle("right-collapsed", !show);
  const btn = $("rightColToggle");
  btn.setAttribute("aria-pressed", String(show));
  btn.innerHTML = show ? "Visuals ▸" : "◂ Visuals";
  btn.title = (show ? "Hide" : "Show") +
    " the visuals column (3D orbit, sky view, O−C plot). Auto-shows at ≥1500px wide, auto-hides below.";
  $("rightColTab").hidden = show;
  if (show) {                          // nudge a relayout so the canvas/SVG repaint at full size
    requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
  }
}
$("rightColToggle").addEventListener("click", () => applyRightCol(!rightColShown));
$("rightColTab").addEventListener("click", () => applyRightCol(true));
// initial state from the viewport width, then auto show/hide each time it crosses 1500px
const _rightColMQ = window.matchMedia(`(min-width: ${RIGHTCOL_BREAKPOINT}px)`);
applyRightCol(_rightColMQ.matches);
_rightColMQ.addEventListener("change", (e) => applyRightCol(e.matches));

loadNongravModels();
loadConstraintElements();

// Back/forward between shared object links, and auto-fit the object named in the URL on
// first load (so a pasted ?object=… link opens straight onto that fit).
window.addEventListener("popstate", fitFromUrl);
fitFromUrl();
