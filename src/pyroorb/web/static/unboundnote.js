"use strict";
// PyRoOrb web UI — unbound-orbit notes. Vanilla JS, no build step. Loaded AFTER app.js.
//
// payload.unbound (engine_api.serialise_fit) carries two informational records, never a warning:
//  - check: what fit_object._prefer_bound_ecc weighed when the free fit came out at e > 1 (the
//    rules in hyperbolic.py). An orbit kept unbound -- 1I, 2I, 3I -- gets NO note; silence is the
//    point. A bound orbit shown although an unbound one fits gets one sentence saying so and why:
//    either bounding it barely moves the residuals, or the arc has too few nights to believe an
//    unbound orbit (a wrong linkage fits hyperbolic too).
//  - reminder: a CLEARLY unbound orbit whose excess energy a planet could have supplied -- unbound
//    now is not unbound on the way in (hyperbolic.inbound_reminder). Its text comes from the engine
//    so the command line can say the same thing.
//
// Listens on the "pyroorb:render" event app.js's render() dispatches (like dedupnote.js).

(function () {
  function ensureBox() {
    let box = document.getElementById("unboundNote");
    if (!box) {
      const anchor = document.getElementById("constraintBanner");
      if (!anchor) return null;
      box = document.createElement("div");
      box.id = "unboundNote";
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

  function num(x, d) {
    return (typeof x === "number" && isFinite(x)) ? x.toFixed(d) : "?";
  }

  // A bar reads as a bar: "3σ", not "3.0σ" -- while an odd one (few tracklets make the
  // evidence, never the bar, move) still keeps its decimal.
  function bar(x) {
    return num(x, 1).replace(/\.0$/, "");
  }

  // Both bound cases end the same way: the bound orbit is a decision made once, at the end of the
  // automatic fit, and Multi-step's free re-solve (engine_api._apply_interactive) does not repeat
  // it -- so say what that button will show and how to judge it, rather than let the orbit flip
  // unexplained on the first click.
  const MULTISTEP_TAIL =
    " Multi-step ↻ re-solves with no limit on e and shows the lowest-rms orbit, which may be the " +
    "unbound one. Lowest rms is not the same as right: a pure minimisation can fit noise and make " +
    "an ordinary object look interstellar. Judge it by the O−C residuals: if they show a " +
    "systematic trend under this bound orbit that disappears after Multi-step, the unbound " +
    "solution may genuinely be the better one.";

  function boundNote(c) {
    if (!c || c.kept !== "bound" || typeof c.e_free !== "number") return null;
    const fits = `e = ${num(c.e_free, 5)} at ${num(c.rms_free, 3)}″, against ${num(c.rms_bound, 3)}″ bound`;
    if (c.damaged) {
      const n = c.nights;
      return {
        head: "ℹ Bound orbit shown — an unbound one fits better, on too short an arc to believe",
        detail: `An unbound solution fits better (${fits}), but ${n} night${n === 1 ? "" : "s"} ` +
                `over ${num(c.arc_days, 1)} days is too few to believe an unbound orbit: a wrong ` +
                `linkage fits hyperbolic too. More nights will settle it.` + MULTISTEP_TAIL,
      };
    }
    // The σ and its bar both come from the engine (`check.sigma`, `check.sigma_bar`): the
    // rule is a fixed 3σ and the engine's statistic is a chi2(1) under the
    // station-night error model (nightmodel.py), so sqrt of it IS that σ on any
    // arc. They were not always comparable: the engine once reported a per-observation Δχ²
    // here while deciding on an F-quantile of it, and on a two-tracklet arc this sentence
    // read "only 9.0σ, short of the 3σ it takes". Never re-derive either number here, and
    // never write the bar as a literal — a UI literal shadowing an engine constant is how
    // that reappears.
    // What the σ was measured on, stated as a fact and not as an inference: each
    // station-night's shared offset and rate error are priced once by the model, so the
    // count says what the evidence was weighed on -- but how much a night is worth depends
    // on its station, so a sentence like "cannot be worth much more than that" would be
    // true of two station-nights and false of two hundred.
    const looks = (typeof c.station_nights === "number" && c.station_nights > 0)
      ? `, on ${c.station_nights} independent station-night${c.station_nights === 1 ? "" : "s"} of ` +
        `astrometry`
      : "";
    return {
      head: "ℹ Bound orbit shown — an unbound one fits about as well",
      detail: `An unbound solution fits about as well (${fits}). We held this orbit below e = 1 ` +
              (typeof c.sigma === "number"
                ? `because the data disfavour it at only ${num(c.sigma, 1)}σ${looks}, short ` +
                  `of the ${bar(c.sigma_bar)}σ it takes to refuse a bound orbit. ` +
                  `Near-parabolic comets far outnumber `
                : `because doing so barely changes the residuals: near-parabolic comets far outnumber `) +
              `interstellar objects, and on a short arc a small rms gain can buy almost any ` +
              `eccentricity.` + MULTISTEP_TAIL,
    };
  }

  // An unbound orbit the engine could not EARN. Rule 2 believes an unbound orbit on a thin
  // arc only when the data refuse the bound one -- and "the bound one" is whatever the comparator
  // found. On a long, sparse arc the comparator can only walk from the hyperbolic optimum
  // (`unbound-unsearched`), and sometimes no orbit with e < 1 could be fitted at all
  // (`unbound-no-bound-fit`); the free fit is still the best orbit in hand and is shown, but the
  // note says what was and was not tried, so an e of 38 on nine observations over nine years reads
  // as one solution among several rather than as a finding. A genuinely earned unbound orbit
  // (1I, 2I, 3I) still gets no note.
  function unearnedNote(c) {
    if (!c || (c.kept !== "unbound-unsearched" && c.kept !== "unbound-no-bound-fit")) return null;
    const n = c.nights;
    const arc = `${n} night${n === 1 ? "" : "s"} over ${num(c.arc_days, 1)} days`;
    if (c.kept === "unbound-no-bound-fit") {
      return {
        head: "ℹ Unbound orbit shown — no bound orbit could be fitted to compare it with",
        detail: `The best-fitting orbit is unbound (e = ${num(c.e_free, 4)} at ${num(c.rms_free, 3)}″) and ` +
                `no orbit with e < 1 could be fitted to these observations at all, so nothing was weighed ` +
                `against it. ${arc} is too few to believe an unbound orbit on its own: a wrong linkage fits ` +
                `hyperbolic too. Read the elements as the one solution that converged, not as a finding.`,
      };
    }
    return {
      head: "ℹ Unbound orbit shown — the only bound alternative tried fits far worse, and no wider search was possible",
      detail: `The best-fitting orbit is unbound (e = ${num(c.e_free, 4)} at ${num(c.rms_free, 3)}″). The only ` +
              `bound alternative the engine could try on an arc this long and sparse — a local walk from that ` +
              `solution — fits at ${num(c.rms_bound, 2)}″, so the data refuse IT; but a bound orbit elsewhere, ` +
              `with a different number of revolutions across the gaps, was not searched for. ${arc} is too few ` +
              `to believe an unbound orbit on its own: a wrong linkage fits hyperbolic too. Read the elements ` +
              `as one solution among several, not as a finding.`,
    };
  }

  function refresh(p) {
    const box = ensureBox();
    if (!box) return;
    const u = p && p.unbound;
    const parts = [];
    const b = u ? (boundNote(u.check) || unearnedNote(u.check)) : null;
    if (b) parts.push(b);
    if (u && u.reminder && u.reminder.text) {
      parts.push({ head: "ℹ Unbound now — not necessarily interstellar", detail: u.reminder.text });
    }
    box.hidden = parts.length === 0;
    box.innerHTML = parts.map((x) =>
      `<div class="cw-head">${esc(x.head)}</div><div class="cw-detail">${esc(x.detail)}</div>`).join("");
  }

  document.addEventListener("pyroorb:render", (e) => refresh(e.detail));
})();
