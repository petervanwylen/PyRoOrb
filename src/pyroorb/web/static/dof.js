"use strict";
// PyRoOrb web UI — degrees-of-freedom banner. Vanilla JS, no build step. Loaded AFTER app.js.
//
// A fit's statistical degrees of freedom are residual ROWS minus free parameters: each
// optical observation supplies TWO rows (RA, Dec), so a 6-element gravity fit is exactly
// determined at 3 observations (dof 0) and over-determined only from 4 up. The engine fits
// down to that floor -- arc_extend.FIT_MIN_OBS, and find_orb's own limit -- rather than
// refusing, because determinacy is genuinely reached there. What is NOT reached is any
// information about the fit's quality: at dof <= 0 the parameters can absorb every residual,
// so a small rms says nothing about the orbit (it is whatever the solver stopped at, and an
// exact solution exists), and the +/- sigmas the panel prints come from a covariance whose
// variance factor had no scale to estimate (`_compute_state_cov` clamps the divisor to 1 so
// nothing raises). Just above zero the sigmas are real but rest on one or two spare rows.
// Both are worth saying out loud next to an elements table.
//
// This is the count-based half of the story only. The failure that usually matters on a short
// arc -- a range-degenerate geometry -- is invisible to any observation count (six obs inside
// one tracklet are as degenerate as three); the unconstrained-orbit banner above catches that
// independently. The two are NOT separate banners when both apply (deliberately: two
// stacked warnings read as far more going on than there is): while the unconstrained banner is
// up (`p.constraint`) this one stays hidden and app.js's renderConstraintWarn appends
// `window.dofClause(p)` -- one sentence -- to that banner instead. The sigma sentence goes too,
// since a suppressed or opted-in orbit shows no +/- sigmas to caveat. Alone (a few obs over
// several nights that the covariance does call determined) it is still its own banner.
//
// Payload contract: p.dof = {dof, n_resids, n_params, n_obs} (serialise_fit), or null when
// the engine could not compute it. Renders on the "pyroorb:render" event like convergence.js,
// and reuses .constraint-warn so the three banners stack as one visual family.

(function () {
  const THIN = 2;            // dof at or below this (and above 0) is "barely over-determined"

  function ensureBanner() {
    let box = document.getElementById("dofBanner");
    if (!box) {
      // after the convergence banner when it exists, else straight after the constraint one
      const anchor = document.getElementById("convergedBanner")
                  || document.getElementById("constraintBanner");
      if (!anchor) return null;
      box = document.createElement("div");
      box.id = "dofBanner";
      box.className = "constraint-warn";
      box.hidden = true;
      anchor.insertAdjacentElement("afterend", box);
    }
    return box;
  }

  function texts(d) {
    const counts = `${d.n_obs} observation${d.n_obs === 1 ? "" : "s"} give ` +
                   `${d.n_resids} residuals for ${d.n_params} fitted parameters`;
    if (d.dof <= 0) {
      return {
        severe: true,
        head: "No degrees of freedom",
        detail: `${counts}, so the orbit passes exactly through its astrometry ` +
                "(dof " + d.dof + "). Whatever residuals remain are the solver's, not the " +
                "orbit's — there is no spare information to test the fit against, and the ± " +
                "sigmas below are not measured uncertainties.",
      };
    }
    return {
      severe: false,
      head: "Barely over-determined",
      detail: `${counts} — ${d.dof} degree${d.dof === 1 ? "" : "s"} of freedom. The ± sigmas ` +
              "rest on that much spare information, so treat them as indicative.",
    };
  }

  function thin(p) {
    const d = p && p.dof;
    return (d && typeof d.dof === "number" && d.dof <= THIN) ? d : null;
  }

  // The folded form for the unconstrained banner: the count, without the sigma caveat.
  window.dofClause = function (p) {
    const d = thin(p);
    if (!d) return null;
    const n = `${d.n_obs} observation${d.n_obs === 1 ? "" : "s"}`;
    return d.dof <= 0
      ? `With ${n} for ${d.n_params} fitted parameters there are no degrees of freedom ` +
        `(dof ${d.dof}): any orbit that fits passes exactly through the astrometry.`
      : `${n[0].toUpperCase()}${n.slice(1)} for ${d.n_params} fitted parameters leave only ` +
        `${d.dof} degree${d.dof === 1 ? "" : "s"} of freedom.`;
  };

  function refresh(p) {
    const box = ensureBanner();
    if (!box) return;
    const d = thin(p);
    const show = !!d && !(p && p.constraint);
    box.hidden = !show;
    if (!show) { box.innerHTML = ""; box.classList.remove("severe"); return; }
    const t = texts(d);
    box.classList.toggle("severe", t.severe);
    box.innerHTML = `<div class="cw-head">⚠ ${t.head}</div>` +
                    `<div class="cw-detail">${t.detail}</div>`;
  }

  document.addEventListener("pyroorb:render", (e) => refresh(e.detail));
})();
