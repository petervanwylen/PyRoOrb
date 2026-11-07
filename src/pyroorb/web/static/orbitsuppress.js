// Orbit suppression: what the elements panel and the Export menu do when the fit's
// `constraint` verdict says this arc does not determine an orbit (src/pyroorb/constraint.py).
//
// By design, in that state there is no nominal orbit to show. Not a value in
// grey, not a value with a big +/- -- none. A reader cannot tell a free-valley nominal from a
// measured one, it will not reproduce against find_orb (every code lands somewhere different in
// the same valley), and an exported MPCORB line outlives every caveat attached to it. What the
// panel shows instead is the interval the server sends per element: the central 5-95% of the
// orbits that fit this astrometry.
//
// A sibling module rather than app.js edits (app.js is over the 128 KiB write cap); app.js calls
// the two globals below. Classic scripts share global scope, so `$` and friends are available.
(function () {
  "use strict";

  // Format one end of an interval. Wide is the point here, so precision is modest and the
  // exponent shows up rather than a wall of zeros.
  function nfmt(v) {
    if (typeof v !== "number" || !isFinite(v)) return "—";
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(2);
    return String(Number(v.toPrecision(5)));
  }

  // The tbody HTML for a suppressed orbit, or null when the fit is determined (app.js then
  // renders its ordinary value +/- sigma rows). `rows` is whichever element set is on screen --
  // heliocentric or planetocentric -- so the satellite toggle cannot walk around the rule.
  window.suppressedElementRows = function (p, rows) {
    if (p && p.nominal_optin) return optinRows(p, rows);
    if (!p || !p.orbit_suppressed) return null;
    const c = p.constraint || {};
    const pct = c.ranges_pct || [5, 95];
    const how = c.ranges_method === "sr"
      ? "the Statistical-Ranging family (all distance solutions this tracklet admits)"
      : "the covariance clone family";
    const lbl = `${nfmt(pct[0])}–${nfmt(pct[1])}%`;
    return (rows || []).map(function (e) {
      const s = SYM[e.name] || e.name;
      const u = e.unit ? ` ${e.unit}` : "";
      const r = e.range;
      const cell = (r && typeof r[0] === "number" && typeof r[1] === "number")
        ? `${nfmt(r[0])} … ${nfmt(r[1])}${u}`
        : `<span class="os-none">not determined</span>`;
      const title = (r ? `${lbl} of ${how}. No single value is shown because this arc does not `
                       + `determine one.` : "The family of fitting orbits could not be characterised.");
      return `<tr><td class="k" title="${e.label}">${s}</td>` +
             `<td class="v os-range" title="${title}">${cell}</td>` +
             `<td class="s os-pct" title="${title}">${r ? lbl : ""}</td></tr>`;
    }).join("");
  };

  // The asked-for orbit: the values come back, each marked "≈", and the
  // sigma column says what a linear sigma here would hide -- that the arc does not determine
  // the value. The 5-95% span of the fitting family stays one hover away.
  function optinRows(p, rows) {
    const c = p.constraint || {};
    const pct = c.ranges_pct || [5, 95];
    const lbl = `${nfmt(pct[0])}–${nfmt(pct[1])}%`;
    return (rows || []).map(function (e) {
      const s = SYM[e.name] || e.name;
      const u = e.unit ? ` ${e.unit}` : "";
      const r = e.range;
      const span = (r && typeof r[0] === "number" && typeof r[1] === "number")
        ? ` The ${lbl} span of the orbits that fit is ${nfmt(r[0])} … ${nfmt(r[1])}${u}.` : "";
      const title = `Shown at your request: this arc does not determine ${e.label || s}.${span}`;
      const v = (typeof e.value === "number" && isFinite(e.value))
        ? `≈ ${typeof fmt === "function" ? fmt(e.value, e.name === "Tp" ? 3 : 6) : nfmt(e.value)}${u}`
        : `<span class="os-none">—</span>`;
      return `<tr><td class="k" title="${e.label}">${s}</td>` +
             `<td class="v os-optin" title="${title}">${v}</td>` +
             `<td class="s os-pct" title="${title}">undetermined</td></tr>`;
    }).join("");
  }

  // The 3D heliocentric pane. The server sends no `orbit3d` for a suppressed fit, so the
  // canvas would otherwise draw the Sun and planets and simply omit the object -- a picture
  // that looks like a rendering bug rather than a statement. Say it instead -- and
  // the pane draws the orbits that DO fit (orbitfan.js), which the note names.
  function orbitPaneNote(supp, fam) {
    const bd = document.querySelector("#viz3d .bd");
    if (!bd) return;
    let note = document.getElementById("orbitSuppressNote");
    if (!note) {
      note = document.createElement("div");
      note.id = "orbitSuppressNote";
      note.className = "orbit-supp-note";
      bd.appendChild(note);
    }
    note.hidden = !supp;
    const n = fam && Array.isArray(fam.orbits) ? fam.orbits.length : 0;
    note.classList.toggle("with-fan", supp && n > 0);
    note.textContent = !supp ? ""
      : n ? `This arc does not determine one orbit — the faint lines are ${n} that fit it` +
            (fam.source === "range" ? ", across the distances the astrometry allows (brighter = " +
                                      "more typical of the known asteroid population)." : ".")
          : "No orbit to draw — this arc does not determine one. The element ranges are in the panel on the left.";
  }

  // The Export menu. The orbit products refuse (the server refuses them too -- an export URL is
  // a plain GET); the ephemeris stays, because a sky prediction is not an orbit and is what a
  // short-arc object is actually wanted for.
  // app.js's ELEM_SYM is a top-level `const`, which is NOT a window property in a classic
  // script -- so the two Greek symbols are repeated here rather than reached for and missed.
  const SYM = { asc_node: "\u03a9", arg_per: "\u03c9" };

  const ORBIT_EXPORTS = ["mpec", "elements", "mpcorb", "sof", "state", "covar"];
  window.applyOrbitSuppression = function (p) {
    const menu = document.getElementById("exportMenu");
    const supp = !!(p && p.orbit_suppressed);
    orbitPaneNote(supp, p && p.orbit_family);
    optinBar(p);
    ORBIT_EXPORTS.forEach(function (kind) {
      const b = menu && menu.querySelector(`button[data-export="${kind}"]`);
      if (!b) return;
      b.disabled = supp;
      if (supp) {
        b.title = "Unavailable: this arc does not determine an orbit, so there is no orbit to "
                + "export. The element ranges on the left are what it does say.";
      } else {
        b.removeAttribute("title");
      }
    });
  };

  // "Show an orbit anyway". The banner keeps saying the arc does not
  // determine an orbit; this bar offers the one-click way past it, and once taken says WHICH
  // orbit is on screen and how to go back. POST /api/nominal (engine_api.nominal_optin): on a
  // single tracklet, or a longer arc delivered near-parabolic or unbound, the most probable orbit
  // given the known population, else the nominal fit.
  // The request is per orbit -- any re-solve returns the page to the ranges by itself.
  function ensureBar() {
    let bar = document.getElementById("optinBar");
    if (!bar) {
      const anchor = document.getElementById("constraintBanner");
      if (!anchor) return null;
      bar = document.createElement("div");
      bar.id = "optinBar";
      bar.className = "constraint-warn optin-bar";
      bar.hidden = true;
      anchor.insertAdjacentElement("afterend", bar);
    }
    return bar;
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function au(x) { return (typeof x === "number" && isFinite(x)) ? `${nfmt(x)} au` : "?"; }

  function optinBar(p) {
    const bar = ensureBar();
    if (!bar) return;
    const info = p && p.nominal_optin;
    const live = !!(p && (p.session || p.restore || p.refit));
    if (info) {
      let what;
      if (info.source === "population") {
        const pc = info.rho_pct || {};
        what = `the <b>most probable orbit</b> given the known asteroid population, at ${au(info.rho_med)} ` +
               `from the observer (5–95%: ${au(pc["5"])} – ${au(pc["95"])})`;
      } else {
        what = "the <b>nominal least-squares orbit</b>";
      }
      bar.hidden = false;
      bar.innerHTML =
        `<div class="cw-head">≈ Showing ${what}, at your request.</div>` +
        `<div class="cw-detail">Other orbits fit this astrometry as well; values are marked ≈ and ` +
        `exports now publish this orbit. Hazard figures (PHA, MOID) stay blank. ` +
        `<button type="button" id="optinOff">Back to the ranges</button> ` +
        `<span id="optinStatus"></span></div>`;
      const b = document.getElementById("optinOff");
      if (b) b.addEventListener("click", () => request(false));
      return;
    }
    if (p && p.orbit_suppressed && live) {
      bar.hidden = false;
      bar.innerHTML =
        `<div class="cw-detail">Want an orbit to look at anyway? ` +
        `<button type="button" id="optinOn">Show an orbit anyway</button> ` +
        `<span class="os-hint">— on a single night, or where the fit is near-parabolic, the most ` +
        `probable one given the known asteroid population; the warning stays and the values ` +
        `are marked ≈.</span> ` +
        `<span id="optinStatus"></span></div>`;
      const b = document.getElementById("optinOn");
      if (b) b.addEventListener("click", () => request(true));
      return;
    }
    bar.hidden = true;
    bar.innerHTML = "";
  }

  async function request(on) {
    const status = document.getElementById("optinStatus");
    const say = (h) => { if (status) status.innerHTML = h; };
    try {
      if (typeof CURRENT === "undefined" || !CURRENT) return;
      if (typeof ensureLive === "function" && !(await ensureLive())) return;
      say(`<span class="loading">${on ? "finding the orbit…" : "restoring…"}</span>`);
      const r = await fetch("/api/nominal", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session: CURRENT.session, on: on }),
      });
      let p = await r.json();
      if (typeof FitJob !== "undefined" && FitJob.settle) {
        p = await FitJob.settle(p, (t) => say(`<span class="loading">${esc(t)}</span>`));
      }
      if (!p || !p.ok) {
        say(`<span class="bad">${esc((p && p.error) || "request failed")}</span>`);
        return;
      }
      CURRENT = p;
      render(p);
      const ws = document.getElementById("workspace");
      if (ws) ws.hidden = false;
    } catch (e) {
      say(`<span class="bad">request failed: ${esc(e.message)}</span>`);
    }
  }
})();
