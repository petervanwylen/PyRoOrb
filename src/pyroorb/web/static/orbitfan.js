"use strict";
// PyRoOrb web UI — the family of orbits an undetermined arc admits, drawn faintly in BOTH 3D
// views (heliocentric Orbit3D and Orbarium). Vanilla JS, no build step. Loaded AFTER app.js.
//
// When constraint.assess says the arc does not determine an orbit, the server sends no
// `orbit3d` (or, after "show an orbit anyway", the asked-for one) and, always,
// `orbit_family`: up to 12 orbits that fit -- range-profile nodes across the allowed distances
// on a single tracklet, covariance clones elsewhere (constraint.orbit_family). Why:
// a scene with the planets and no object read as a rendering bug; "it is one of these" is what
// the arc actually says. Each line's opacity follows its weight (the population posterior on a
// tracklet, popprior.py) -- shading only, nothing is decided by it.
//
// Hooked in without touching app.js: both views draw their planets through
// OrthoView.prototype._drawPlanets before the object, so the family goes in right after the
// planets and under the object's own orbit. The latest payload's family is kept from the
// "pyroorb:render" event rather than read off CURRENT, which app.js may set after render().

(function () {
  if (typeof OrthoView === "undefined") return;
  let family = null;

  const FAN_COLOR = "#56d4dd";
  const ALPHA_MIN = 0.10, ALPHA_SPAN = 0.35;

  const orig = OrthoView.prototype._drawPlanets;
  OrthoView.prototype._drawPlanets = function (f, jd, opts) {
    orig.call(this, f, jd, opts);
    const orbits = family && Array.isArray(family.orbits) ? family.orbits : [];
    if (!orbits.length || !f) return;
    const { ctx, stroke } = f;
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = FAN_COLOR;
    for (const o of orbits) {
      if (!o || !Array.isArray(o.path) || o.path.length < 2) continue;
      const w = typeof o.weight === "number" && isFinite(o.weight) ? Math.max(0, Math.min(1, o.weight)) : 1;
      ctx.globalAlpha = ALPHA_MIN + ALPHA_SPAN * w;
      stroke(o.path);
    }
    ctx.restore();
  };

  // Frame: an undetermined orbit has no aphelion to frame, so make sure the family fits.
  function reach(orbits) {
    let r = 0;
    for (const o of orbits) for (const p of (o.path || [])) r = Math.max(r, Math.hypot(p[0], p[1], p[2]));
    return r;
  }

  document.addEventListener("pyroorb:render", (e) => {
    const p = e.detail;
    family = p && p.orbit_family ? p.orbit_family : null;
    // Orbit3D frames itself on the object's orbit; with none (or one far smaller than the
    // family) widen to the family's reach, capped so one near-parabolic member cannot shrink
    // the inner planets to a dot.
    try {
      if (typeof orbit3d !== "undefined" && orbit3d) {
        if (family && family.orbits && family.orbits.length && !(p && p.orbit3d)) {
          orbit3d.span = Math.max(1.6, Math.min(reach(family.orbits), 12)) * 1.15;
        }
        orbit3d.draw();
      }
      if (typeof orbarium !== "undefined" && orbarium) orbarium.draw();
    } catch (err) { /* a drawing hiccup must never break the page */ }
  });
})();
