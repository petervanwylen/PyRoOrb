"""Ephemeris corridor windows for the ITF search.

Sweeps a geocentric (obscode 500) ephemeris of the fitted orbit across the
ITF era and cuts it into quadratic **corridor windows**: each window carries
RA/Dec quadratics (degrees vs seconds), a search box, the predicted sky-plane
uncertainty, apparent magnitude and on-sky speed.  Candidate retrieval
(:mod:`pyroorb.itf.candidates`) then matches tracklets against these windows.

This replaces find_orb's ``epout.txt`` text parsing: the ephemeris comes from
:func:`ephemeris.compute_ephemeris` (our own propagator) and the per-node
uncertainty is the fit's real covariance projected on the sky (J.Cov.J^T
semi-major axis) rather than find_orb's ephemeris-uncertainty column.  The
window construction itself (node spacing, quadratic through three nodes,
fast-mover splitting, 0/360-seam double windows, the geocentric parallax
allowance) is ported from the proven notebook pipeline.
"""

import math
from dataclasses import dataclass

import numpy as np

from pyroorb.ephemerides import compute_ephemeris

from .obs80_fields import EARTH_RAD_AU, unix_from_jd, shift_above_180, is_crossing_360


@dataclass
class Window:
    """One corridor window: the object should be inside the quadratic corridor
    ``ra(t) = ra_quad[0]*dt^2 + ra_quad[1]*dt + ra_quad[2]`` (dt = epoch_sec -
    unix_cur; same for dec) between ``unix_start`` and ``unix_end``.

    ``uncert``/``uncert_ra`` are 1-sigma corridor half-widths in degrees
    (Dec / RA-coordinate); ``max_speed`` is on-sky motion in arcsec/min
    (numerically equal to arcmin/hr, the find_orb ephemeris unit the legacy
    tolerances were tuned on).  RA values may live in the shifted (-180, 180]
    or (180, 540] frame near the 0/360 seam -- seam windows are emitted twice,
    once per frame, exactly like the legacy double loop.
    """
    unix_cur: float
    unix_start: float
    unix_end: float
    ra_quad: tuple
    dec_quad: tuple
    min_ra: float
    max_ra: float
    min_dec: float
    max_dec: float
    uncert: float
    uncert_ra: float
    max_speed: float
    dist_au: float
    mag: float | None = None
    elong: float | None = None          # solar elongation, deg (ML feature)
    # The same three nodes fitted as a UNIT VECTOR -- (x_quad, y_quad, z_quad),
    # each a 3-tuple in the same ``dt`` as ra_quad/dec_quad.  See
    # :meth:`predict_sky` for why this exists beside the RA/Dec pair.  Optional
    # so a hand-built Window (the SQL fuzz, the unit tests) stays constructible.
    vec_quad: tuple | None = None

    def predict(self, epoch_sec):
        """Corridor (ra, dec) prediction in window RA frame at ``epoch_sec``.

        RETRIEVAL currency: this is the polynomial the database predicate is
        written in (:mod:`itf.candidates` ships ``ra_a/ra_b/ra_c`` into SQL),
        so its form is a contract with the server and with the motion
        prefilter.  For a trajectory JUDGEMENT use :meth:`predict_sky`.
        """
        dt = epoch_sec - self.unix_cur
        ra = self.ra_quad[0] * dt * dt + self.ra_quad[1] * dt + self.ra_quad[2]
        dec = self.dec_quad[0] * dt * dt + self.dec_quad[1] * dt + self.dec_quad[2]
        return ra, dec

    def predict_sky(self, epoch_sec):
        """(ra, dec) from the UNIT-VECTOR quadratic -- the pole-safe prediction.

        RA is a singular coordinate near the poles: at dec 82 deg an object
        crossing the sky at an ordinary rate sweeps ~20 deg of RA in 3 days,
        with curvature to match, and a quadratic through three nodes cannot
        hold it.  The error is small in POSITION and large in RATE, which is
        exactly the wrong way round for :func:`itf.search._tracklet_candidate`,
        whose ``qual`` is a rate comparison judged against a 1-2.5 arcsec gate
        while the corridor it sits in is thousands of arcsec wide.

        Measured on 2016 MB1 (i 53 deg, dec +82 deg at the epoch), a real
        MPC-linked 2024 tracklet of it: the RA/Dec quadratic under-predicts a
        257.6" displacement by 10.7" and reports ``qual`` 11.20" -- over the
        6.5" bound, so "unlikely", never fitted, and an entire apparition lost.
        The SAME three nodes fitted as a unit vector give 3.24", against 3.18"
        for a true geocentric ephemeris at the two instants (the remainder is
        diurnal parallax, which the gate's ``traj_unc`` already allows).  The
        tracklet fits the orbit at 0.27".

        Interpolating the direction rather than its spherical coordinates costs
        one extra polyfit per window at build time and nothing at predict time.
        Falls back to :meth:`predict` for a Window built without ``vec_quad``.
        """
        if self.vec_quad is None:
            return self.predict(epoch_sec)
        dt = epoch_sec - self.unix_cur
        x, y, z = (q[0] * dt * dt + q[1] * dt + q[2] for q in self.vec_quad)
        r = math.sqrt(x * x + y * y + z * z)
        if not r:
            return self.predict(epoch_sec)
        return math.degrees(math.atan2(y, x)) % 360.0, \
            math.degrees(math.asin(max(-1.0, min(1.0, z / r))))


def _node(row):
    """Reduce an ephemeris row to the fields the corridor needs."""
    return {
        "unix": unix_from_jd(row["jd_utc"]),
        "ra": row["ra_deg"],
        "dec": row["dec_deg"],
        "delta": row["delta_au"],
        "mag": row.get("v_mag"),
        "speed": row.get("motion_arcsec_min") or 0.0,
        "unc_deg": (row.get("unc_smaj_arcsec") or 0.0) / 3600.0,
        "elong": row.get("elong_deg"),
    }


def _sep_deg(a, b):
    """On-sky separation between two nodes, degrees (small-angle, cos-dec)."""
    cosd = math.cos(math.radians(0.5 * (a["dec"] + b["dec"])))
    dra = ((b["ra"] - a["ra"] + 180.0) % 360.0 - 180.0) * cosd
    return math.hypot(dra, b["dec"] - a["dec"])


def refine_fast_segments(fit, obscodes, cfg, nodes):
    """Re-sample the ephemeris FINELY wherever the object moves fast on the sky.

    A window is a quadratic through three nodes, so its fidelity depends on how
    far the object travels between them -- and a fast mover breaks it.  2015
    KJ19 is the case: a NEO running 650-2900"/hr, whose tracklets the corridor
    served and then classified `unlikely` at every depth, because the motion
    mismatch the classifier measures came out 2-17x too large.  Per tracklet,
    window-derived vs a direct ephemeris on the same orbit: 18" vs 4", 42" vs
    9", 140" vs 9".  Against a 1-2.5" gate (plus drift allowance) that is the
    difference between linking the object and never trying it.  At the default
    3 d spacing only 13 of its 37 tracklets passed the motion gate; at 1 d, 35.

    Uniform refinement is the wrong price -- 1 d spacing triples the ephemeris
    of every sweep, and a main-belt object needs none of it.  So the step
    adapts: wherever consecutive nodes are more than ``cfg.fast_node_deg``
    apart on the sky, that stretch is recomputed at a step scaled to bring the
    per-node motion back to that bound (floored at ``cfg.min_node_step_days``),
    and spliced in.  Fast stretches are short -- they are close approaches --
    so the extra nodes land only where the geometry demands them.
    ``cfg.fast_node_budget`` caps the total added, coarsening the refinement
    rather than letting a many-approach orbit blow the sweep up.

    Returns the merged node list (time-ordered); the input list when nothing
    qualifies.  Node spacing is used by the quadratic through its ACTUAL times,
    so a non-uniform list is fine.
    """
    if not getattr(cfg, "fast_node_deg", 0) or len(nodes) < 2:
        return nodes
    runs, i = [], 0
    while i < len(nodes) - 1:
        if _sep_deg(nodes[i], nodes[i + 1]) <= cfg.fast_node_deg:
            i += 1
            continue
        j, worst = i, 0.0
        while (j < len(nodes) - 1
               and _sep_deg(nodes[j], nodes[j + 1]) > cfg.fast_node_deg):
            worst = max(worst, _sep_deg(nodes[j], nodes[j + 1]))
            j += 1
        runs.append([nodes[i]["unix"], nodes[j]["unix"], worst])
        i = j
    if not runs:
        return nodes

    def _step_for(worst):
        return max(cfg.ephem_step_days * cfg.fast_node_deg / worst,
                   cfg.min_node_step_days)

    # honour the budget by coarsening every run in proportion, never by
    # dropping one: a silently unrefined approach is the bug this fixes
    extra = sum((b - a) / 86400.0 / _step_for(w) for a, b, w in runs)
    scale = max(1.0, extra / float(cfg.fast_node_budget or extra or 1))
    out = list(nodes)
    for a, b, worst in runs:
        step = _step_for(worst) * scale
        jd0 = a / 86400.0 + 2440587.5
        jd1 = b / 86400.0 + 2440587.5
        try:
            fine = [_node(r) for r in compute_ephemeris(
                fit, obscodes, jd0, jd1, step, site="500", full=False)]
        except Exception:
            continue                      # keep the coarse nodes for this run
        out = [n for n in out if not (a < n["unix"] < b)] + fine
    out.sort(key=lambda n: n["unix"])
    dedup = [out[0]]
    for n in out[1:]:
        if n["unix"] - dedup[-1]["unix"] > 1.0:
            dedup.append(n)
    return dedup


def sweep_nodes(fit, obscodes, cfg, jd_start=None, jd_stop=None):
    """Geocentric ephemeris nodes for the corridor sweep.

    Defaults to ``cfg.ephem_start`` .. (start + ephem_steps*step), or to ~30
    days past the fit's last observation when ``ephem_steps`` is None.
    """
    from datetime import datetime, timezone

    if jd_start is None:
        dt = datetime.fromisoformat(cfg.ephem_start).replace(tzinfo=timezone.utc)
        jd_start = dt.timestamp() / 86400.0 + 2440587.5
    if jd_stop is None:
        if cfg.ephem_steps is not None:
            jd_stop = jd_start + cfg.ephem_steps * cfg.ephem_step_days
        else:
            now_jd = (datetime.now(timezone.utc).timestamp() / 86400.0
                      + 2440587.5)
            # QUANTIZED to the whole day: the grid is then identical for every
            # sweep of a search (and every search that day), which is what lets
            # the geocentric observer positions memoize across recursion depths
            # and LOV variants (obs_sources._geocentric_positions) -- otherwise
            # `now` differs by microseconds per call and every sweep re-runs the
            # same erfa work.  Also makes a day's searches reproducible.
            jd_stop = math.floor(now_jd) + 30.0
    rows = compute_ephemeris(fit, obscodes, jd_start, jd_stop,
                             cfg.ephem_step_days, site="500", full=False)
    return refine_fast_segments(fit, obscodes, cfg, [_node(r) for r in rows])


def faint_cut(cfg, nodes):
    """The magnitude cut the corridor applies to an object with these nodes.

    A cut the user set (``cfg.mag_limit_explicit``) is returned as given.
    Otherwise ``cfg.mag_limit``, deepened when needed to ``cfg.mag_floor_margin``
    fainter than the brightest node at or after ``cfg.mag_floor_since`` (see
    the config note).  Without such a node -- nodes all older than that, or no
    predicted magnitude -- the configured cut stands.
    """
    margin = cfg.mag_floor_margin
    if cfg.mag_limit_explicit or margin is None or not cfg.mag_floor_since:
        return cfg.mag_limit
    from datetime import datetime, timezone
    since = datetime.fromisoformat(cfg.mag_floor_since).replace(
        tzinfo=timezone.utc).timestamp()
    mags = [n["mag"] for n in nodes
            if n.get("mag") is not None and n["unix"] >= since]
    if not mags:
        return cfg.mag_limit
    return max(cfg.mag_limit, min(mags) + margin)


def corridor_windows(nodes, cfg, max_uncert_deg, mag_limit=None):
    """Cut ephemeris nodes into corridor :class:`Window`\\ s.

    Mirrors the legacy pipeline: windows of 4 node steps (quadratic through
    nodes i, i+2, i+4), split into two 2-step halves when the sky motion
    across the window exceeds ``cfg.max_fallback_deg`` (fast movers), the
    0/360-seam double emission, and the per-window gate
    ``uncert < max_uncert and mag < mag_limit``.  ``uncert`` adds the
    geocentric-parallax allowance ``earth_radius / delta`` on top of the
    covariance sky-sigma, since all nodes are geocentric.

    ``mag_limit`` defaults to :func:`faint_cut` of these nodes; a caller
    holding only a SLICE of the object's track (the archive era, one run of a
    variant) passes the cut judged on the whole track instead.
    """
    if mag_limit is None:
        mag_limit = faint_cut(cfg, nodes)
    windows = []
    for i in range(0, len(nodes) - 4, 4):
        raw = [dict(n) for n in (nodes[i], nodes[i + 1], nodes[i + 2],
                                 nodes[i + 3], nodes[i + 4])]
        crossing_loops = 1
        if is_crossing_360(raw[0]["ra"], raw[2]["ra"], raw[4]["ra"]):
            crossing_loops = 2
            for n in raw:
                n["ra"] = shift_above_180(n["ra"])
        for loop in range(crossing_loops):
            if loop == 1:
                # second seam pass: same window in the (180, 540] frame so a
                # plain BETWEEN on stored RA matches the other side of 360
                raw = [dict(n, ra=n["ra"] + 360.0) for n in raw]
            cur0, cur1, mid0, mid1, nex0 = raw

            if (abs(cur0["ra"] - nex0["ra"]) > cfg.max_fallback_deg
                    or abs(cur0["dec"] - nex0["dec"]) > cfg.max_fallback_deg):
                triplets = [(cur0, cur1, mid0), (mid0, mid1, nex0)]
            else:
                triplets = [(cur0, mid0, nex0)]

            for cur, mid, nex in triplets:
                # 1-sigma corridor half-width (deg): worst node sky-sigma plus
                # the geocentric-parallax allowance for a topocentric observer
                uncert = (max(nex["unc_deg"], cur["unc_deg"])
                          + EARTH_RAD_AU / min(nex["delta"], cur["delta"])
                          * 57.3)
                # BRIGHTEST of the window's nodes, not the first one.  The
                # gate asks "could this window hold a detection at all", which
                # is a question about the whole window; reading it off `cur`
                # made the answer depend on where the 4-node tiling happened to
                # start.  2016 EZ157 showed what that costs: inserting refined
                # nodes EARLIER in the sweep re-phased the tiling, the window
                # over 2021-10-16 came to start on a 23.50 node instead of a
                # 23.4 one, and an apparition the search had been recovering
                # vanished -- same orbit, same nodes near the epoch, different
                # alignment.
                mags = [n["mag"] for n in (cur, mid, nex) if n["mag"] is not None]
                mag = min(mags) if mags else None
                if uncert >= max_uncert_deg:
                    continue
                if mag is not None and mag >= mag_limit:
                    continue

                t0 = cur["unix"]
                ts = [0.0, mid["unix"] - t0, nex["unix"] - t0]
                ra_quad = np.polyfit(ts, [cur["ra"], mid["ra"], nex["ra"]], 2)
                dec_quad = np.polyfit(ts, [cur["dec"], mid["dec"], nex["dec"]], 2)
                # ...and the same nodes as a unit vector, for the trajectory
                # gate (see Window.predict_sky).  The RA frame shift above is
                # irrelevant here: a direction has no seam.
                vecs = np.array([[math.cos(math.radians(n["dec"]))
                                  * math.cos(math.radians(n["ra"])),
                                  math.cos(math.radians(n["dec"]))
                                  * math.sin(math.radians(n["ra"])),
                                  math.sin(math.radians(n["dec"]))]
                                 for n in (cur, mid, nex)])
                vec_quad = tuple(tuple(float(c) for c in np.polyfit(ts, vecs[:, k], 2))
                                 for k in range(3))

                min_dec = min(cur["dec"], mid["dec"], nex["dec"])
                windows.append(Window(
                    unix_cur=t0,
                    unix_start=cur["unix"],
                    unix_end=nex["unix"],
                    ra_quad=tuple(float(c) for c in ra_quad),
                    dec_quad=tuple(float(c) for c in dec_quad),
                    vec_quad=vec_quad,
                    min_ra=min(cur["ra"], mid["ra"], nex["ra"]),
                    max_ra=max(cur["ra"], mid["ra"], nex["ra"]),
                    min_dec=min_dec,
                    max_dec=max(cur["dec"], mid["dec"], nex["dec"]),
                    uncert=uncert,                     # 1-sigma, unscaled
                    uncert_ra=uncert / math.cos(math.radians(min_dec)),
                    max_speed=max(cur["speed"], mid["speed"], nex["speed"]),
                    dist_au=min(nex["delta"], cur["delta"]),
                    mag=mag,
                    elong=cur.get("elong"),
                ))
    return windows
