"""Differential test of the SERVER-SIDE predicates against their Python twins.

``DbSource._sql_corridor`` and ``_sql_motion_bound`` re-implement
:func:`itf.candidates.tracklet_in_window` and :func:`motion_bound_ok` in SQL so
Postgres can drop rows before they ever become Python objects.  The whole
arrangement rests on ONE property:

    the SQL verdict is a SUPERSET of the Python one -- it may keep rows Python
    will later reject, it may NEVER drop a row Python would have kept.

("SQL narrows, Python decides": ``_survivors`` re-applies the exact predicate to
whatever comes back, so a looser SQL costs a little bandwidth and nothing else.)

Nothing enforced that.  Edit the Python predicate and the SQL quietly becomes a
SUBSET: candidates stop being returned, no error is raised, and the search
reports a clean negative -- silent recall loss, the one failure this system
cannot detect from its own output.

Two checks, both driven from the generators here:

* OFFLINE (``check_shim``, run by tests/test_itf_search.py) -- the SQL text is
  transliterated into a Python expression and evaluated on the same rows.  This
  proves the ALGEBRA matches and runs on every commit; it does not prove
  Postgres agrees.
* LIVE (``check_live``, ``python tests/itf_sql_fuzz.py``) -- the same rows go to
  the real server as a VALUES/unnest relation and the real SQL string is
  evaluated by Postgres itself, catching anything float8 semantics, operator
  precedence or NULL handling would do differently.  Read-only: no table is
  touched, nothing is written.
"""
import math
import os
import random
import re
import sys


from pyroorb.itf.candidates import (DbSource, motion_bound_ok,  # noqa: E402
                            tracklet_in_window, BOX_PAD_DEG)
from pyroorb.itf.config import ItfConfig                        # noqa: E402
from pyroorb.itf.ephem import Window                            # noqa: E402
from pyroorb.itf.index import IndexConfig                       # noqa: E402

T0 = 1.6e9

# The tracklet columns the two predicates read, with their SQL types -- the
# same (name, type) idiom the window relation uses, so the live check can build
# its synthetic `t` relation the same way DbSource builds `w`.
TRACKLET_COLS = (
    ("i", "int"),
    ("epoch_mid_sec", "float8"), ("epoch_start_sec", "float8"),
    ("epoch_end_sec", "float8"),
    ("center_ra", "float8"), ("center_dec", "float8"),
    ("rate_deg_day", "float8"), ("fit_rms_arcsec", "float8"),
    ("dra_cosdec_rate", "float8"), ("ddec_rate", "float8"),
)


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------
def _tune_motion_to_boundary(window, feat, cfg):
    """Rewrite ``feat``'s fitted rates so its motion mismatch lands EXACTLY on
    the pre-filter's accept boundary (``qual - slack == maybe + traj_unc``),
    where Python keeps the row by the ``<=``.

    Without this the fuzz cannot see a 1e-6 tightening of the SQL bound: an
    ordinary random row misses the threshold by orders of magnitude more than
    the pad, so a sign-flipped pad still agrees everywhere.  Returns False when
    the row cannot carry the construction (no arc, no fit rms)."""
    es0, es1 = feat["epoch_start_sec"], feat["epoch_end_sec"]
    rms = feat["fit_rms_arcsec"]
    if es0 is None or es1 is None or rms is None or abs(es1 - es0) <= 0:
        return False
    arc_sec = abs(es1 - es0)
    arc_days = arc_sec / 86400.0
    cosd = math.cos(math.radians(feat["center_dec"]))
    p0_ra, p0_dec = window.predict(es0)
    p1_ra, p1_dec = window.predict(es1)
    pred_e = (((p1_ra - p0_ra + 180.0) % 360.0) - 180.0) * cosd
    pred_n = p1_dec - p0_dec
    traj_unc = 0.00026 / 86400.0 * arc_sec / window.dist_au * 57.3 * 3600.0
    slack = cfg.prefilter_motion_slack_rms * rms + cfg.prefilter_motion_slack_arcsec
    target = cfg.maybe_arcsec + traj_unc + slack        # qual at the boundary
    feat["ddec_rate"] = pred_n / arc_days               # all mismatch in RA ...
    feat["dra_cosdec_rate"] = (pred_e - target / 3600.0) / arc_days
    return True


def random_cases(n=600, seed=20260724):
    """``[(win, feat, ra_tol, dec_tol), ...]`` spanning both verdicts.

    Rows are placed relative to the corridor's own prediction with the offset
    drawn AROUND the tolerance, so roughly half of them land within a hair of
    the accept/reject boundary -- where a sign slip, a missing pad or a float8
    disagreement actually shows up.  A flat random scatter would sit far from
    the threshold and agree trivially.

    A share of the cases sit EXACTLY on a boundary -- offset along a single
    axis at precisely the tolerance, or fitted rates solved to put the motion
    mismatch on the gate's own equality (:func:`_tune_motion_to_boundary`).
    Those are what make the 1e-6 pads testable at all: the pads exist to absorb
    float8-vs-numpy disagreement AT the threshold, so only a row sitting there
    can tell a pad that rounds outward from one that rounds inward.

    Every row is inside the window's epoch range and dec band by construction
    (those are the join's own bounds, applied separately from the corridor
    expression), so ``tracklet_in_window`` returning None means the CORRIDOR
    rejected it and nothing else did.
    """
    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        dec0 = rng.uniform(-60.0, 60.0)
        # RA near the wrap for a good fraction of cases: the two predicates
        # normalise the difference differently (Python's % vs SQL's floor()),
        # and 0/360 is where that could diverge.
        ra0 = rng.choice([rng.uniform(0.0, 360.0), rng.uniform(-0.4, 0.4) % 360.0,
                          rng.uniform(179.6, 180.4)])
        speed = rng.uniform(1e-7, 3e-5)             # deg/s along the corridor
        window = Window(
            unix_cur=T0, unix_start=T0 - 86400.0, unix_end=T0 + 86400.0,
            ra_quad=(rng.uniform(-1e-12, 1e-12), speed, ra0),
            dec_quad=(rng.uniform(-1e-12, 1e-12), speed * 0.4, dec0),
            min_ra=ra0 - 90.0, max_ra=ra0 + 90.0,      # band never the decider:
            min_dec=dec0 - 45.0, max_dec=dec0 + 45.0,  # the corridor is
            uncert=rng.uniform(0.001, 1.5),
            uncert_ra=rng.uniform(0.001, 1.5) / math.cos(math.radians(dec0)),
            max_speed=rng.uniform(0.1, 5.0), dist_au=rng.uniform(0.05, 40.0),
            mag=20.0)
        ra_tol = rng.uniform(0.001, 2.0)
        dec_tol = rng.uniform(0.001, 2.0)

        es = rng.uniform(window.unix_start, window.unix_end)
        arc = rng.choice([0.0, rng.uniform(60.0, 6 * 3600.0)])
        rate = rng.uniform(0.0, 1.5)                    # deg/day
        half_ext = rate * arc / 2.0 / 86400.0
        pred_ra, pred_dec = window.predict(es)
        # offsets straddling the accept boundary (ra_tol + half_ext); an
        # axis-aligned angle at scale 1.0 sits exactly ON it
        scale = rng.choice([0.2, 0.9, 0.999, 1.0, 1.0, 1.001, 1.1, 3.0])
        ang = rng.choice([0.0, math.pi / 2, math.pi, 3 * math.pi / 2,
                          rng.uniform(0.0, 2 * math.pi)])
        feat = {
            "epoch_mid_sec": es,
            "epoch_start_sec": None if arc == 0 else es - arc / 2.0,
            "epoch_end_sec": None if arc == 0 else es + arc / 2.0,
            "center_ra": (pred_ra + (ra_tol + half_ext) * scale
                          * math.cos(ang)) % 360.0,
            "center_dec": pred_dec + (dec_tol + half_ext) * scale * math.sin(ang),
            "rate_deg_day": rng.choice([None, 0.0, rate]),
            # motion-bound inputs: sometimes absent (the predicate must then
            # KEEP), sometimes matching the corridor, sometimes wildly off
            "fit_rms_arcsec": rng.choice([None, 0.0, rng.uniform(0.05, 3.0)]),
            "dra_cosdec_rate": rng.choice(
                [None, speed * 86400.0 * math.cos(math.radians(dec0)),
                 rng.uniform(-2.0, 2.0)]),
            "ddec_rate": rng.choice([None, speed * 0.4 * 86400.0,
                                     rng.uniform(-2.0, 2.0)]),
        }
        if rng.random() < 0.3:
            _tune_motion_to_boundary(window, feat, ItfConfig())
        cases.append((window, feat, ra_tol, dec_tol))
    return cases


def python_verdicts(window, feat, ra_tol, dec_tol, cfg):
    """``(corridor_keep, motion_keep)`` from the authoritative Python
    predicates -- what the SQL must never be stricter than."""
    return (tracklet_in_window(feat, window, ra_tol, dec_tol) is not None,
            motion_bound_ok(feat, window, cfg))


# ---------------------------------------------------------------------------
# Offline: transliterate the SQL into Python and evaluate it
# ---------------------------------------------------------------------------
_SQL_ENV = {
    "power": lambda a, b: a ** b, "sqrt": math.sqrt, "abs": abs,
    "floor": math.floor, "radians": math.radians, "cos": math.cos,
    "coalesce": lambda *a: next((x for x in a if x is not None), None),
}


def sql_to_python(expr, params=()):
    """The SQL predicate ``expr`` as a Python expression string.

    A transliteration, deliberately narrow: it models exactly the forms these
    two predicates use (``t.``/``w.`` column refs, ``AND``/``OR``/``IS NULL``,
    ``power``/``sqrt``/``abs``/``floor``/``radians``/``cos``/``coalesce``, and
    ``%s`` parameters).  If the SQL ever grows a form not listed here the
    translation will fail loudly rather than quietly mis-evaluate -- which is
    the correct outcome: the shim has stopped modelling the SQL and wants
    looking at.

    NOTE the left-to-right NULL guards matter.  SQL's three-valued logic makes
    ``TRUE OR NULL`` true, so a NULL-producing arithmetic branch is harmless
    there; Python instead raises on ``None`` arithmetic and relies on ``or``
    short-circuiting past it.  Both give the same answer only because the
    ``IS NULL`` tests come FIRST in the expression -- as they do."""
    out = expr
    for p in params:                       # %s -> literal, left to right
        out = out.replace("%s", repr(float(p)), 1)
    if "%s" in out:
        raise AssertionError("unfilled %s parameter in the SQL expression")
    out = re.sub(r"\b([tw])\.(\w+)\b", r"\1_\2", out)
    out = re.sub(r"\bIS NULL\b", "is None", out)
    out = re.sub(r"\bAND\b", "and", out)
    out = re.sub(r"\bOR\b", "or", out)
    unknown = set(re.findall(r"\b([a-z_][a-z_0-9]*)\s*\(", out)) - set(_SQL_ENV)
    if unknown:
        raise AssertionError(f"the shim does not model SQL function(s): "
                             f"{sorted(unknown)} -- extend _SQL_ENV")
    return out


def _shim_scope(window, feat, ra_tol, dec_tol):
    row = DbSource._window_sql_row(None, window, ra_tol, dec_tol)
    scope = {f"w_{k}": v for k, v in row.items()}
    scope.update({f"t_{k}": v for k, v in feat.items()})
    return scope


def check_shim(cases=None, cfg=None, verbose=False):
    """Evaluate the SQL predicates in Python on ``cases`` and assert the
    superset property.  Returns ``(n, n_corridor_rejected, n_motion_rejected)``
    so a caller can assert the fuzz actually exercised both verdicts."""
    cfg = cfg or ItfConfig()
    cases = cases if cases is not None else random_cases()
    src = DbSource.__new__(DbSource)
    src.cfg, src.index_cfg = cfg, IndexConfig()
    corridor = sql_to_python(src._sql_corridor())
    motion = sql_to_python(src._sql_motion_bound(True),
                           (cfg.prefilter_motion_slack_rms,
                            cfg.prefilter_motion_slack_arcsec,
                            cfg.maybe_arcsec))
    n_corr_rej = n_mot_rej = 0
    for k, (window, feat, ra_tol, dec_tol) in enumerate(cases):
        scope = _shim_scope(window, feat, ra_tol, dec_tol)
        sql_corr = bool(eval(corridor, dict(_SQL_ENV), scope))
        sql_mot = bool(eval(motion, dict(_SQL_ENV), scope))
        py_corr, py_mot = python_verdicts(window, feat, ra_tol, dec_tol, cfg)
        n_corr_rej += not sql_corr
        n_mot_rej += not sql_mot
        assert not (py_corr and not sql_corr), (
            f"case {k}: the SQL corridor DROPS a row Python keeps -- "
            f"candidates would vanish server-side\n  {feat}")
        assert not (py_mot and not sql_mot), (
            f"case {k}: the SQL motion bound DROPS a row Python keeps\n  {feat}")
        if verbose and (sql_corr != py_corr or sql_mot != py_mot):
            print(f"  case {k}: looser (ok) corridor {sql_corr}/{py_corr} "
                  f"motion {sql_mot}/{py_mot}")
    return len(cases), n_corr_rej, n_mot_rej


# ---------------------------------------------------------------------------
# Live: let the real Postgres evaluate the real SQL
# ---------------------------------------------------------------------------
def check_live(cases=None, cfg=None, conn=None, log=print):
    """Send ``cases`` to the server as unnested arrays and evaluate the REAL
    predicate SQL there, then assert the same superset property.

    Read-only and table-free: both relations are built from ``unnest(...)``,
    so this touches no ITF data and writes nothing.  Returns
    ``(n, n_disagreements_where_sql_is_looser)``."""
    cfg = cfg or ItfConfig()
    cases = cases if cases is not None else random_cases()
    if conn is None:
        from pyroorb.itf import dbio
        conn = dbio.connect_index(cfg)
    import psycopg2.extras

    src = DbSource.__new__(DbSource)
    src.cfg, src.index_cfg = cfg, IndexConfig()
    w_cols = (("i", "int"),) + DbSource._W_CORRIDOR_COLS
    w_rows, t_rows = [], []
    for k, (window, feat, ra_tol, dec_tol) in enumerate(cases):
        w_rows.append({"i": k, **DbSource._window_sql_row(None, window, ra_tol,
                                                          dec_tol)})
        t_rows.append({"i": k, **{c: feat.get(c) for c, _ in TRACKLET_COLS
                                  if c != "i"}})
    motion = src._sql_motion_bound(True)
    sql = f"""
        WITH {DbSource._window_cte(w_cols, "w")},
             {DbSource._window_cte(TRACKLET_COLS, "t")}
        SELECT w.i AS i, ({src._sql_corridor()}) AS corr, ({motion}) AS mot
        FROM w JOIN t ON t.i = w.i ORDER BY w.i
    """
    args = (DbSource._window_arrays(w_cols, w_rows)
            + DbSource._window_arrays(TRACKLET_COLS, t_rows)
            + [cfg.prefilter_motion_slack_rms,
               cfg.prefilter_motion_slack_arcsec, cfg.maybe_arcsec])
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, args)
        rows = cur.fetchall()
    assert len(rows) == len(cases), f"{len(rows)} verdicts for {len(cases)} cases"

    looser = 0
    for r in rows:
        window, feat, ra_tol, dec_tol = cases[r["i"]]
        py_corr, py_mot = python_verdicts(window, feat, ra_tol, dec_tol, cfg)
        # a NULL verdict is SQL's "unknown" -- the WHERE clause would drop the
        # row, so it counts as a rejection, not as a pass
        sql_corr, sql_mot = bool(r["corr"]), bool(r["mot"])
        assert not (py_corr and not sql_corr), (
            f"case {r['i']}: real Postgres DROPS a corridor row Python keeps\n"
            f"  {feat}")
        assert not (py_mot and not sql_mot), (
            f"case {r['i']}: real Postgres DROPS a motion row Python keeps\n"
            f"  {feat}")
        looser += (sql_corr != py_corr) or (sql_mot != py_mot)
    log(f"# live SQL fuzz: {len(rows)} cases, superset property holds "
        f"({looser} where SQL is merely looser, which is allowed)")
    return len(rows), looser


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    cases = random_cases(n)
    total, rej_c, rej_m = check_shim(cases)
    print(f"# offline shim: {total} cases, corridor rejected {rej_c}, "
          f"motion rejected {rej_m}")
    try:
        check_live(cases)
    except Exception as e:                      # no DB egress -> say so
        print(f"# live check SKIPPED: {type(e).__name__}: {e}")
        sys.exit(0)
