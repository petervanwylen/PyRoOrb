"""The ONE Earth-orientation (EOP) table every ITRS<->GCRS rotation uses.

Why a pinned table at all: left to itself astropy's default ``IERS_Auto`` re-checks
(and, in a sandbox that cannot download, re-parses) the IERS file on EVERY
transform -- ~1 s per observation, which once dominated runtime. So one table is
built once per process and set as ``iers.earth_orientation_table``; auto-download
stays off and runtime never touches the network.

Why THIS table: the pin used to be the bundled IERS-B (EOP 20 C04) alone, which is
FINAL values only and ends ~5 months before the ``astropy-iers-data`` release it
ships in (astropy-iers-data 0.2026.6.8: 2026-05-02). Every observation after that
-- every NEOCP object, every recent tracklet, every future ephemeris -- was rotated
with UT1-UTC = 0 and the 50-year-mean pole, and astropy said so twice per batch
in the web server log. The same package also ships ``finals2000A.all`` (IERS
Bulletin A: rapid values, then predictions ~1 year past the release; 2027-06-05
for 0.2026.6.8). So the table is IERS-B for its whole span -- byte-identical to
the old pin there, so no pre-2026-05 fit moves at all -- continued by Bulletin A
from the day after B ends.

Measured (10 stations, Sep-Nov 2026): the old fallback displaced the
observer by 17-25 m (UT1-UTC ~ +0.04..+0.06 s = 0.6-0.9" of Earth rotation, plus
~0.15" of pole). As astrometry that is 2e-5" for a main-belt object, 7e-4" at
0.05 au, 0.003" at 0.01 au and 0.017" at 0.002 au -- far below any astrometric
sigma, and fits of 2026 RA / SR / RU move by <1e-5" rms. It could grow (|UT1-UTC|
<= 0.9 s by UTC's definition: 420 m, 0.29" at 0.002 au), and 20 m is on the scale
of a good radar delay, so the table is extended rather than the warning ignored.

Past the table's end (or before 1962) astropy still falls back to UT1-UTC = 0
and the mean pole. Those two per-transform warnings are silenced here on
purpose -- the bound above is the reason -- and replaced by ONE log line at pin
time when the table ends within ``STALE_WARN_DAYS`` of today: the fix then is
``uv pip install -U astropy-iers-data`` (a redeploy picks up the latest), never a
runtime download.
"""
import logging
import warnings

log = logging.getLogger(__name__)

STALE_WARN_DAYS = 30.0      # log once when the predictions run out this soon (or already have)

# The two astropy messages the fallback emits (astropy/coordinates/builtin_frames/utils.py):
# polar motion "before"/"after" the table, and UT1-UTC via IERSRangeError. Matched by
# message start so an unrelated AstropyWarning is never swallowed.
_SILENCED = (
    r"Tried to get polar motions for times (before|after) IERS data is valid",
    r"\(some\) times are outside of range covered by IERS table",
)

_PINNED = None


def build_table():
    """IERS-B (final) through its last day, then Bulletin A (rapid + predicted), as one
    ``iers.IERS_A`` instance. Both come from the installed ``astropy-iers-data``; no
    network. ~0.15 s, once."""
    import numpy as np
    import astropy.units as u
    from astropy.table import Table
    from astropy.utils import iers

    b = iers.IERS_B.open()
    a = iers.IERS_A.open(iers.IERS_A_FILE)
    a = a[a["MJD"] > b["MJD"][-1]]
    nb = len(b)
    cols = {
        "MJD": np.concatenate([b["MJD"].to_value(u.d), a["MJD"].to_value(u.d)]) * u.d,
        "UT1_UTC": np.concatenate([b["UT1_UTC"].to_value(u.s), a["UT1_UTC"].to_value(u.s)]) * u.s,
    }
    for c in ("PM_x", "PM_y", "dX_2000A", "dY_2000A"):
        cols[c] = np.concatenate([b[c].to_value(u.arcsec), a[c].to_value(u.arcsec)]) * u.arcsec
    for c in ("UT1Flag", "PolPMFlag", "NutFlag"):
        cols[c] = np.concatenate([np.full(nb, "B"), np.asarray(a[c], dtype="U1")])
    t = iers.IERS_A(Table(cols))
    p = np.flatnonzero(t["UT1Flag"] == "P")
    t.meta["predictive_index"] = int(p[0]) if len(p) else len(t) - 1   # IERS_A.read's meta
    t.meta["predictive_mjd"] = float(t["MJD"][t.meta["predictive_index"]].value)
    t.meta["b_end_mjd"] = float(b["MJD"][-1].value)
    return t


def pin():
    """Build (once) and install the combined table; idempotent and never raises. Returns
    the table, or None when it could not be built (astropy then keeps its default)."""
    global _PINNED
    if _PINNED is not None:
        return _PINNED
    silence_fallback_warnings()
    try:
        from astropy.utils import iers
        iers.conf.auto_download = False
        try:
            table = build_table()
        except Exception as exc:                  # a data package without finals2000A
            log.warning("EOP: combined IERS table unavailable (%s); pinning IERS-B alone", exc)
            table = iers.IERS_B.open()
        iers.earth_orientation_table.set(table)
        _PINNED = table
        _stale_check(table)
    except Exception as exc:
        log.warning("EOP: could not pin an Earth-orientation table (%s)", exc)
        return None
    return _PINNED


def silence_fallback_warnings():
    """Install the two message-matched ignore filters (see the module docstring for the bound
    that justifies them). Idempotent in effect; separate so a test can re-apply them inside
    its own ``catch_warnings`` block."""
    for msg in _SILENCED:
        warnings.filterwarnings("ignore", message=msg)


def table_end_mjd(table=None):
    """Last MJD the pinned table covers (predictions included)."""
    t = table if table is not None else pin()
    return float(t["MJD"][-1].value)


def _stale_check(table):
    from astropy.time import Time
    end = table_end_mjd(table)
    left = end - Time.now().mjd
    if left < STALE_WARN_DAYS:
        log.warning("EOP: Earth-orientation table ends %s (%.0f days from now); later epochs "
                    "use UT1-UTC=0 and the mean pole (<= 0.3\" at 0.002 au, far less beyond). "
                    "Upgrade astropy-iers-data to extend it.",
                    Time(end, format="mjd").iso[:10], left)
