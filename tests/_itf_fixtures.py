"""Shared helpers for the ITF **recovery** fixtures.

An ITF fixture freezes one object's PUBLISHED astrometry (verbatim OBS80) so the
harness seed is deterministic and needs no network -- exactly like the main
`fixtures/` archive, but carrying the raw 80-col lines (the ITF harness is
line-based) instead of parsed obs dicts.  At test time the frozen astrometry is
re-parsed through the SAME path a live fetch uses (`read_obs_file`), the harness
plants the object's other apparitions as fake ITF, and -- when the SBN database
is reachable -- the live ITF supplies REAL tracklets as distractors beside them.

`build_itf_fixtures.py` writes these; `test_itf_recovery.py` reads them.
"""
import glob
import json
import os

ITF_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "itf_fixtures")


def freeze_obs80(obs):
    """Concatenate an obs list's verbatim OBS80 lines (incl. the 2nd line of a
    two-line record) into the frozen-astrometry blob."""
    out = []
    for o in obs:
        if o.get("obs80_line"):
            out.append(o["obs80_line"])
            if o.get("obs80_line2"):
                out.append(o["obs80_line2"])
    return "\n".join(out) + "\n"


def freeze_ades(obs):
    """The ADES-only fields the ITF registry carries (``itf.obsreg._CARRIED``:
    the Veres-2017 sigma, reported rms, mode, band, ...), keyed by each obs's
    verbatim OBS80 line.  The 80-column text cannot hold them, and without them
    a re-parse weights the observations differently from the live ADES fetch
    the search actually fits (2008 GU20: SDSS at 0.7" and a 673 night at 0.3"
    became the OBS80 fallbacks, and the recovery changed)."""
    from pyroorb.itf.obsreg import _CARRIED
    out = {}
    for o in obs:
        line = o.get("obs80_line")
        extras = {k: o[k] for k in _CARRIED if o.get(k) is not None}
        if line and extras:
            out[line] = extras
    return out


def load_obs(fx, obscodes):
    """Reconstruct ``(obs, meta)`` from a fixture's frozen OBS80 -- faithful to
    ``fetch_object_obs`` (same parse/debias path), no network.  A fixture that
    froze its ADES fields too (``"ades"``, see :func:`freeze_ades`) gets them
    back, so the search weights each observation as it does live; older
    fixtures without them re-parse exactly as before."""
    from pyroorb.obs_sources import read_obs_file
    obs, _fmt = read_obs_file(fx["obs80"], obscodes, in_format="obs80")
    obs = [o for o in obs if o.get("obs80_line")]
    ades = fx.get("ades") or {}
    for o in obs:
        extras = ades.get(o["obs80_line"])
        if extras:
            o.update(extras)
    obs.sort(key=lambda o: o["jd_utc"])
    return obs, {"designation": fx["designation"]}


# Fields a REBUILD must not silently drop: they are decisions about the fixture,
# not measurements of the object, so they cannot be re-derived from the astrometry
# the way seed_U or n_fake_tracklets can.  Without this, re-running
# build_itf_fixtures.py on a marked fixture quietly un-marks it and the flap comes
# back with nobody having chosen it.
_STICKY_FIELDS = ("offline_only",)


def write_fixture(path, payload):
    from _fixtures import write_fixture as _atomic_write   # reuse the retrying writer
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            prev = load_fixture(path)
        except Exception:
            prev = {}
        for k in _STICKY_FIELDS:
            if k in prev and k not in payload:
                payload = {**payload, k: prev[k]}
    _atomic_write(path, payload)


def load_fixture(path):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def iter_fixtures():
    return sorted(glob.glob(os.path.join(ITF_FIX, "*.json")))
