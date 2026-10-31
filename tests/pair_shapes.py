"""Dump the night-level SHAPE and ident verdict of every PAIR_ fixture -- offline, read-only.

The calibration companion to the ident rule (`ident_check`): one block per pair giving each
designation's retention and kept/full arc, then every opposition's nights with obs count,
keep/DROP/partial state and residual range, followed by identify() / examine_or_no(). Because
the whole ident rule is a judgement about the SHAPE of the drops, this is what a threshold
change has to be checked against: re-run it before and after and diff the verdict headers --
a tuning meant to catch two pairs must not quietly move the other ninety.

    python tests/pair_shapes.py                 # all PAIR_ fixtures
    python tests/pair_shapes.py OF201 XO232     # only labels/designations matching these

Writes nothing (no fixture is ever touched); see testing-framework.md for the three scripts
that do own fixtures."""
import sys, os, glob, math, warnings; warnings.filterwarnings("ignore")
import numpy as np
from _fixtures import load_fixture, FIX
import pyroorb.fit_object as fo; from pyroorb import obs_sources; from pyroorb import ident_check
from pyroorb.orbit_common import NIGHT_GAP_DAYS, OPP_GAP_DAYS
OC = obs_sources.load_obscodes()
labels = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(FIX, "PAIR_*.json")))
if len(sys.argv) > 1:
    labels = [l for l in labels if any(a in l for a in sys.argv[1:])]
for lbl in labels:
    fx = load_fixture(os.path.join(FIX, lbl + ".json"))
    obs = [dict(o) for o in fx["obs"]]
    for o in obs:
        o["obs80_line"] = f"     {o.get('packed','XXXXXXX'):7} C2020 01 01.00000 ...{o['obscode']}"
    try:
        fit = fo.fit_orbit(obs, OC, desig=fx["meta"]["designation"], meta=fx["meta"])
    except Exception as ex:
        print(f"### {lbl}: FIT ERROR {ex}"); continue
    mask = np.asarray(fit._mask, dtype=bool); R = fit._resid_all
    times = [o["jd_utc"] for o in fit._obs]
    ident = ident_check.identify(fit)
    print(f"### {lbl} rms {fit.rms_residual():.2f}\" {fit.n_observations()}/{fit.n_total_observations()}"
          f"  ident={'YES' if ident else 'None'}  {ident_check.examine_or_no(fit)}")
    for d in sorted({o.get('designation') for o in fit._obs}):
        idx = [i for i, o in enumerate(fit._obs) if o.get('designation') == d]
        kept = [i for i in idx if mask[i]]; kt = [times[i] for i in kept]
        print(f"  - {d}: {len(kept)}/{len(idx)} ({len(kept)/len(idx)*100:.0f}%) full "
              f"{max(times[i] for i in idx)-min(times[i] for i in idx):.2f} d kept "
              f"{(max(kt)-min(kt)) if kt else 0:.2f} d")
    apps = ident_check._split(times, OPP_GAP_DAYS)
    print(f"  opps={len(apps)}")
    for a, app in enumerate(apps):
        nights = ident_check._split([times[i] for i in app], NIGHT_GAP_DAYS)
        parts = []
        for ni in nights:
            gi = [app[k] for k in ni]
            ne = sum(1 for i in gi if not mask[i])
            rs = [math.hypot(R[i,0], R[i,1]) for i in gi] if R is not None else [0.0]
            tag = "DROP" if ne == len(gi) else (f"p{ne}" if ne else "keep")
            parts.append(f"[{times[gi[0]]-times[app[0]]:+.1f}d n{len(gi)} {tag} r{min(rs):.1f}-{max(rs):.1f}\"]")
        print(f"   opp{a} {' '.join(parts)}")
