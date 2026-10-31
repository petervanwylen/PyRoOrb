"""Per-object sweep breakdown by _fit_from_start call (phase), offline.
Usage: python _phase_sweeps.py TE4 VM6 VD32 ...
Marks each _fit_from_start invocation with its (extend,start) so we can see
which phase (initial bridge / drop-screen / sigma_a probe / re-link) burns sweeps.
"""
from pyroorb import obs_sources
import sys, os, warnings; warnings.filterwarnings("ignore")
from _fixtures import iter_fixtures, load_fixture
import pyroorb.fit_object as fo; from pyroorb import arc_extend

orig_ls = arc_extend.least_squares
ph = {"name": "?", "tot": {}, "order": []}
def cls(*a, **k):
    r = orig_ls(*a, **k)
    n = ph["name"]
    if n not in ph["tot"]:
        ph["order"].append(n)
    ph["tot"][n] = ph["tot"].get(n, 0) + int(getattr(r, "nfev", 0) or 0)
    return r
arc_extend.least_squares = cls

orig_ffs = fo._fit_from_start
cnt = {"n": 0}
def wrapped(*a, **k):
    cnt["n"] += 1
    ext = k.get("extend", True); st = k.get("start", None); act = k.get("activate", True)
    ph["name"] = f"#{cnt['n']:02d} ext={int(ext)} act={int(act)} start={st}"
    return orig_ffs(*a, **k)
fo._fit_from_start = wrapped

oc = obs_sources.load_obscodes()
filt = sys.argv[1:]
for p in iter_fixtures():
    fx = load_fixture(p); label = fx["label"]
    if filt and not any(f in label for f in filt):
        continue
    ph["tot"].clear(); ph["order"].clear(); cnt["n"] = 0
    fit = fo.fit_orbit(fx["obs"], oc, desig=fx["meta"]["designation"], meta=fx["meta"], verbose=False)
    print(f"\n=== {label}: used {fit.n_observations()}/{fit.n_total_observations()} "
          f"rms={fit.rms_residual():.3f}  total_sweeps={sum(ph['tot'].values())} ===")
    for n in ph["order"]:
        print(f"  {ph['tot'][n]:6d}  {n}")
