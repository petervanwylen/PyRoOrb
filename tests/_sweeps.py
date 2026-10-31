"""Count integration sweeps (Sum res.nfev) per object, offline from fixtures.

Monkeypatches arc_extend.least_squares to accumulate nfev. Reports per-object
sweeps + used/total/rms so we can see both COST (sweeps) and QUALITY (link).
Usage: python _sweeps.py [substr ...]   (filter objects by label substring)
"""
from pyroorb import obs_sources
import sys, os, time, warnings; warnings.filterwarnings("ignore")
from _fixtures import iter_fixtures, load_fixture
import pyroorb.fit_object as fo; from pyroorb import arc_extend

_orig_ls = arc_extend.least_squares
_counter = {"nfev": 0, "calls": 0}
def _counting_ls(*a, **k):
    result = _orig_ls(*a, **k)
    _counter["nfev"] += int(getattr(result, "nfev", 0) or 0)
    _counter["calls"] += 1
    return result
arc_extend.least_squares = _counting_ls

oc = obs_sources.load_obscodes()
filt = sys.argv[1:]
rows = []
for p in iter_fixtures():
    fx = load_fixture(p); label = fx["label"]
    if filt and not any(f in label for f in filt):
        continue
    _counter["nfev"] = 0; _counter["calls"] = 0
    t0 = time.perf_counter()
    fit = fo.fit_orbit(fx["obs"], oc, desig=fx["meta"]["designation"], meta=fx["meta"], verbose=False)
    dt = time.perf_counter() - t0
    rows.append((label, _counter["nfev"], _counter["calls"], fit.n_observations(),
                 fit.n_total_observations(), fit.rms_residual(), dt))
    print(f"  ..{label:22} {_counter['nfev']:7d} sweeps {dt:6.1f}s", flush=True)

rows.sort(key=lambda r: -r[1])
print(f"{'object':22} {'sweeps':>8} {'calls':>7} {'used/tot':>10} {'rms':>7} {'secs':>7}")
tot_sw = tot_t = 0
for label, sw, calls, used, total, rms, dt in rows:
    tot_sw += sw; tot_t += dt
    print(f"{label:22} {sw:8d} {calls:7d} {used:5d}/{total:<4d} {rms:7.3f} {dt:7.2f}")
print(f"{'TOTAL':22} {tot_sw:8d} {'':7} {'':10} {'':7} {tot_t:7.2f}")
