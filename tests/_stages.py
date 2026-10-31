"""Profile the 48-object suite by PIPELINE STAGE (offline, from fixtures)."""
import sys, os, time, warnings; warnings.filterwarnings("ignore")
from _fixtures import iter_fixtures, load_fixture
import numpy as np; import pyroorb.fit_object as fo; from pyroorb import arc_extend
oc = fo.load_obscodes()
STAGES = ["iod", "subarc_extend", "linkage", "reject_drop", "final_fit", "products"]
rows = []
for p in iter_fixtures():
    fx = load_fixture(p); label = fx["label"]
    arc_extend.STAGE_TIMES.clear()
    t0 = time.perf_counter()
    fo.fit_orbit(fx["obs"], oc, desig=fx["meta"]["designation"], meta=fx["meta"], verbose=False)
    tot = time.perf_counter() - t0
    st = dict(arc_extend.STAGE_TIMES); st["TOTAL"] = tot
    st["other"] = tot - sum(st.get(s, 0) for s in STAGES)
    rows.append((label, st))
# aggregate totals
print(f"{'STAGE':14} {'total_s':>9} {'%':>6} {'mean_s':>8} {'max_s':>8} {'max_obj':>18}")
grand = sum(r[1]["TOTAL"] for r in rows)
for s in STAGES + ["other"]:
    vals = [(r[1].get(s, 0.0), r[0]) for r in rows]
    tot = sum(v for v, _ in vals); mx, mxobj = max(vals)
    cv = (np.std([v for v,_ in vals]) / (np.mean([v for v,_ in vals]) + 1e-9))
    print(f"{s:14} {tot:9.1f} {100*tot/grand:5.1f}% {tot/len(rows):8.3f} {mx:8.2f} {mxobj:>18}  CV={cv:.2f}")
print(f"{'TOTAL':14} {grand:9.1f}")
