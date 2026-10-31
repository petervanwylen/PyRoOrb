"""Analyse the _sys_sweep.py output: where the significance statistic lands, and what
a (floor, se-ratio) pair would newly flag on top of today's gate."""
import sys, json, argparse
import numpy as np


def load(p):
    return [r for r in json.load(open(p)) if not r.get("error") and not r.get("skipped")]


def fires_new(r, floor, se_ratio):
    """Rows that clear BOTH the new absolute floor and the significance ratio."""
    return [s for s in r["stats"]
            if s["between"] >= floor and s["ratio_se"] >= se_ratio]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--floor", type=float, default=0.30)
    ap.add_argument("--se", type=float, default=3.0)
    a = ap.parse_args()
    R = load(a.sweep)
    print(f"{len(R)} fixtures fitted "
          f"({sum(1 for r in R if r['label'].startswith('PAIR_'))} pairs)\n")

    allrows = [s for r in R for s in r["stats"]]
    rs = np.array([s["ratio_se"] for s in allrows if np.isfinite(s["ratio_se"])])
    rw = np.array([s["ratio_within"] for s in allrows if np.isfinite(s["ratio_within"])])
    print(f"{len(allrows)} apparition x axis rows")
    for name, arr in (("ratio_se", rs), ("ratio_within", rw)):
        q = np.percentile(arr, [50, 75, 90, 95, 99])
        print(f"  {name:13} median {q[0]:6.2f}  p75 {q[1]:6.2f}  p90 {q[2]:6.2f}  "
              f"p95 {q[3]:6.2f}  p99 {q[4]:6.2f}  max {arr.max():7.2f}")

    cur = [r for r in R if r["current_verdict"]]
    print(f"\ncurrent gate fires on {len(cur)}/{len(R)}:")
    for r in sorted(cur, key=lambda r: r["label"]):
        print(f"   {r['label']:<34} rms {r['rms']:.3f}\"  {r['current_verdict']}")

    print(f"\n--- candidate: between >= {a.floor}\" AND between/SE >= {a.se} ---")
    grid_f = [0.25, 0.30, 0.35, 0.40, 0.50]
    grid_s = [2.5, 3.0, 4.0, 5.0, 6.0]
    print(f"{'floor':>7} " + "".join(f"{('se>=%.1f' % s):>10}" for s in grid_s))
    for f in grid_f:
        row = f"{f:>7.2f} "
        for s in grid_s:
            n = sum(1 for r in R if fires_new(r, f, s) and not r["current_verdict"])
            row += f"{n:>10d}"
        print(row + "   (NEW fires, excl. already-flagged)")

    new = [r for r in R if fires_new(r, a.floor, a.se) and not r["current_verdict"]]
    print(f"\n{len(new)} objects NEWLY flagged at floor={a.floor} se={a.se}:")
    for r in sorted(new, key=lambda r: -max(s["ratio_se"] for s in fires_new(r, a.floor, a.se))):
        w = max(fires_new(r, a.floor, a.se), key=lambda s: s["ratio_se"])
        print(f"   {r['label']:<34} rms {r['rms']:.3f}\" {r['used']}/{r['total']}  "
              f"app{w['app']} {w['axis']:<3} between {w['between']:.2f}\" "
              f"se {w['se_pool']:.3f}\" ratio_se {w['ratio_se']:5.1f} "
              f"(ratio_within {w['ratio_within']:.2f}, {w['n_nights']}n)")


if __name__ == "__main__":
    main()
