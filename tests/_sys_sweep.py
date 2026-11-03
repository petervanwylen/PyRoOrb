"""Measurement sweep for the systematic-misfit detector (scratch tooling, not a test).

Refits every frozen fixture OFFLINE and records, per apparition and axis, both the
CURRENT gate's statistics (between-night RMS vs raw within-night scatter) and the
proposed SIGNIFICANCE statistic (between-night RMS vs the standard ERROR of the
per-night means). Writing both for every object is what lets a threshold be chosen
from the observed distribution instead of guessed.

    python _sys_sweep.py [-j N] [-o OUT.json] [--only SUBSTR]
"""
import sys, os, json, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from _fixtures import load_fixture, iter_fixtures

_OC = None


def _obscodes():
    global _OC
    if _OC is None:
        from pyroorb import obs_sources
        _OC = obs_sources.load_obscodes()
    return _OC


def stats_for(fit):
    """Per apparition+axis stats, mirroring systematic_misfit's own splitting exactly
    (same _split calls, same qualifying-night rule) so the numbers are comparable to
    what the live gate sees -- including its leave-one-out robustness step."""
    import pyroorb.ident_check as ic
    if fit._resid_all is None:
        return []
    m = np.asarray(fit._mask, dtype=bool)
    idx = [i for i in range(len(fit._obs)) if m[i]]
    if len(idx) < 6:
        return []
    idx.sort(key=lambda i: fit._obs[i]["jd_utc"])
    times = [fit._obs[i]["jd_utc"] for i in idx]
    R = fit._resid_all
    out = []
    for a, app in enumerate(ic._split(times, ic.OPP_GAP_DAYS)):
        nights = [ni for ni in ic._split([times[k] for k in app], ic.NIGHT_GAP_DAYS)
                  if len(ni) >= ic.SYS_MIN_NIGHT_OBS]
        if len(nights) < ic.SYS_MIN_NIGHTS:
            continue
        for ax in (0, 1):
            vals = [np.array([R[idx[app[k]], ax] for k in ni]) for ni in nights]
            means = np.array([v.mean() for v in vals])
            within = float(np.mean([v.std() for v in vals]))
            # standard error of each night's MEAN -- the noise the mean actually carries
            ses = np.array([(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
                            for v in vals])
            between = float(np.sqrt(np.mean(means ** 2)))
            if len(means) >= ic.SYS_LOO_MIN_NIGHTS:      # same LOO step as the live gate
                drop = int(np.argmax(np.abs(means)))
                between = float(np.sqrt(np.mean(np.delete(means, drop) ** 2)))
                se_pool = float(np.sqrt(np.nanmean(np.delete(ses, drop) ** 2)))
            else:
                se_pool = float(np.sqrt(np.nanmean(ses ** 2)))
            out.append({
                "app": a, "axis": "RA" if ax == 0 else "Dec",
                "n_nights": len(nights), "n_obs": [int(len(v)) for v in vals],
                "between": between, "within": within, "se_pool": se_pool,
                "ratio_within": between / within if within > 1e-6 else float("inf"),
                "ratio_se": between / se_pool if se_pool and se_pool > 1e-9 else float("inf"),
            })
    return out


def one(path):
    import pyroorb.fit_object as fo
    import pyroorb.ident_check as ic
    label = os.path.splitext(os.path.basename(path))[0]
    try:
        fx = load_fixture(path)
        label = fx["label"]
        if fx.get("benchmark", False):
            return {"label": label, "skipped": "benchmark"}
        t0 = time.time()
        fit = fo.fit_orbit(fx["obs"], _obscodes(), desig=fx["meta"]["designation"],
                           meta=fx["meta"], verbose=False, force=fx.get("force", False))
        return {"label": label, "secs": round(time.time() - t0, 1),
                "rms": round(fit.rms_residual(), 4),
                "used": fit.n_observations(), "total": fit.n_total_observations(),
                "n_opp": fit.n_oppositions() if hasattr(fit, "n_oppositions") else None,
                "current_verdict": ic.systematic_misfit(fit),
                "stats": stats_for(fit)}
    except Exception as ex:
        return {"label": label, "error": f"{type(ex).__name__}: {ex}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("-o", default="sys_sweep.json")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    paths = [p for p in iter_fixtures() if not a.only or a.only in p]
    print(f"{len(paths)} fixtures, {a.j} workers", flush=True)
    result, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.j) as ex:
        for i, r in enumerate(ex.map(one, paths), 1):
            result.append(r)
            tag = "ERR " if r.get("error") else ("skip" if r.get("skipped") else "ok  ")
            print(f"[{i}/{len(paths)}] {tag} {r['label']}"
                  + (f"  {r.get('error','')}" if r.get("error") else ""), flush=True)
    with open(a.o, "w", encoding="utf-8") as fp:
        json.dump(result, fp)
    print(f"wrote {a.o} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
