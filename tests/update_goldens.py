"""RE-BASELINE the regression goldens -- the USUAL way to update fixtures (only the
golden block is rewritten; the stored obs/meta are untouched). Run it deliberately,
ONLY after test_regression.py has confirmed a change is an improvement, not a drift.

It re-fits each fixture's STORED obs (offline, no re-fetch, deterministic), so the
diff isolates the engine change. By default it is a DRY RUN: it prints the
golden -> current delta for every object and writes nothing. Pass --write to commit
the new baseline. Shares its golden/atomic-write logic with build_fixtures.py via
_fixtures.py, so the two can never disagree on how a baseline is computed.

    python tests/update_goldens.py             # dry run: show drift, write nothing
    python tests/update_goldens.py --write     # actually update the goldens
    python tests/update_goldens.py --repeat 2  # time each fixture twice, record the MIN

`--repeat N` re-fits each fixture N times and records the SMALLEST CPU time. A fit is
deterministic, so every repeat must produce the same orbit -- the run says so if one does
not, which is a finding in itself. What varies is the MEASUREMENT: a box shared with other
work can inflate CPU seconds by ~1.7x on identical work, and a baseline recorded during a
transient is a baseline every later run fails against. The minimum is the least-contended
sample, which is the honest estimate of what the fit costs.

(Run from the repo root, like the suite. It is a script, not a pytest module -- it
WRITES fixtures, so it must never be something a test run can trigger.)

(To add NEW objects or refresh the obs snapshot itself -- which needs the network --
use build_fixtures.py instead.)
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
from _fixtures import (compute_golden, iter_fixtures, load_fixture, nongrav_golden,
                       write_fixture)
from test_regression import refit


def _timed_refit(fx, repeat):
    """Re-fit `repeat` times; return (fit, min CPU seconds, drift note or None)."""
    best_fit, best_secs, drift = None, float("inf"), None
    for _ in range(max(1, repeat)):
        fit, secs = refit(fx)
        if best_fit is not None and (fit.n_observations() != best_fit.n_observations()
                                     or abs(fit.rms_residual() - best_fit.rms_residual()) > 1e-9):
            drift = (f"NONDETERMINISTIC: {best_fit.n_observations()}@{best_fit.rms_residual():.4f}"
                     f" vs {fit.n_observations()}@{fit.rms_residual():.4f}")
        if secs < best_secs:
            best_fit, best_secs = fit, secs
        elif best_fit is None:
            best_fit = fit
    return best_fit, best_secs, drift


def main():
    write = "--write" in sys.argv
    # BACKFILL mode: record only the force-model keys (`nongrav` / `nongrav_signs`) and leave
    # used / rms / status / secs exactly as they are. The nongrav guard was added to goldens that
    # already existed, and a full --write re-baseline to pick it up would also re-record every
    # TIMING baseline -- which must be measured serially on a quiet box (see test_regression's
    # docstring), a much stronger precondition than "tell me which model each object delivers".
    # So the two are separable: --nongrav arms the new guard on the existing baselines, and a
    # normal --write re-baselines everything as before.
    nongrav_only = "--nongrav" in sys.argv
    repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1
    tot_old = tot_new = 0.0
    changed = 0
    rows = []
    for p in iter_fixtures():
        fx = load_fixture(p); old = fx["golden"]
        # refit() (test_regression) owns BOTH the fit call and the timing currency --
        # CPU seconds -- so a re-baseline always records what the guard measures.
        fit, secs, drift = _timed_refit(fx, repeat)
        new = compute_golden(fit, secs)
        tot_old += old.get("secs", 0); tot_new += secs
        diffs = []
        if drift:
            diffs.append(drift)
        if new["used"] != old["used"]:    diffs.append(f"used {old['used']}->{new['used']}")
        if new["rms"] != old["rms"]:      diffs.append(f"rms {old['rms']}->{new['rms']}")
        if new["status"] != old["status"]: diffs.append(f"status {old['status']}->{new['status']}")
        if abs(new["secs"] - old.get("secs", 0)) >= 1.0:
            diffs.append(f"{old.get('secs')}s->{new['secs']:.1f}s")
        # The force model, if this fixture has ever been baselined with one. A flip is always
        # worth a line: it is a hard fail in the suite and there is no tolerance to absorb it.
        if "nongrav" in old and new["nongrav"] != old["nongrav"]:
            diffs.append(f"nongrav {old['nongrav']}->{new['nongrav']}")
        if new["used"] != old["used"] or new["rms"] != old["rms"] or new["status"] != old["status"] \
                or ("nongrav" in old and new["nongrav"] != old["nongrav"]):
            changed += 1
        rows.append((fx["label"], new, diffs))
        if write:
            fx["golden"] = {**old, **nongrav_golden(fit)} if nongrav_only else new
            write_fixture(p, fx)
        flag = ("  <<< " + "; ".join(diffs)) if diffs else ""
        print(f"  {fx['label']:20} {new['used']}/{new['total']} @ {new['rms']}\" "
              f"{new['status']} [{new['secs']:.1f}s]{flag}", flush=True)
    print(f"\n{len(rows)} objects | result-changed: {changed} | "
          f"suite CPU time {tot_old:.0f}s -> {tot_new:.0f}s")
    if write and nongrav_only:
        print("WROTE the nongrav keys only (used/rms/status/secs left untouched).")
    else:
        print("WROTE new goldens." if write else
              "DRY RUN -- nothing written (pass --write to update, --nongrav to backfill "
              "only the force-model keys).")


if __name__ == "__main__":
    main()
