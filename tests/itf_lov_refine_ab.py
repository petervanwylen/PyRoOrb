"""Live A/B of the continuous LOV offset refinement (itf/lov_refine.py) over the recovery
fixtures, INTERLEAVED per fixture so a slow database cannot decide the result.

The verdict is ``test_itf_recovery.check_recovery`` -- the suite's OWN
definition of a recovery (apparitions, the per-fixture tracklet golden, the rms
bar and the dropped-seed-obs bar), and the seed apparition it pins.  A private
success criterion here would measure something the suite does not: the first
cut of this script scored ``recovered >= n_fake_tracklets`` and re-derived the
seed, and duly reported a "MISS" on a fixture the suite passes.

The claim under test is a SPEED one as much as a recall one: a variant round
spends its time on trial fits of wrong takes and the recursion under them, so
putting the truth in the first trial of the right round should cost fewer fits,
not just find more.  The refinement's own price (one shared integration plus
~1200 LOV re-corrections, ~78 s on a severe seed) is charged to every run
whether it pays off or not, so the fixtures that already link on an early
variant are the ones that can LOSE -- which is why every fixture runs, not just
the severe ones.

    cd mvp && ../.venv/bin/python tests/itf_lov_refine_ab.py [FIXTURE ...]

Env: REPEATS (default 1).
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

from pyroorb.itf.config import ItfConfig                            # noqa: E402
from pyroorb.itf.candidates import DbSource                         # noqa: E402
from pyroorb.itf.dbio import connect_pub                            # noqa: E402
from test_itf_recovery import check_recovery                # noqa: E402

FIXDIR = "tests/itf_fixtures"
ARMS = (("lov_refine", True), ("base", False))
REPEATS = int(os.environ.get("REPEATS", "1"))


def main(names):
    rows = []
    for name in names:
        path = f"{FIXDIR}/{name}.json"
        for rep_i in range(REPEATS):
            for arm, on in ARMS:
                cfg = ItfConfig(lov_refine=on)
                try:
                    src = DbSource(connect_pub(cfg), cfg,
                                   reconnect=lambda c=cfg: connect_pub(c))
                except Exception as e:                       # noqa: BLE001
                    print(f"{name:>12} {arm:>5}: DB unreachable: {e}")
                    return
                t0 = time.time()
                try:
                    ok, msg = check_recovery(path, src, cfg_overrides={
                        "lov_refine": on})
                except Exception as e:                       # noqa: BLE001
                    ok, msg = False, f"{type(e).__name__}: {e}"
                el = time.time() - t0
                rows.append((name, rep_i, arm, ok, el))
                print(f"{name:>12} run{rep_i} {arm:>5}: "
                      f"{'OK  ' if ok else 'MISS'} {el:7.1f}s  {msg[:88]}",
                      flush=True)
                try:
                    src.conn.close()
                except Exception:                            # noqa: BLE001
                    pass
    print("\n=== summary (best of repeats) ===")
    print(f"{'fixture':>12} {'lov_refine':>10} {'base':>9} {'delta':>9}  verdicts")
    tot = {a: 0.0 for a, _ in ARMS}
    n_ok = {a: 0 for a, _ in ARMS}
    for name in names:
        cell = {}
        for arm, _ in ARMS:
            r = [x for x in rows if x[0] == name and x[2] == arm]
            if r:
                cell[arm] = (min(x[4] for x in r), any(x[3] for x in r))
        if len(cell) < 2:
            continue
        a, b = cell["lov_refine"], cell["base"]
        tot["lov_refine"] += a[0]
        tot["base"] += b[0]
        n_ok["lov_refine"] += bool(a[1])
        n_ok["base"] += bool(b[1])
        flag = "" if a[1] == b[1] else ("  <== RECALL GAIN" if a[1]
                                        else "  <== RECALL LOSS")
        print(f"{name:>12} {a[0]:8.1f}s {b[0]:8.1f}s {a[0] - b[0]:+8.1f}s  "
              f"lov_refine={'OK' if a[1] else 'MISS'} "
              f"base={'OK' if b[1] else 'MISS'}{flag}")
    print(f"{'TOTAL':>12} {tot['lov_refine']:8.1f}s {tot['base']:8.1f}s "
          f"{tot['lov_refine'] - tot['base']:+8.1f}s   "
          f"recovered lov_refine={n_ok['lov_refine']} base={n_ok['base']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted(f[:-5] for f in os.listdir(FIXDIR) if f.endswith(".json"))
    main(args)
