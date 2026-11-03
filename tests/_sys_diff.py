"""Diff two _sys_sweep.py runs: which objects changed systematic-misfit verdict."""
import sys, json

old = {r["label"]: r for r in json.load(open(sys.argv[1], encoding="utf-8"))}
new = {r["label"]: r for r in json.load(open(sys.argv[2], encoding="utf-8"))}
gained, lost, kept = [], [], []
for lab, n in new.items():
    o = old.get(lab)
    if o is None or n.get("error") or o.get("error") or n.get("skipped"):
        continue
    a, b = o.get("current_verdict"), n.get("current_verdict")
    if not a and b:
        gained.append((lab, n, b))
    elif a and not b:
        lost.append((lab, a))
    elif a and b:
        kept.append((lab, a, b))
    # sanity: the fit itself must not have moved
    if abs(o["rms"] - n["rms"]) > 1e-4 or o["used"] != n["used"]:
        print(f"!! FIT MOVED {lab}: rms {o['rms']}->{n['rms']} used {o['used']}->{n['used']}")

print(f"{len(new)} objects | verdict GAINED {len(gained)} | LOST {len(lost)} | kept {len(kept)}\n")
if lost:
    print("LOST (must be empty -- the change is OR-ed and can only add):")
    for lab, a in lost:
        print(f"   {lab}: {a}")
print("\nGAINED:")
for lab, n, b in sorted(gained):
    print(f"   {lab:<34} rms {n['rms']:.3f}\" {n['used']}/{n['total']}  {b}")
print("\nkept (text may differ -- the SE multiple is now appended):")
for lab, a, b in sorted(kept):
    if a != b:
        print(f"   {lab:<34} {a}\n{'':<37}-> {b}")
