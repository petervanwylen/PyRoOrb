"""The search's DECISIONS, recorded for ranker training and evaluation.

Every earlier collection recorded candidate rows grouped by corridor WINDOW,
from a search that ran differently from the served one (the collector bypassed
the motion pre-filter and kept 'unlikely' candidates, so its trial queues saw a
different field).  But the ranker's P is consumed in units the window does not
describe:

* a ROUND (one ``_iterate`` call: one trial orbit's corridor, its trial queue of
  trials) -- the trial queue orders the round's candidates by P;
* a variant SCAN -- the LOV profile picks its priority variants from the global
  top of P across every variant swept so far;
* the TAKES -- which trials were fitted, in what order, and what each cost.

So a :class:`DecisionLog` rides on the search context and records exactly
that: every round (depth, whether its corridor is a LOV variant's, the variant
k, the trial orbit's U), every trial the trial queue tried with its outcome and fit
time, every barren resume, and the variant visit order.  The candidate rows
the collector records carry the round and scan ids, so a row is placed in the
decision it competed in.  Recording is PASSIVE: nothing here is read back by
the search, and a search with a log decides exactly as one without.
"""

from __future__ import annotations

import time


class DecisionLog:
    """Round/trial/scan events for one search.  Shared by reference through
    every ``dataclasses.replace`` of the search context."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.events = []
        self.n_rounds = 0
        self.n_scans = 0
        self.cur_round = -1
        self.cur_scan = -1
        # set by the variant scan around each depth-0 variant round
        self.variant_lov_sigma = None

    def now(self):
        return round(time.perf_counter() - self.t0, 3)

    def begin_scan(self, n_variants, U):
        self.n_scans += 1
        self.cur_scan = self.n_scans
        self.events.append({"ev": "scan", "scan": self.cur_scan,
                            "n_variants": int(n_variants),
                            "U": _r(U), "t": self.now()})
        return self.cur_scan

    def end_scan(self):
        self.cur_scan = -1
        self.variant_lov_sigma = None

    def begin_round(self, label, depth, in_variant_corridor, eph_trial):
        self.n_rounds += 1
        self.cur_round = self.n_rounds
        self.events.append({
            "ev": "round", "round": self.cur_round, "label": label,
            "depth": int(depth), "scan": self.cur_scan,
            "in_variant_corridor": bool(in_variant_corridor),
            "variant_k": (_r(self.variant_lov_sigma) if in_variant_corridor
                          else None),
            "U": _r(getattr(eph_trial, "U", None)),
            "used": getattr(eph_trial, "used", None),
            "t": self.now()})
        return self.cur_round

    def round_size(self, rid, n_all, n_likely):
        self.events.append({"ev": "size", "round": rid, "n_all": int(n_all),
                            "n_likely": int(n_likely)})

    def trial(self, rid, trksubs, outcome, fit_sec=None, rms=None, used=None):
        self.events.append({"ev": "rung", "round": rid,
                            "trksubs": list(trksubs), "outcome": outcome,
                            "fit_sec": _r(fit_sec), "rms": _r(rms),
                            "used": used, "t": self.now()})

    def barren(self, rid, trksubs):
        self.events.append({"ev": "barren", "round": rid,
                            "trksubs": list(trksubs), "t": self.now()})

    def note(self, **kw):
        kw.setdefault("t", self.now())
        self.events.append(kw)


def _r(v, nd=4):
    try:
        return None if v is None else round(float(v), nd)
    except (TypeError, ValueError):
        return None


def trksubs_of(tracklets):
    """The serving trksub of each candidate tracklet dict (its first row's)."""
    out = []
    for t in tracklets:
        rows = t.get("rows") or ()
        out.append(rows[0].get("trksub") if rows else None)
    return out
