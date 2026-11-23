"""A far bridge that failed is not run again within one fit for the same input.

The recovery stage re-anchors on a short sub-arc of the apparition the regular pass seeded
from; that anchor extends to the same apparition fit and asks the same multi-revolution
bridge, which fails again at the same cost (on a two-apparition NEO 28 years apart, about
35 s of a 167 s identification fit). A failed bridge returns the anchor state unchanged, so
the record may answer a repeat only when the input is the same: observations, weights,
linked set and target apparition (the key), and the anchor solution's rms and offset at the
far apparition to ``BRIDGE_REPEAT_RTOL``.  Over the regression fixtures it never answers.
"""
from types import SimpleNamespace

import numpy as np

import pyroorb.arc_extend as ae
import pyroorb.fit_object as fo


def _ev(n=10, sig=0.5):
    t = np.linspace(2450000.0, 2461000.0, n)
    return SimpleNamespace(times_tdb=t, ra=np.linspace(10, 20, n), dec=np.linspace(-5, 5, n),
                           sig_ra=np.full(n, sig), sig_dec=np.full(n, sig),
                           fit_sig_ra=np.full(n, sig), fit_sig_dec=np.full(n, sig),
                           nongrav=None, ident_mode=True, n=n)


def _masks(n=10):
    mask = np.zeros(n, bool); mask[5:] = True
    nbm = np.zeros(n, bool); nbm[:5] = True
    return mask, nbm


def test_no_record_outside_a_fit():
    mask, nbm = _masks()
    assert ae._bridge_key(_ev(), mask, nbm) is None
    assert not ae._bridge_failed_before(None, 1.0, 1.0)


def test_repeat_answered_only_for_the_same_input():
    @ae.bridge_failure_scope
    def body():
        ev = _ev()
        mask, nbm = _masks()
        key = ae._bridge_key(ev, mask, nbm)
        assert key is not None
        ae._note_bridge_failure(key, 469679.0, 0.379)
        assert ae._bridge_failed_before(key, 469671.6, 0.379)        # same solution, other epoch
        assert not ae._bridge_failed_before(key, 469679.0, 0.40)     # another anchor solution
        assert not ae._bridge_failed_before(key, 400000.0, 0.379)    # another offset
        mask2 = mask.copy(); mask2[4] = True
        assert ae._bridge_key(ev, mask2, nbm) != key                 # another linked set
        assert ae._bridge_key(_ev(sig=0.8), mask, nbm) != key         # relaxed weights
        ev_ng = _ev(); ev_ng.nongrav = object()
        assert ae._bridge_key(ev_ng, mask, nbm) is None              # nongrav: never recorded
        return key
    key = body()
    assert ae._BRIDGE_FAILURES.get() is None                          # scope ends with the call
    assert not ae._bridge_failed_before(key, 469679.0, 0.379)


def test_nested_fits_keep_their_own_record():
    mask, nbm = _masks()

    @ae.bridge_failure_scope
    def inner(key):
        return ae._bridge_failed_before(key, 1000.0, 0.5)

    @ae.bridge_failure_scope
    def outer():
        key = ae._bridge_key(_ev(), mask, nbm)
        ae._note_bridge_failure(key, 1000.0, 0.5)
        seen_inside = inner(key)
        return seen_inside, ae._bridge_failed_before(key, 1000.0, 0.5)
    assert outer() == (False, True)


def test_a_repeated_failure_skips_the_bridge(monkeypatch):
    calls = []

    def run(ev, state, *a, **k):
        calls.append(1)
        return False, state
    monkeypatch.setattr(ae, "_bridge_far_multirev_run", run)
    mask, nbm = _masks()

    @ae.bridge_failure_scope
    def body():
        ev = _ev()
        s = np.arange(6.0)
        for off in (469679.0, 469671.6):                              # the second is the repeat
            ok, out = ae._bridge_far_multirev(ev, s, mask, nbm, (0, 4), off, 0.379,
                                              mask | nbm, False)
            assert not ok and out is s
    body()
    assert len(calls) == 1


def test_fit_orbit_carries_the_scope():
    assert getattr(fo.fit_orbit, "__wrapped__", None) is not None
