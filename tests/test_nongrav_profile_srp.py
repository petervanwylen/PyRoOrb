"""Is the profile-seeded SRP rescue REACHABLE once BIC has chosen a model?

`discover_nongrav` pays the expensive profile-seeded srp trial only when a structural tell fires --
the gravity fit gutted a coherent multi-night block of one apparition -- because an A1-only LM
seeded from A1 = 0 sticks in the gravity corner and the cheap trial reads +0%. That tell was ANDed
with `sel["choice"] is None`, i.e. "only if the cheap pass found nothing".

The AND was a COST guard, written when the effect-size floors made "found nothing" the common
outcome. Since selection became a BIC bar, round 1 finds a justified model far more often,
so the second conjunct now suppresses the rescue on exactly the few-apparition, deep-degeneracy
arcs it was built for -- an object whose A2-only model is justified can no longer discover that
A1+A2 fits it far better. 2026 NA1 is the documented shape of that failure (A2-only 0.886" vs
A1+A2 0.530"); it survives only because its round-1 choice happens to be None.

These drive the real `discover_nongrav` control flow with the evaluator, the tell and the settle
stubbed, so they test the shipped branch rather than a restatement of it.
"""
import numpy as np
import pytest

import pyroorb.fit_object as fo
from pyroorb import ident_check


class _FakeEv:
    """The handful of evaluator methods discover_nongrav touches before the branch."""
    n = 12

    def restore_link_relaxation(self):
        pass

    def fit(self, state, mask, thin=False):
        return np.asarray(state, float), None

    def resid_sigmas(self, state):
        return np.arange(self.n, dtype=float)


class _FakeFit:
    history = ()
    _state = np.zeros(6)
    _ctx = {"ev": _FakeEv(), "backend": type("B", (), {"predict_radec_jac_ng": lambda *a: None})()}

    def semimajor_axis(self):
        return 1.2, 0.0

    # inside the asteroid nongrav gate, which reads perihelion (q < NONGRAV_Q_MAX_AU)
    def perihelion_distance(self):
        return 1.0, 0.0

    def n_oppositions(self):
        return 4


class _Stop(Exception):
    """Raised in place of _ng_settle to end the call right after the branch under test."""


def _run(monkeypatch, choice, block_nights):
    """Call discover_nongrav with everything but the branch stubbed; return the profile_srp
    flags of each _ng_evaluate call it made."""
    calls = []

    def fake_evaluate(ctx, seed, history, mask, max_nfev, verbose, **kw):
        calls.append(bool(kw.get("profile_srp", False)))
        return {"choice": choice, "models": {}, "rms0": 1.0, "chi2_0": 100.0, "N": 50,
                "grav": _FakeFit()}

    monkeypatch.setattr(fo, "_ng_evaluate", fake_evaluate)
    monkeypatch.setattr(fo, "_coherent_block_nights", lambda fit: block_nights)
    # discover_nongrav imports ident_check INSIDE the function, so patch the module.
    monkeypatch.setattr(ident_check, "systematic_misfit", lambda fit: None)
    monkeypatch.setattr(fo, "_ng_settle", lambda *a, **k: (_ for _ in ()).throw(_Stop()))
    try:
        fo.discover_nongrav(_FakeFit())
    except _Stop:
        pass
    return calls


def test_the_tell_alone_buys_the_seeded_trial(monkeypatch):
    """The structural tell fired AND round 1 already chose a model. The seeded trial must still
    run: 'we have an answer' is not 'we have the RIGHT FAMILY', and the tell is what says the
    cheap radial trial went blind."""
    calls = _run(monkeypatch, choice="yarkovsky", block_nights=fo.COHERENT_BLOCK_NIGHTS)
    assert True in calls, (
        "the profile-seeded srp trial never ran even though the coherent-block tell fired -- "
        "a better A1+A2 solution is unreachable for every object BIC justifies in round 1")


def test_no_tell_means_no_seeded_trial(monkeypatch):
    """The cost guard that MATTERS is the tell itself, and it must still hold: without a gutted
    coherent block nothing pays for the seed, whatever round 1 chose."""
    assert _run(monkeypatch, choice=None,
                block_nights=fo.COHERENT_BLOCK_NIGHTS - 1) == [False]
    assert _run(monkeypatch, choice="yarkovsky",
                block_nights=fo.COHERENT_BLOCK_NIGHTS - 1) == [False]


def test_the_original_case_still_works(monkeypatch):
    """Round 1 found nothing and the tell fired -- the 2026 NA1 path, unchanged."""
    assert _run(monkeypatch, choice=None, block_nights=fo.COHERENT_BLOCK_NIGHTS) == [False, True]
