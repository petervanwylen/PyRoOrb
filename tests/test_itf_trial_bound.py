"""An ITF trial fit that runs out of its wall-clock bound is an ordinary failed
fit (ok=False), never an exception that escapes into the search."""
from pyroorb import deadline
from pyroorb.obs_sources import load_obscodes
from pyroorb.itf.config import ItfConfig
from pyroorb.itf.fitter import fit_astrometry
from pyroorb.itf.harness import split_oppositions, obs_lines

import os

from _itf_fixtures import ITF_FIX, load_fixture, load_obs


def _seed_text():
    oc = load_obscodes()
    fx = load_fixture(os.path.join(ITF_FIX, "2014_HY176.json"))
    obs, _meta = load_obs(fx, oc)
    return obs_lines(split_oppositions(obs)[fx["seed_apparition"]]), oc


def test_a_trial_fit_past_its_bound_fails_cleanly():
    text, oc = _seed_text()
    tr = fit_astrometry(text, oc, max_sec=1e-6)
    assert not tr.ok and "ITF trial fit" in tr.error
    # the bound is scoped to that one fit: nothing is left set behind it
    assert deadline.active() is None
    # ...and unbounded (or generously bounded) the same fit succeeds
    assert fit_astrometry(text, oc, max_sec=120).ok


def test_the_search_bounds_its_trial_fits_by_default():
    assert ItfConfig().trial_fit_max_sec == 30.0


def test_no_yarkovsky_rescue_on_a_single_apparition(monkeypatch):
    """Yarkovsky is a drift across apparitions: a one-apparition arc cannot
    measure it, so the retry never fits (2007 DB61's 45-tracklet search latched
    on a 2026-only take); two apparitions retry as before (2021 JK2)."""
    from types import SimpleNamespace
    from pyroorb.itf import search as S
    calls = []
    monkeypatch.setattr(S, "_nongrav_gate", lambda ctx, *fits: 0.5)
    monkeypatch.setattr(S, "_nongrav_fit",
                        lambda ctx, text: calls.append(text) or SimpleNamespace(ok=False))
    monkeypatch.setattr(S, "_note_fit", lambda ctx: None)
    ctx = SimpleNamespace(cfg=ItfConfig(), telemetry={})

    def line(date):
        return f"     PR00001  C{date} 10 00 00.00 +10 00 00.0          20.0 G      F51"
    one = "\n".join(line(d) for d in ("2026 02 21.50000", "2026 02 24.50000",
                                       "2026 03 03.50000")) + "\n"
    assert S._yarkovsky_retry(ctx, one, "rejected", lambda tr: tr, "RESCUED") is None
    assert calls == []
    two = one + line("2018 03 28.50000") + "\n"
    S._yarkovsky_retry(ctx, two, "rejected", lambda tr: tr, "RESCUED")
    assert calls == [two]
