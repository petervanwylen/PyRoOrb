"""OFFLINE tests for the published-orbit FAST PATH (fit_object._fit_from_mpc_seed +
mpc_orbit). No network: the frame conversion is checked algebraically, and the
seed->fit->drop/reclaim machinery is exercised by FEEDING IT a seed derived from the
fixture's own IOD fit (the published MPC orbit is just a better-quality version of that
same heliocentric-equatorial state, so a converged orbit stands in for it perfectly).

    python -m pytest tests/test_mpc_seed.py  # or under pytest

The live API + real published-orbit conversion are validated separately (see PROVENANCE
/ the dev notes); those can't live in the offline suite without making it network- and
time-dependent, which is exactly what keeping the seed OUT of fit_orbit avoids.
"""
import os

import numpy as np
import pytest

import pyroorb.fit_object as fo
from pyroorb import obs_sources
from pyroorb import orbital_elements
from pyroorb import mpc_orbit
from _fixtures import FIX, load_fixture

# compares a seeded fit against a full IOD fit
pytestmark = pytest.mark.slow

# A qualifying single-designation fixture: > MPC_SEED_MIN_OBS obs across > MPC_SEED_MIN_OPP
# oppositions, so _qualifies_for_mpc_seed accepts it. (802 obs / 18 oppositions.)
_FIXTURE = "107474.json"


def test_ecl_eq_roundtrip():
    """mpc_orbit._ecl_to_eq must be the exact inverse of orbital_elements._eq_to_ecl -- i.e. the
    ecliptic->equatorial rotation that turns the MPC CAR block into the engine's frame."""
    rng = [np.array([1.3, -0.7, 0.25]), np.array([-2.1, 0.4, -1.8]),
           np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    for v in rng:
        back = orbital_elements._eq_to_ecl(mpc_orbit._ecl_to_eq(v))
        assert np.allclose(back, v, atol=1e-12), f"not inverse for {v}: got {back}"
    # The two modules must agree on the obliquity, or the round-trip would only be
    # approximately inverse -- guard the constant explicitly.
    assert abs(mpc_orbit._OBLIQ - orbital_elements.OBLIQ) < 1e-12


def test_extract_mpc_orb_shape():
    """The real API nests the orbit as [{... "mpc_orb": [ {orbit} ] ...}, ...]; the
    extractor must dig that out and tolerate a missing/empty payload (-> None)."""
    sample = [{"ele220": []}, {"mpc_orb": [{"CAR": {"ok": 1}}]}]
    assert mpc_orbit._extract_mpc_orb(sample) == {"CAR": {"ok": 1}}
    assert mpc_orbit._extract_mpc_orb([{"nope": 1}]) is None
    assert mpc_orbit._extract_mpc_orb({}) is None


def test_fast_path_matches_iod():
    """Seeding the fast path with a good orbit must reproduce the IOD fit: same link
    quality (rms within tolerance, no meaningful loss of included obs) and it must
    actually take the MPC branch (method == 'mpc'), exercising the mid-arc re-anchor +
    drop/reclaim loop offline."""
    fx = load_fixture(os.path.join(FIX, _FIXTURE))
    obs = fx["obs"]
    oc = obs_sources.load_obscodes()
    # Force single-object intent (pre-existing fixtures predate the n_requested field).
    meta = {**fx["meta"], "n_requested": 1}
    assert fo._qualifies_for_mpc_seed(obs, meta), "fixture should qualify for the fast path"

    base = fo.fit_orbit(obs, oc, desig=meta["designation"], meta=meta)   # IOD baseline
    # A converged heliocentric-equatorial state at a known epoch -- the stand-in seed.
    seed = {"state_eq": np.asarray(base._state, float),
            "epoch_tdb": float(base.epoch_tdb),
            "n_opp": None, "nobs_total": None, "nongrav": False}
    fast = fo.fit_orbit(obs, oc, desig=meta["designation"], meta=meta, mpc_seed=seed)

    assert fast.iod_method == "mpc", f"expected the MPC branch, got {fast.iod_method!r}"
    assert abs(fast.rms_residual() - base.rms_residual()) < 0.05, (
        f"rms drift: fast {fast.rms_residual():.3f}\" vs IOD {base.rms_residual():.3f}\"")
    assert fast.n_observations() >= base.n_observations() - max(3, 0.02 * len(obs)), (
        f"fast path clipped too many obs: {fast.n_observations()} vs IOD {base.n_observations()}")


def test_no_seed_keeps_iod():
    """With no seed supplied (the regression suite's path), a qualifying object must still
    go through IOD -- the fast path is opt-in via mpc_seed, never automatic inside fit."""
    fx = load_fixture(os.path.join(FIX, _FIXTURE))
    obs = fx["obs"]
    meta = {**fx["meta"], "n_requested": 1}
    fit = fo.fit_orbit(obs, obs_sources.load_obscodes(), desig=meta["designation"], meta=meta)
    assert fit.iod_method != "mpc", "no-seed fit must not take the MPC fast path"


# ---------------------------------------------------------------------------
# A published COMET orbit's A1/A2/A3 ride along as a seed (2026-09-05): the re-anchor runs
# under the published model, the rescue's re-fit reuses the seed, discovery warm-starts.
# ---------------------------------------------------------------------------
def _mpc_orb(model, coeffs, values, dt=False):
    """A minimal mpc_orb dict in the API's shape: the non-grav flags plus a CAR block."""
    names = ["x", "y", "z", "vx", "vy", "vz"] + list(values)
    vals = [1.0, 0.5, 0.1, -0.005, 0.01, 0.001] + [values[k] for k in values]
    kinds = {k: False for k in ("marsden", "srp", "yabushita", "yarkovsky", "yc")}
    kinds[model] = True
    flags = {k: False for k in ("A1", "A2", "A3", "DT", "srp", "yarkovsky")}
    for k in coeffs:
        flags[k] = True
    flags["DT"] = dt
    return {"CAR": {"coefficient_names": names, "coefficient_values": vals},
            "non_grav_booleans": {"non_grav_coefficients": flags, "non_grav_model": kinds,
                                  "non_gravs": True}}


def _cv(m):
    car = m["CAR"]
    return dict(zip(car["coefficient_names"], car["coefficient_values"]))


def test_comet_nongrav_params_from_published_orbit():
    """Marsden / Yeomans-Chodas (no DT) A's come out in physical au/day^2 with absent axes 0;
    anything that is not the engine's water-ice g(r) -- a DT solution, SRP, Yarkovsky,
    Yabushita -- or a flagged axis with no value is refused (None), never seeded as zeros."""
    f = mpc_orbit._comet_nongrav_params
    m = _mpc_orb("yc", ("A1", "A2"), {"A1": 2.5e-10, "A2": 8.2e-12})       # 10P's shape
    assert f(m, _cv(m)) == {"A1": 2.5e-10, "A2": 8.2e-12, "A3": 0.0}
    m = _mpc_orb("marsden", ("A1", "A2", "A3"), {"A1": 1e-9, "A2": -2e-11, "A3": 3e-12})
    assert f(m, _cv(m)) == {"A1": 1e-9, "A2": -2e-11, "A3": 3e-12}
    m = _mpc_orb("yc", ("A1", "A2"), {"A1": 2.5e-10, "A2": 8.2e-12}, dt=True)
    assert f(m, _cv(m)) is None
    for kind in ("srp", "yarkovsky", "yabushita"):
        m = _mpc_orb(kind, ("A1",), {"A1": 1e-11})
        assert f(m, _cv(m)) is None, kind
    m = _mpc_orb("marsden", ("A1", "A2"), {"A1": 1e-9})          # A2 flagged, no value
    assert f(m, _cv(m)) is None
    m = _mpc_orb("marsden", ("A1",), {"A1": 0.0})                 # all-zero seed = no seed
    assert f(m, _cv(m)) is None


def _comet_seed_from(base, A):
    return {"state_eq": np.asarray(base._state, float)[:6], "epoch_tdb": float(base.epoch_tdb),
            "n_opp": None, "nobs_total": None, "nongrav": True,
            "comet_nongrav": {"A1": A[0], "A2": A[1], "A3": A[2]}}


def test_comet_seed_reanchors_under_the_model_and_leaves_the_seed(monkeypatch):
    """With comet A's on the seed the fast path (a) propagates the seed to mid-arc under the
    comet g(r) -- a visibly different mid-arc state from the gravity-only re-anchor -- and
    (b) leaves `mpc_seed` on the ctx for the rescue's re-fit. The A's go no further (the
    model sequence warm start was measured a null and removed), with or without
    PYROORB_MPC_COMET_SEED."""
    fx = load_fixture(os.path.join(FIX, _FIXTURE))
    obs = fx["obs"]
    oc = obs_sources.load_obscodes()
    meta = {**fx["meta"], "n_requested": 1}
    base = fo.fit_orbit(obs, oc, desig=meta["designation"], meta=meta, auto_nongrav=False)
    import pyroorb.nongrav as ng
    A = [3e-9, 0.0, 0.0]                             # a large radial A1: unmissable in the re-anchor
    grav = fo._seeded_midarc_evaluator(obs, oc, base._state[:6], float(base.epoch_tdb))
    cometary = fo._seeded_midarc_evaluator(obs, oc, base._state[:6], float(base.epoch_tdb),
                                           nongrav=(A, ng.COMET_GR))
    assert grav is not None and cometary is not None
    assert abs(grav[3] - cometary[3]) < 1e-9         # same mid-arc epoch
    assert not np.allclose(grav[2], cometary[2], rtol=0, atol=1e-9), (
        "the out-gassing must move the re-anchored state")

    seed = _comet_seed_from(base, A)
    fast = fo.fit_orbit(obs, oc, desig=meta["designation"], meta=meta, mpc_seed=seed,
                        auto_nongrav=False)
    assert fast.iod_method == "mpc"
    assert fast._ctx["mpc_seed"] is seed
    assert "nongrav_seed" not in fast._ctx and not getattr(fast, "nongrav_model", None)

    monkeypatch.setenv("PYROORB_MPC_COMET_SEED", "0")
    plain = fo.fit_orbit(obs, oc, desig=meta["designation"], meta=meta, mpc_seed=seed,
                         auto_nongrav=False)
    assert plain.iod_method == "mpc"
    assert plain._ctx["mpc_seed"] is seed            # the state seed still rides for the rescue


def test_rescue_refit_reuses_the_published_seed(monkeypatch):
    """_rescue_single_night_poison's from-scratch re-fit must receive the seed that started
    the outer fit (10P: 844 of 919 s were this re-fit going through IOD without it). Gates
    are forced open and the inner fit_orbit replaced by a recorder, so this checks the
    plumbing, not the verdict."""
    fx = load_fixture(os.path.join(FIX, _FIXTURE))
    obs = fx["obs"]
    oc = obs_sources.load_obscodes()
    meta = {**fx["meta"], "n_requested": 1}
    base = fo.fit_orbit(obs, oc, desig=meta["designation"], meta=meta, auto_nongrav=False)
    seed = _comet_seed_from(base, [1e-10, 0.0, 0.0])
    best = fo.fit_orbit(obs, oc, desig=meta["designation"], meta=meta, mpc_seed=seed,
                        auto_nongrav=False)
    assert best.iod_method == "mpc"
    best._mask[:2] = False                           # >= 2 exclusions: past the user's floor
    monkeypatch.setattr(fo, "SOFT_FIT_RMS", -1.0)    # "elevated" -> no clean exit
    monkeypatch.setattr(fo, "RESCUE_OVERDET_OPP", 10 ** 9)
    monkeypatch.setattr(fo, "_clusters", lambda times: [(0, 0)])   # one single-night opp
    seen = {}

    def recorder(*a, **kw):
        seen.update(kw)
        raise RuntimeError("stop here")             # the rescue returns `best` on any error
    monkeypatch.setattr(fo, "fit_orbit", recorder)
    out = fo._rescue_single_night_poison(best, obs, oc, meta["designation"], meta, False)
    assert out is best
    assert seen.get("mpc_seed") is seed, "the re-fit must be handed the published-orbit seed"
    assert seen.get("_allow_sn_drop") is False and seen.get("auto_nongrav") is False


# ---------------------------------------------------------------------------
# Subsampled discovery first round (2026-09-05): the model-comparison trials on a large arc run
# on a stratified subsample first; a declined subsample falls back to the full round.
# ---------------------------------------------------------------------------
class _FakeEv:
    """Just what _ng_subsample_mask reads: times + the evaluator's own opposition spread."""
    def __init__(self, times):
        self.times_tdb = np.asarray(times, float)

    def _spread_over_oppositions(self, idx, cap):
        from pyroorb import arc_extend
        return arc_extend.Evaluator._spread_over_oppositions(self, idx, cap)


def _dense_arc(n_nights_per_opp=40, per_night=20, n_opp=6, gap_days=500.0):
    """A synthetic dense multi-opposition arc: n_opp apparitions of n_nights nights x per_night."""
    t = []
    t0 = 2450000.0
    for o in range(n_opp):
        for d in range(n_nights_per_opp):
            base = t0 + o * gap_days + d
            t.extend(base + 0.3 + 0.01 * np.arange(per_night))
    return np.asarray(t)


def test_subsample_mask_keeps_every_night_and_stays_small():
    t = _dense_arc()                                   # 6 x 40 x 20 = 4800 obs
    ev = _FakeEv(t)
    mask = np.ones(len(t), bool)
    mask[::7] = False                                  # some exclusions must be honoured
    sub = fo._ng_subsample_mask(ev, mask)
    assert sub is not None and sub.dtype == bool and sub.shape == mask.shape
    assert not np.any(sub & ~mask), "the subsample may only pick INCLUDED obs"
    n = int(sub.sum())
    assert 0 < n <= 1.5 * fo.NONGRAV_SUBSAMPLE_TARGET, n
    # every apparition survives, with its endpoints (linspace keeps both ends of a group)
    days = np.floor(t).astype(int)
    opp = (days - days.min()) // 500
    for o in range(6):
        assert np.any(sub & (opp == o)), f"apparition {o} lost"
    # deterministic
    assert np.array_equal(sub, fo._ng_subsample_mask(ev, mask))
    # below the threshold: no subsample at all
    small = np.zeros_like(mask); small[:fo.NONGRAV_SUBSAMPLE_MIN_OBS] = True
    assert fo._ng_subsample_mask(ev, small) is None


def test_first_round_falls_back_when_the_subsample_declines():
    t = _dense_arc()
    ev = _FakeEv(t)
    full = np.ones(len(t), bool)
    calls = []

    def evaluate_declines(mask):
        calls.append(int(mask.sum()))
        return {"choice": None if mask.sum() < len(t) else "comet2"}
    info = {}
    sel, used, subsampled = fo._ng_first_round(ev, full, evaluate_declines, info, False)
    assert subsampled is False and used is full and sel["choice"] == "comet2"
    assert len(calls) == 2 and calls[0] < calls[1] == len(t), calls
    assert info["subsample"]["declined"] is True and info["subsample"]["n_full"] == len(t)

    calls.clear(); info = {}

    def evaluate_accepts(mask):
        calls.append(int(mask.sum()))
        return {"choice": "comet2"}
    sel, used, subsampled = fo._ng_first_round(ev, full, evaluate_accepts, info, False)
    assert subsampled is True and int(used.sum()) == calls[0] < len(t) and len(calls) == 1
    assert info["subsample"]["declined"] is False

    # a small arc never subsamples and records nothing
    small = np.zeros_like(full); small[:100] = True
    calls.clear(); info = {}
    sel, used, subsampled = fo._ng_first_round(ev, small, evaluate_accepts, info, False)
    assert subsampled is False and used is small and calls == [100] and "subsample" not in info


# ---------------------------------------------------------------------------
# 2026-09-06: a catastrophic SEEDED fit gets a cheap per-apparition alternative (no IOD), and
# a discovered model orbit that is itself catastrophic is refused when the gravity orbit was
# not (21P: 2622" gravity -> comet2 delivered at 491" with A1 = 7e-8).
# ---------------------------------------------------------------------------
class _Orb:
    def __init__(self, rms_n, n, rms=1.0, method="mpc", ctx=None):
        self._rn, self._n, self._rms, self.iod_method, self._ctx = rms_n, n, rms, method, ctx
        self.history = []
    def rms_normalized(self):  return self._rn
    def n_observations(self):  return self._n
    def rms_residual(self):    return self._rms


def test_delivery_sane_refuses_a_catastrophic_model_over_a_clean_gravity():
    info = {}
    assert fo._ng_delivery_sane(_Orb(180.0, 7432), _Orb(1.0, 1246), {}, info) is False
    assert "catastrophic" in info["reason"] and "reverted to gravity" in info["reason"]
    # a catastrophic model over a catastrophic gravity is left to the other rules
    assert fo._ng_delivery_sane(_Orb(180.0, 7432), _Orb(50.0, 7434), {}, {}) is True
    # the normal case
    assert fo._ng_delivery_sane(_Orb(0.9, 7000), _Orb(9.0, 5900), {}, {}) is True


def test_catastrophic_link_takes_the_seeded_route_for_an_mpc_fit(monkeypatch):
    seen = {}
    monkeypatch.setattr(fo, "_catastrophic_seeded_alternative",
                        lambda fit, obs, clusters, verbose=False: seen.setdefault("fit", fit))
    monkeypatch.setattr(fo, "_clusters", lambda times: [(0, 4), (5, 9)])
    obs = [{"jd_utc": 2450000.0 + i * (1 if i < 5 else 400)} for i in range(10)]
    f = _Orb(30.0, 10, ctx={"ev": object()})
    assert fo._reject_catastrophic_link(f, obs, None, "x", {}) is f and seen["fit"] is f
    # a clean fit never reaches it; an IOD fit keeps the IOD route (not dispatched here)
    seen.clear()
    fo._reject_catastrophic_link(_Orb(1.0, 10, ctx={"ev": object()}), obs, None, "x", {})
    assert not seen


def test_seeded_alternative_prefers_the_richest_clean_apparition(monkeypatch):
    """Apparition rms (normalized) by design: [0..9] clean at 1.5, [10..29] catastrophic at
    40, [30..34] clean at 0.5 -> the 10-obs clean apparition wins (clean beats catastrophic
    regardless of count; among clean, richest wins -- the prefer-the-recent rule belongs to
    the apparition-wise extension, not here: see _catastrophic_seeded_alternative). No clean
    apparition -> the fit stands."""
    n = 35
    clusters = [(0, 9), (10, 29), (30, 34)]
    table = {(0, 9): 1.5, (10, 29): 40.0, (30, 34): 0.5}

    class Ev:
        def fit(self, state, mask, max_nfev=None, thin=True):
            return np.asarray(state, float), object()
        def rms_norm(self, st, mask):
            idx = np.flatnonzero(mask)
            return table[(int(idx[0]), int(idx[-1]))]
    built = {}

    def fake_build(ctx, st, mask, result, history):
        idx = np.flatnonzero(mask)
        built["span"] = (int(idx[0]), int(idx[-1]))
        return _Orb(table[built["span"]], int(mask.sum()), rms=0.7)
    monkeypatch.setattr(fo, "_build_orbit", fake_build)
    obs = [{"jd_utc": 2450000.0 + i} for i in range(n)]
    fit = _Orb(60.0, n, ctx={"ev": Ev()})
    fit._state = np.zeros(6)
    out = fo._catastrophic_seeded_alternative(fit, obs, clusters)
    assert out is not fit and built["span"] == (0, 9) and out.n_observations() == 10
    assert out.history and out.history[-1]["stage"] == "catastrophic"

    table.update({(0, 9): 9.0, (30, 34): 12.0})          # nothing clean -> keep the fit
    assert fo._catastrophic_seeded_alternative(fit, obs, clusters) is fit
