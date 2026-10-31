"""The non-gravitational sky Jacobian and the profile seed that drives it.

predict_radec_jac_ng does its RA/Dec geometry in one vectorized pass over every observation
and particle. It must agree to the last bit with the per-observation geometry kernel
(forward_model._radec_from_helio) applied to the same integration, because the discovery's
model comparison is decided on chi2 differences of a few units and must not move when the
geometry is restructured.

_profile_seed_axis memoizes its A = 0 grid point across axes (zero_cache); the cached
profile must return exactly what the uncached one does, while running one fewer state fit.
"""
import numpy as np
import pytest

import pyroorb.forward_model as fm

T0 = 2452996.9
STATE = np.array([0.5, -0.8, 0.1, 0.012, 0.008, -0.002])     # an Apollo-like heliocentric state


def _assist():
    fa = pytest.importorskip("force_assist")
    try:
        fa.get_ephem()
    except Exception as ex:                                  # pragma: no cover - data missing
        pytest.skip(f"ASSIST ephemeris unavailable: {ex}")
    return fa


def _arc(n=120, seed=3):
    rng = np.random.default_rng(seed)
    tt = np.sort(T0 + rng.uniform(-900, 2500, n))            # observations on both sides of t0
    rs = [np.array([np.cos(t / 58.1), np.sin(t / 58.1), 0.02]) for t in tt]
    return tt, rs


@pytest.mark.parametrize("free, A", [([], [0.0, 0.0, 0.0]),
                                     ([1], [0.0, -1e-13, 0.0]),
                                     ([0, 1], [3e-12, -1e-13, 0.0])])
def test_vectorized_geometry_matches_the_per_observation_kernel(free, A):
    fa = _assist()
    tt, rs = _arc()
    pred, jac = fa.predict_radec_jac_ng(T0, STATE, A, free, None, tt, rs)

    # Rebuild the same particles predict_radec_jac_ng rides, integrate them through
    # helio_states_multi (same scaffold, same epsilon, same per-particle A), and apply
    # the per-observation kernel.
    k = len(free)
    hstate = 1.49e-8 * np.maximum(1.0, np.abs(STATE))
    hA = np.array([1e-3 * (abs(A[i]) if A[i] else 1.0) for i in free])
    states = [STATE.copy() for _ in range(7 + k)]
    for j in range(6):
        states[1 + j][j] += hstate[j]
    As = np.array([list(A) for _ in range(7 + k)], float)
    for j in range(k):
        As[7 + j][free[j]] += hA[j]
    hs = fa.helio_states_multi(T0, states, tt, nongrav=(As, None))
    for i in range(len(tt)):
        ra, dec = fm._radec_from_helio(hs[i, :, :3], hs[i, :, 3:], rs[i])
        assert pred[i, 0] == ra[0] and pred[i, 1] == dec[0]
        dra = (ra - ra[0] + 180) % 360 - 180
        assert np.array_equal(jac[i, 0, :6], dra[1:7] / hstate)
        assert np.array_equal(jac[i, 1, :6], (dec[1:7] - dec[0]) / hstate)
        for j in range(k):
            assert jac[i, 0, 6 + j] == dra[7 + j] / hA[j]
            assert jac[i, 1, 6 + j] == (dec[7 + j] - dec[0]) / hA[j]


@pytest.mark.slow
def test_profile_zero_point_cache_is_exact_and_saves_a_fit():
    fa = _assist()
    import pyroorb.fit_object as fo
    import pyroorb.nongrav as ng

    tt, rs = _arc(n=60)
    # Synthetic astrometry from a small transverse drift, so the profile has a valley to find.
    truth, _ = fa.predict_radec_jac_ng(T0, STATE, [0.0, -2e-13, 0.0], [], ng.INVSQ_GR, tt, rs)
    rng = np.random.default_rng(7)
    ra = truth[:, 0] + rng.normal(0, 0.2 / 3600, len(tt))
    dec = truth[:, 1] + rng.normal(0, 0.2 / 3600, len(tt))
    cosd = np.cos(np.radians(dec))
    sig = np.full(len(tt), 0.2)
    p0 = np.concatenate([STATE, [0.0]])
    args = (fa, T0, p0, 1e-14, ng.INVSQ_GR, tt, rs, ra, dec, cosd, sig, sig, len(tt))
    grids = {0: fo.A1_PROFILE_AMR_GRID * ng.SRP1AU, 1: fo.A2_PROFILE_GRID}

    calls = [0]
    real = fa.predict_radec_jac_ng
    class Counting:
        def predict_radec_jac_ng(self, *a, **kw):
            calls[0] += 1
            return real(*a, **kw)
    counting = Counting()

    def run(axis, cache):
        before = calls[0]
        out = fo._profile_seed_axis(counting, *args[1:], axis=axis, grid_phys=grids[axis],
                                    slot=0, continue_from_zero=True, early_stop=True,
                                    zero_cache=cache)
        return out, calls[0] - before

    plain = [run(axis, None) for axis in (0, 1)]
    cache = {}
    cached = [run(axis, cache) for axis in (0, 1)]
    for (p_plain, _), (p_cached, _) in zip(plain, cached):
        assert np.array_equal(p_plain, p_cached)
    assert cached[0][1] == plain[0][1]                       # first axis fills the cache
    assert cached[1][1] < plain[1][1]                        # second axis reuses the A = 0 fit


@pytest.mark.slow
def test_null_profile_stops_after_one_step_each_side():
    """On data with no non-gravitational signal both neighbours of A = 0 rise, so the outward
    walk fits exactly three grid points (zero, the first positive, the first negative) and
    returns the zero point; the negative side's first step is judged against A = 0 too."""
    fa = _assist()
    import pyroorb.fit_object as fo
    import pyroorb.nongrav as ng

    tt, rs = _arc(n=60)
    truth, _ = fa.predict_radec_jac_ng(T0, STATE, [0.0, 0.0, 0.0], [], ng.INVSQ_GR, tt, rs)
    rng = np.random.default_rng(11)
    ra = truth[:, 0] + rng.normal(0, 0.2 / 3600, len(tt))
    dec = truth[:, 1] + rng.normal(0, 0.2 / 3600, len(tt))
    cosd = np.cos(np.radians(dec))
    sig = np.full(len(tt), 0.2)
    p0 = np.concatenate([STATE, [0.0]])

    grid_points = []
    real = fo._make_ng_residual
    def counting(*a, **kw):
        grid_points.append(tuple(kw["a_of_p"](None) if "a_of_p" in kw else a[11](None)))
        return real(*a, **kw)
    fo_mp = pytest.MonkeyPatch()
    fo_mp.setattr(fo, "_make_ng_residual", counting)
    try:
        out = fo._profile_seed_axis(fa, T0, p0, 1e-14, ng.INVSQ_GR, tt, rs, ra, dec, cosd, sig, sig,
                                    len(tt), axis=1, grid_phys=fo.A2_PROFILE_GRID, slot=0,
                                    continue_from_zero=True, early_stop=True)
    finally:
        fo_mp.undo()
    a2 = [g[1] for g in grid_points]
    step_up = min(x for x in fo.A2_PROFILE_GRID if x > 0)
    step_down = max(x for x in fo.A2_PROFILE_GRID if x < 0)
    assert a2 == [0.0, step_up, step_down]
    assert out[6] == 0.0
