"""The station-night error model (nightmodel.py): the whitener is S^-1/2, a one-
restriction Delta chi2 between two of its optima is chi2(1) on two nights, a night is worth what
the block's correlation says, a station's nights share a decaying offset, and the arc's own
scatter re-scales k with the measured prior. Pure numpy."""
import math

import numpy as np
import pytest

import pyroorb.nightmodel as nm
import pyroorb.hyperbolic as hyp


def _arc(n_nights=2, per_night=4, stations=("F51", "G96"), sigma=0.3, night_step=1.0):
    jd, codes, sig = [], [], []
    for b in range(n_nights):
        for i in range(per_night):
            jd.append(2460000.0 + b * night_step + i * 0.01)
            codes.append(stations[b % len(stations)])
            sig.append(sigma)
    return np.array(jd), codes, np.array(sig), np.array(sig)


def _obs_resid(v):
    """A (2m,) interleaved sigma-unit vector as an (m, 2) arcsec-unit array is the caller's job;
    here the tests stay in sigma units (sigma = 1 where they need arcsec)."""
    return v.reshape(-1, 2)


def test_station_night_index_is_one_station_and_a_half_day_gap():
    jd = np.array([0.0, 0.01, 0.3, 1.0, 0.02, 0.05])
    codes = ["F51", "F51", "F51", "F51", "G96", "G96"]
    gid = nm.station_night_index(jd, codes)
    assert gid[0] == gid[1] == gid[2]                        # 0.3 d later is the same night
    assert gid[3] != gid[0]                                  # a day later is not
    assert gid[4] == gid[5] and gid[4] != gid[0]             # another station the same night is another block
    assert len(set(gid)) == 3


def test_the_whitener_is_the_inverse_square_root_of_the_block_covariance():
    """W^T W = S^-1 per block and axis, S^1/2 its inverse: r^T S^-1 r = |W r|^2 for any r."""
    rng = np.random.default_rng(1)
    jd, codes, sra, sdec = _arc(n_nights=3, per_night=5, stations=("703", "F51", "703"), night_step=2.0)
    W = nm.NightWhitener(jd, codes, sra, sdec)
    assert W.n_blocks == 2 and W.n_nights == 3              # 703's two nights are ONE block (correlated)
    rid = nm.station_run_index(jd, codes, nm.RUN_BREAK_FACTOR * nm.NIGHT_CORR[1])
    for rows, Wb, S_half in W.blocks:
        obs, ax = rows // 2, rows[0] % 2
        sel = np.flatnonzero(rid == rid[obs[0]])
        assert set(sel) == set(obs)
        p = nm.station_params(codes[sel[0]])
        S = nm.block_covariance(jd[sel], (sra if ax == 0 else sdec)[sel], (p[ax], p[2 + ax], p[4 + ax]))
        assert np.allclose(Wb.T @ Wb, np.linalg.inv(S), rtol=1e-8, atol=1e-10)
        assert np.allclose(S_half @ S_half, S, rtol=1e-8, atol=1e-12)
    v = rng.standard_normal(2 * len(jd))
    Wm = W.apply(np.eye(2 * len(jd)))
    assert W.chi2(v) == pytest.approx(float(v @ Wm.T @ Wm @ v), rel=1e-9)


def test_block_covariance_has_the_model_written_in_it():
    """Within a night: k^2 on the diagonal plus tau^2 everywhere plus the rate term; across two
    nights of one station: tau^2 rho(dt), no rate term."""
    jd = np.array([0.0, 0.01, 0.02, 5.0, 5.01])
    S = nm.block_covariance(jd, np.ones(5), (0.5, 0.2, 10.0))
    t1 = jd[:3] - jd[:3].mean()
    assert S[0, 0] == pytest.approx(0.25 + 0.04 + 100 * t1[0] ** 2)
    assert S[0, 1] == pytest.approx(0.04 + 100 * t1[0] * t1[1])
    assert S[0, 3] == pytest.approx(0.04 * nm.night_kernel(5.005 - 0.01))  # across nights: the kernel only
    assert 0 < nm.night_kernel(5.0) < nm.night_kernel(0.1) < 1.0 and nm.night_kernel(0.0) == 1.0
    assert nm.night_kernel(1000.0) < 1e-6


def test_a_model_draw_has_chi2_of_its_dof_under_the_whitener_and_not_under_the_diagonal():
    """Model residuals whitened are unit-normal: chi2 ~ 2n. The same residuals in units of the
    assigned sigma are NOT (the offsets and the k < 1 scatter both move it), which is the whole
    reason the gate could not be scaled per observation."""
    rng = np.random.default_rng(2)
    jd, codes, sra, sdec = _arc(n_nights=3, per_night=6, stations=("703", "704", "F51"))
    W = nm.NightWhitener(jd, codes, sra, sdec)
    chi2w, chi2d = [], []
    for _ in range(600):
        v = W.draw(rng)
        chi2w.append(W.chi2(v)); chi2d.append(float(v @ v))
    n = 2 * len(jd)
    assert np.mean(chi2w) == pytest.approx(n, rel=0.08)
    assert np.var(chi2w) == pytest.approx(2 * n, rel=0.3)
    assert not (0.85 < np.mean(chi2d) / n < 1.15)


def test_a_one_restriction_delta_chi2_between_whitened_optima_is_chi2_1_on_two_nights():
    """The statement the gate rests on. A linear model y = A x + eps with eps from the night
    model and a design shaped like two nights of an orbit fit -- a mean position per night and
    axis, a rate per axis -- so the free fit absorbs each night's offset as a real fit does. The
    restriction pins the RA night-mean DIFFERENCE (the quantity the range, hence e, is read
    from). Under the whitener the Delta chi2 between the two optima is chi2(1). Under the
    diagonal weighting with the fit's own reduced chi2 (the retired reading) the same quantity
    is inflated by the offsets times the observations per night -- what read 2010 LP68's own
    orbit as 9 sigma. Both nights from ONE station, a day apart: the correlated case."""
    rng = np.random.default_rng(3)
    jd, codes, sra, sdec = _arc(n_nights=2, per_night=8, stations=("703", "703"), sigma=1.0)
    n = len(jd)
    gid = nm.station_night_index(jd, codes)
    A = np.zeros((2 * n, 6))
    t = jd - jd.mean()
    for i in range(n):
        A[2 * i, 0 if gid[i] == 0 else 2] = 1.0            # RA mean of night 1 / night 2
        A[2 * i + 1, 1 if gid[i] == 0 else 3] = 1.0        # Dec mean of night 1 / night 2
        A[2 * i, 4] = t[i]; A[2 * i + 1, 5] = t[i]         # a rate per axis
    g = np.zeros(6); g[0], g[2] = 1.0, -1.0
    W = nm.NightWhitener(jd, codes, sra, sdec)
    Aw = W.apply(A)
    Cw = np.linalg.inv(Aw.T @ Aw); Cd = np.linalg.inv(A.T @ A)
    dw, dd = [], []
    for _ in range(1500):
        v = W.draw(rng)
        xw = Cw @ Aw.T @ W.apply(v)
        dw.append(float(g @ xw) ** 2 / float(g @ Cw @ g))
        xd = Cd @ A.T @ v
        s2 = float((v - A @ xd) @ (v - A @ xd)) / (2 * n - 6)
        dd.append(float(g @ xd) ** 2 / float(g @ Cd @ g) / s2)
    dw, dd = np.array(dw), np.array(dd)
    assert np.median(dw) == pytest.approx(0.455, abs=0.08)           # chi2(1)'s median
    assert np.mean(dw >= hyp.BOUND_DAMAGE_DCHI2) < 0.012              # 0.27% nominal; sampling
    assert np.median(dd) > 3.0                                        # the per-observation reading, inflated
    assert np.mean(dd >= hyp.BOUND_DAMAGE_DCHI2) > 0.2                # ...refusing the truth routinely


def test_a_night_is_worth_kish_and_two_correlated_nights_less_than_two_independent_ones():
    assert nm.kish_neff(4, 0.0) == 4.0
    assert nm.kish_neff(4, 1.0) == 1.0
    assert nm.kish_neff(4, 0.5) == pytest.approx(1.6)
    # two nights from two stations: separate blocks, the block's 1^T R^-1 1 IS Kish per night
    jd, codes, sra, sdec = _arc(n_nights=2, per_night=4, stations=("F51", "G96"), sigma=0.2)
    W = nm.NightWhitener(jd, codes, sra, sdec)
    expect = 0.0
    for c in ("F51", "G96"):
        p = nm.station_params(c)
        for ax in range(2):
            S0 = nm.block_covariance(jd[:4], np.full(4, 0.2), (p[ax], p[2 + ax], p[4 + ax]), rate=False)
            d = np.sqrt(np.diag(S0)); expect += float(d @ np.linalg.inv(S0) @ d)
            rho = p[2 + ax] ** 2 / (p[2 + ax] ** 2 + (p[ax] * 0.2) ** 2)
            assert float(d @ np.linalg.inv(S0) @ d) == pytest.approx(nm.kish_neff(4, rho), rel=1e-9)
    assert W.n_eff == pytest.approx(expect)
    assert 4 < W.n_eff < 16 and W.n_blocks == 2
    # the same two nights from ONE station a day apart share an offset: worth less
    jd1, codes1, s1a, s1d = _arc(n_nights=2, per_night=4, stations=("F51", "F51"), sigma=0.2)
    W1 = nm.NightWhitener(jd1, codes1, s1a, s1d)
    jd2, codes2, s2a, s2d = _arc(n_nights=2, per_night=4, stations=("F51", "F51"), sigma=0.2, night_step=400.0)
    W2 = nm.NightWhitener(jd2, codes2, s2a, s2d)
    assert W1.n_blocks == 1 and W2.n_blocks == 2                        # 400 d apart: past the kernel
    assert W1.n_eff < W2.n_eff
    # the kernel's own worth: without it (rho0 = 0) the one block reads as two independent nights
    W0 = nm.NightWhitener(jd1, codes1, s1a, s1d, corr=(0.0, nm.NIGHT_CORR[1]))
    assert W0.n_eff == pytest.approx(W2.n_eff, rel=1e-6)


def test_within_scale_recovers_a_planted_scatter_and_leans_on_the_measured_prior_when_thin():
    rng = np.random.default_rng(5)
    jd, codes, sra, sdec = _arc(n_nights=6, per_night=10, stations=("G96", "T08"), sigma=0.4)
    W2 = nm.NightWhitener(jd, codes, sra, sdec, k_scale=2.0)
    v = W2.draw(rng)                                       # arcsec = sigma units x 0.4
    r = v.reshape(-1, 2) * 0.4
    ks, dof, ratio, sdof = nm.within_scale(jd, codes, r[:, 0], r[:, 1], sra, sdec)
    assert dof == 6 * (10 - 2) * 2
    assert sdof == pytest.approx(dof + nm.SCALE_PRIOR_DOF)
    assert ratio == pytest.approx(4.0, rel=0.3)                       # variance ratio of a 2x scatter
    assert 1.6 < ks < 2.2                                             # shrunk a little toward 1 by the prior
    # two nights of two observations leave no line to measure scatter over: the table's k stands
    jd2, codes2, s2a, s2d = _arc(n_nights=2, per_night=2, stations=("F51",))
    assert nm.within_scale(jd2, codes2, np.zeros(4), np.zeros(4), s2a, s2d) == (1.0, 0, None, None)
    # a cleaner-than-population object earns a tighter scale (symmetric shrinkage)
    W_half = nm.NightWhitener(jd, codes, sra, sdec, k_scale=0.5)
    r3 = W_half.draw(rng).reshape(-1, 2) * 0.4
    ks3, _, _, _ = nm.within_scale(jd, codes, r3[:, 0], r3[:, 1], sra, sdec)
    assert ks3 < 0.75
    # the prior is the measured one, not an asserted 8
    assert 1.0 < nm.SCALE_PRIOR_DOF < 20.0


def test_an_unlisted_station_takes_the_pooled_row():
    assert nm.station_params("ZZZ") == tuple(float(v) for v in nm.POOLED[:6])
    assert nm.station_params("F51") != nm.station_params("ZZZ")
    assert all(0.1 < nm.station_params(c)[0] < 1.5 for c in ("F51", "G96", "703", "ZZZ"))


def test_the_singleton_block_pays_its_systematic_and_a_whitened_jacobian_matches_the_residual_map():
    """A lone observation is one block: its whitened residual is r / sqrt(k^2 sigma^2 + tau^2) in
    assigned-sigma units, and apply() is the same linear map on a matrix's columns."""
    jd, codes = np.array([2460000.0]), ["703"]
    W = nm.NightWhitener(jd, codes, np.array([1.0]), np.array([1.0]))
    p = nm.station_params("703")
    z = W.apply(np.array([1.0, 0.0]))
    assert z[0] == pytest.approx(1.0 / math.sqrt(p[0] ** 2 + p[2] ** 2))
    M = np.random.default_rng(6).standard_normal((2, 3))
    assert np.allclose(W.apply(M), np.column_stack([W.apply(M[:, j]) for j in range(3)]))
