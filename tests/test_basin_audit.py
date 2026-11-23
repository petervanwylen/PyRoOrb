"""The post-fit basin audit (pyroorb.basinaudit): a single-apparition fit that converged into a
wrong range basin is noticed and replaced; a clean fit, and a genuinely bad tracklet, are left
alone.

The end-to-end tests start the fit from a deliberately WRONG seed -- a Herget transfer at a
near-Earth range, the start that reproduces each object's wrong basin -- by substituting the
first IOD seed, and check the delivered orbit with the audit on (and, as the premise, off)."""
import math
import types

import numpy as np
import pytest

from _fixtures import fixture_obs
from pyroorb import basinaudit as ba
from pyroorb import fit_object as fo
from pyroorb import obs_sources


# --------------------------------------------------------------------------------- units
def test_cauchy_transform_reproduces_the_cauchy_cost_and_its_derivative():
    r = np.array([-50.0, -3.0, -0.2, 0.0, 1e-8, 0.7, 4.0, 300.0])
    g, d = ba._cauchy(r)
    c = ba.ROBUST_SCALE
    assert np.allclose(g * g, c * c * np.log1p((r / c) ** 2))
    h = 1e-6
    num = (ba._cauchy(r + h)[0] - ba._cauchy(r - h)[0]) / (2 * h)
    assert np.allclose(d, num, rtol=1e-4, atol=1e-6)


def test_bound_barrier_is_zero_when_bound_and_its_gradient_is_right():
    from pyroorb.forward_model import MU_SUN
    r = np.array([0.8, 0.3, 0.1])
    vesc = math.sqrt(2 * MU_SUN / np.linalg.norm(r))
    v = np.array([0.2, 1.0, 0.1])
    bound = np.concatenate([r, v / np.linalg.norm(v) * 0.9 * vesc])
    assert ba._bound_row(bound)[0] == 0.0
    unbound = np.concatenate([r, v / np.linalg.norm(v) * 1.1 * vesc])
    val, grad = ba._bound_row(unbound)
    assert val > 0
    h = 1e-8
    num = np.array([(ba._bound_row(unbound + h * e)[0] - ba._bound_row(unbound - h * e)[0])
                    / (2 * h) for e in np.eye(6)])
    assert np.allclose(grad, num, rtol=1e-5)


class _Fake:
    def __init__(self, used, rms_n, rows=None, total=None):
        self._used, self._rms_n = used, rms_n
        self._rows = rows or []
        self._obs = [{"jd_utc": 2460000.0 + k} for k in range(total or used)]

    def n_observations(self):
        return self._used

    def rms_normalized(self):
        return self._rms_n

    def residuals(self):
        return self._rows


def _rows(n_ok, bad_sig=None, n_bad=0, span=2.0):
    rows = [{"obscode": "F51", "jd_utc": 2460000.0 + span * k / max(n_ok - 1, 1),
             "included": True, "resid_sigmas": 0.5} for k in range(n_ok)]
    rows += [{"obscode": "N86", "jd_utc": 2460001.0 + 0.01 * k, "included": False,
              "resid_sigmas": bad_sig} for k in range(n_bad)]
    return rows


def test_trigger_reads_the_converged_fit():
    assert ba.triggered(_Fake(40, 0.7, _rows(40))) == []
    # a whole tracklet thrown out far beyond any noise
    assert any("tracklet" in w for w in ba.triggered(_Fake(37, 0.8, _rows(37, 100.0, 3))))
    # ...but a tracklet rejected at a few sigma is ordinary rejection
    assert ba.triggered(_Fake(37, 0.8, _rows(37, 5.0, 3))) == []
    # ...and on a long arc a rejected tracklet is not read at all (range is not ambiguous there)
    assert ba.triggered(_Fake(37, 0.8, _rows(37, 100.0, 3, span=128.0))) == []
    assert any("rejected" in w for w in ba.triggered(_Fake(7, 0.9, _rows(7, 3.0, 4))))
    assert any("rms_n" in w for w in ba.triggered(_Fake(40, 2.0, _rows(40))))


def test_adoption_needs_clearly_more_obs_at_a_comparable_rms():
    assert ba.adopt(_Fake(35, 0.65), _Fake(27, 4.27))            # GS26063's wrong -> right
    assert ba.adopt(_Fake(67, 0.75), _Fake(59, 0.95))            # P/2026 U1
    assert not ba.adopt(_Fake(29, 0.65), _Fake(27, 0.60))        # +2 obs is not "clearly"
    assert not ba.adopt(_Fake(40, 1.50), _Fake(30, 0.60))        # more obs, much worse rms
    assert not ba.adopt(_Fake(40, 9.00), _Fake(30, 0.60))        # a catastrophic candidate
    assert ba.adopt(_Fake(209, 0.46), _Fake(212, 60418.0))       # the delivered fit is not an orbit
    assert not ba.adopt(_Fake(103, 0.7), _Fake(100, 0.7))        # +3 on 100 is under 5%
    # never a fit of half the arc or less, however clean -- even against a catastrophic one
    assert not ba.adopt(_Fake(4, 0.6, total=8), _Fake(8, 61.0))
    assert ba.adopt(_Fake(6, 0.5, total=8), _Fake(8, 61.0))


def test_audit_leaves_untriggered_and_multi_apparition_fits_alone(monkeypatch):
    clean = _Fake(40, 0.7, _rows(40))
    clean._ctx = {"ev": object()}
    assert ba.audit(clean) is clean
    sick = _Fake(27, 4.3, _rows(27, 100.0, 3))
    sick._ctx = {"ev": object()}
    sick._obs = [{"jd_utc": 2460000.0}, {"jd_utc": 2460400.0}]   # two apparitions
    assert ba.audit(sick) is sick
    monkeypatch.setenv("PYROORB_BASIN_AUDIT", "0")
    sick._obs = [{"jd_utc": 2460000.0}, {"jd_utc": 2460001.0}]
    assert ba.audit(sick) is sick


# --------------------------------------------------------------------------------- end to end
def _fit(monkeypatch, name, rho=None, audit=True, shift=None):
    """Fit fixture `name`, from a Herget seed at geocentric `rho` au when given, with the
    tracklet obs indices `shift[0]` moved `shift[1]` arcsec in Dec when given."""
    obs, meta = fixture_obs(name)
    obs = sorted((dict(o) for o in obs), key=lambda o: o["jd_utc"])
    if shift is not None:
        for k in shift[0]:
            obs[k]["dec_deg"] += shift[1] / 3600.0
    monkeypatch.setenv("PYROORB_BASIN_AUDIT", "1" if audit else "0")
    if rho is not None:
        real = fo.initial_orbit
        armed = [True]

        def wrong(o, t, rh, R, subarc=None):
            st, m, sc, sub, t0 = real(o, t, rh, R, subarc=subarc)
            if armed[0] and subarc is None:
                armed[0] = False
                seeds = ba.range_seeds(o, R, t0, rho_geo=(rho,), r_helio=())
                assert seeds, "no wrong seed could be built"
                return seeds[0][1], "wrongstart", sc, sub, t0
            return st, m, sc, sub, t0
        monkeypatch.setattr(fo, "initial_orbit", wrong)
    return fo.fit_orbit(obs, obs_sources.load_obscodes(), desig=meta["designation"], meta=meta)


@pytest.mark.slow
def test_gs26063_climbs_out_of_the_atira_basin(monkeypatch):
    """A near-Earth start reproduces the spurious Atira the IOD once seeded (q 0.43, i 3.9,
    27/40 at 3.9", the N86 discovery night rejected at ~100 sigma). The audit must deliver the
    near-parabolic orbit find_orb gives (q 0.678, i 124.1, node 211.2; 33/40 at 0.54")."""
    off = _fit(monkeypatch, "GS26063", rho=0.1, audit=False)
    assert off.perihelion_distance()[0] < 0.5 and off.inclination()[0] < 10, \
        "premise: without the audit this start stays in the Atira basin"
    fit = _fit(monkeypatch, "GS26063", rho=0.1)
    assert 0.65 < fit.perihelion_distance()[0] < 0.70
    assert 121.0 < fit.inclination()[0] < 127.0
    assert 208.0 < fit.ascending_node()[0] < 215.0
    assert fit.n_observations() >= 33 and fit.rms_residual() < 0.8
    kept_n86 = sum(1 for r in fit.residuals() if r["obscode"] == "N86" and r["included"])
    assert kept_n86 == 3


@pytest.mark.slow
def test_catastrophic_single_apparition_fit_is_replaced(monkeypatch):
    """2017 SN33 from a near-Earth start: 68/235 at 246" (rms_n 299) with nothing to rescue
    it on one apparition. The delivered state corrected on every obs is the right basin."""
    fit = _fit(monkeypatch, "2017_SN33", rho=0.1)
    assert fit.n_observations() >= 233 and fit.rms_residual() < 0.6
    assert abs(fit.perihelion_distance()[0] - 1.840) < 0.05
    assert abs(fit.inclination()[0] - 152.1) < 1.0


@pytest.mark.slow
def test_hemisphere_parallax_wrong_basin_is_replaced(monkeypatch):
    """P/2026 U1 from a near-Earth start: 59/68 at 0.58" (q 0.05) -- a plausible-looking fit
    that throws out the southern W68 tracklet at ~7 sigma. Must return to q 1.67, W68 kept."""
    fit = _fit(monkeypatch, "P2026_U1", rho=0.1)
    assert abs(fit.perihelion_distance()[0] - 1.671) < 0.1
    assert fit.n_observations() >= 66
    assert sum(1 for r in fit.residuals() if r["obscode"] == "W68" and r["included"]) >= 7


@pytest.mark.slow
@pytest.mark.parametrize("name,idx,golden_used", [
    ("2026_LW", (71, 72), 78),                 # a 2-obs tracklet on a 4-day NEO arc
    ("RANGEAMBIG_3NIGHT_MBA", None, 11),       # one night of a 3-night arc: the riskiest case
])
def test_a_genuinely_bad_tracklet_stays_rejected(monkeypatch, name, idx, golden_used):
    """CONTROL: shifting one tracklet 30" makes it genuinely bad. The audit may run (the
    rejected tracklet is a trigger) but must not adopt a basin that takes it back in, and the
    orbit must stay where the clean fit is."""
    if idx is None:
        obs, _ = fixture_obs(name)
        obs = sorted(obs, key=lambda o: o["jd_utc"])
        t = [o["jd_utc"] for o in obs]
        nights = [k for k in range(1, len(t)) if t[k] - t[k - 1] > 0.5]
        idx = tuple(range(nights[0], nights[1]))   # the middle night
    clean = _fit(monkeypatch, name)
    fit = _fit(monkeypatch, name, shift=(idx, 30.0))
    inc = fit.is_included()
    assert not any(inc[list(idx)]), "the shifted tracklet was taken back in"
    assert fit.n_observations() >= golden_used - len(idx) - 2
    assert abs(fit.perihelion_distance()[0] - clean.perihelion_distance()[0]) \
        < 0.05 * clean.perihelion_distance()[0]
