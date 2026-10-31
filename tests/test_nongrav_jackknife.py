"""The delete-1-tracklet jackknife's ARITHMETIC, offline.

The fitting half needs ASSIST and lives in the fit-driven tests; everything here is the pure part
-- the cloud -> verdict statistics, which tracklets get refit, the linearized delete-1 shift that
ranks them, and the stall/collapse routing. Those were untestable while they sat inline in
_compute_jackknife_sigmas, which is why a sqrt over the wrong count and a zero-spread divide could
both live there unnoticed.
"""
import numpy as np
import pytest
import pyroorb.fit_object as fo


# --- the cloud -> verdict arithmetic -------------------------------------------------------
def _cloud(n=10, centre=2e-12, spread=1e-13):
    return list(centre + spread * np.linspace(-1, 1, n))


def test_scale_counts_tracklets_not_refits():
    """The delete-1 SE constant is sqrt(n_tracklets - 1): the population of deletable units, not
    the size of the sample of them we could afford. Using the sample shrank sigma (and inflated z)
    by sqrt((m-1)/(n-1)) on every arc denser than the refit cap."""
    v = _cloud()
    few = fo._jackknife_stats(v, 2e-12, n_tracklets=10, attempted=10, rejected=0)
    many = fo._jackknife_stats(v, 2e-12, n_tracklets=100, attempted=10, rejected=0)
    assert many["se"] / few["se"] == pytest.approx(np.sqrt(99 / 9), rel=1e-9)
    assert many["z"] < few["z"]                       # a wider population -> a LOWER significance


def test_rejections_do_not_shrink_the_constant():
    v = _cloud()
    a = fo._jackknife_stats(v, 2e-12, n_tracklets=20, attempted=10, rejected=0)
    b = fo._jackknife_stats(v, 2e-12, n_tracklets=20, attempted=14, rejected=3)
    assert a["se"] == pytest.approx(b["se"], rel=1e-12)


def test_zero_in_the_cloud_is_a_failure_to_survive():
    """A refit that lands exactly on zero is the coefficient COLLAPSING when that tracklet goes."""
    v = [0.0, 0.0, 0.0, 1.9e-9, 2.0e-9, 2.1e-9]
    d = fo._jackknife_stats(v, 2.0e-9, n_tracklets=6, attempted=6, rejected=0)
    assert d["assessed"] and d["spans_zero"] and d["zeros"] == 3


def test_no_spread_is_not_infinite_significance():
    v = [3e-12] * 8
    d = fo._jackknife_stats(v, 3e-12, n_tracklets=8, attempted=8, rejected=0)
    assert d["assessed"] is False and d["reason"] == "no spread across refits"
    assert "z" not in d                                # never an inf for the UI to render


def test_half_range_rescues_a_tied_median():
    """A MAD of zero with a real spread (a tied middle) falls back to the half-range, not to a
    divide by zero."""
    v = [1e-12] * 6 + [3e-12]
    d = fo._jackknife_stats(v, 1e-12, n_tracklets=7, attempted=7, rejected=0)
    assert d["assessed"] and np.isfinite(d["z"]) and d["se"] > 0


def test_too_many_rejections_is_not_assessed():
    v = _cloud(6)
    d = fo._jackknife_stats(v, 2e-12, n_tracklets=20, attempted=15, rejected=9)
    assert d["assessed"] is False and "non-converged" in d["reason"]


def test_too_few_refits_is_not_assessed():
    d = fo._jackknife_stats([1e-12, 2e-12], 2e-12, n_tracklets=20, attempted=4, rejected=2)
    assert d["assessed"] is False and d["reason"] == "too few converged refits"


# --- which tracklets get refit -------------------------------------------------------------
def test_small_arc_refits_every_tracklet():
    trk = [[i] for i in range(fo.JACKKNIFE_MAX_REFITS)]
    assert fo._jackknife_sample(trk) == list(range(len(trk)))


def test_a_dense_arc_is_sampled_across_the_whole_arc():
    """Systematic, in time order, and it must keep BOTH ends -- a nongrav's leverage sits at the
    arc's temporal extremes, so an interior-only sample would never try the deletions that matter."""
    n = 60
    picks = fo._jackknife_sample([[i] for i in range(n)])
    assert len(picks) == fo.JACKKNIFE_MAX_REFITS
    assert picks == sorted(picks) and picks[0] == 0 and picks[-1] == n - 1


# --- stall vs collapse ---------------------------------------------------------------------
class _ZeroFit:
    def __init__(self, params, diag):
        self.nongrav_params, self.nongrav_diag = params, diag


def test_a_profiled_zero_is_evidence_but_an_unexplained_one_is_a_stall():
    tried = _ZeroFit({"A1": 0.0}, {"rescue_chi2": {"A1": 12.0}})
    assert fo._jackknife_unexplained_zeros(tried, "srp", ["A1"]) == ([], [])
    bare = _ZeroFit({"A1": 0.0}, {})
    assert fo._jackknife_unexplained_zeros(bare, "srp", ["A1"]) == (["A1"], [])


def test_a_zero_no_seed_can_explain_is_dropped_rather_than_retried():
    """The rescue only profiles A1/A2 on the non-comet g(r) family, so paying an ~11s forced seed
    for an A3 or a comet-model zero would buy nothing."""
    bare = _ZeroFit({"A1": 0.0}, {})
    assert fo._jackknife_unexplained_zeros(bare, "comet2", ["A1"]) == ([], ["A1"])


def test_a_moved_coefficient_is_never_a_stall():
    moved = _ZeroFit({"A1": 1e-14}, {})
    assert fo._jackknife_unexplained_zeros(moved, "srp", ["A1"]) == ([], [])


# --- the refit loop's zero routing --------------------------------------------------------
# These drive the REAL _compute_jackknife_sigmas loop with refine_nongrav faked, so they test
# the routing between the stall test and the cloud rather than a restatement of it. The fake
# mirrors the fixed engine: since force_seed took precedence over the srp radial-seed branch
# (the fix these tests guard), a forced retry ALWAYS records `rescue_chi2` -- so an exact zero
# out of it is a DOCUMENTED collapse. Before that precedence, an A1-only srp model routed
# force_seed through the diagnostic-free radial-seed branch, the diag came back empty, and
# every genuine collapse was discarded as an unexplained stall -- biasing the cloud away from
# zeros, i.e. inflating z, the phantom-hiding direction.
class _JkRefit:
    def __init__(self, a1, diag, rn=0.5):
        self.nongrav_params, self.nongrav_diag, self._rn = {"A1": a1}, diag, rn

    def rms_normalized(self):
        return self._rn


class _JkFit:
    """The attributes _compute_jackknife_sigmas touches, and nothing else: 8 tracklets of 3
    obs (one station, 10-min spacing, 1-day gaps), an srp A1 fit at 2e-9."""

    def __init__(self, n_trk=8, per=3):
        n = n_trk * per
        self._jackknife_sigmas = None
        self._mask = np.ones(n, bool)
        self.nongrav_model, self.nongrav_params = "srp", {"A1": 2e-9}
        ev = type("Ev", (), {})()
        ev.times_tdb = np.array([i + j * 0.007 for i in range(n_trk) for j in range(per)])
        ev.obscodes = np.array(["G96"] * n)
        self._ctx = {"backend": type("B", (), {"predict_radec_jac_ng": lambda *a: None})(),
                     "ev": ev}

    def rms_normalized(self):
        return 0.5


def _dropped_tracklet(mask, per=3):
    return int(np.flatnonzero(~mask)[0]) // per


def test_a_collapse_the_rescue_documented_stays_in_the_cloud(monkeypatch):
    """Deleting tracklet 0 kills the coefficient: the plain LM stalls bare (empty diag), the
    forced retry lands 0.0 WITH `rescue_chi2` -- the grid tried and zero won. That is the
    evidence the estimator exists to collect, and it must land in the cloud as a
    failure-to-survive, not be rejected as optimizer noise."""
    fit = _JkFit()

    def fake_refine(base, model, mask=None, force_seed=False, **kw):
        a1 = 0.0 if _dropped_tracklet(mask) == 0 else 2e-9 + _dropped_tracklet(mask) * 1e-11
        if not force_seed:
            return _JkRefit(a1, {})
        return _JkRefit(a1, {"rescue_chi2": {"A1": 42.0}})

    monkeypatch.setattr(fo, "refine_nongrav", fake_refine)
    out = fo.FittedOrbit._compute_jackknife_sigmas(fit)
    assert "A1" in out, out                       # a stub error is swallowed into {} -- fail loudly
    d = out["A1"]
    assert d["assessed"] is True and d["forced"] == 1
    assert d["zeros"] == 1 and d["spans_zero"] is True
    assert d["zero_dropped"] == 0
    assert d["n"] == 8                            # every deletion voted, the collapse included


def test_an_unexplained_zero_from_the_forced_retry_is_still_dropped(monkeypatch):
    """The other side of the routing must not loosen: a forced retry that comes back 0.0 with
    NOTHING in the diag (an engine regression -- the rescue did not run) is a stall, never a
    finding, and the refit is rejected rather than believed."""
    fit = _JkFit()

    def fake_refine(base, model, mask=None, force_seed=False, **kw):
        a1 = 0.0 if _dropped_tracklet(mask) == 0 else 2e-9 + _dropped_tracklet(mask) * 1e-11
        return _JkRefit(a1, {})                   # no rescue record, ever

    monkeypatch.setattr(fo, "refine_nongrav", fake_refine)
    out = fo.FittedOrbit._compute_jackknife_sigmas(fit)
    assert "A1" in out, out
    d = out["A1"]
    assert d["zero_dropped"] == 1 and d["zeros"] == 0
    assert d["n"] == 7 and d["spans_zero"] is False

def test_coverage_counts_a_tracklet_that_could_not_be_deleted():
    """An UNDELETABLE tracklet -- one whose removal leaves too little arc to refit -- is
    skipped before `attempted`, so it entered no count at all. `sampled` was derived from the
    PICK LIST, which cannot see it: an arc where every tracklet was picked but one was
    undeletable reported full coverage, and the web panel renders that as "the sign survives
    dropping ANY one tracklet" for a tracklet that was never tried.

    Deriving coverage from `attempted` catches both reasons at once, which is why it lives
    here with the counts instead of in the caller's loop."""
    v = _cloud(8)
    assert fo._jackknife_stats(v, 2e-12, n_tracklets=8, attempted=8,
                               rejected=0)["sampled"] is False      # exhaustive
    assert fo._jackknife_stats(v, 2e-12, n_tracklets=20, attempted=8,
                               rejected=0)["sampled"] is True       # budget-limited
    # all 8 picked, one undeletable -> 7 tried: partial, where the pick list said exhaustive
    assert fo._jackknife_stats(v, 2e-12, n_tracklets=8, attempted=7,
                               rejected=0)["sampled"] is True
    # a NOT-ASSESSED verdict must still carry it -- the panel reads coverage either way
    thin = fo._jackknife_stats(v[:2], 2e-12, n_tracklets=8, attempted=7, rejected=0)
    assert thin["assessed"] is False and thin["sampled"] is True


# --- the 16-perturber delivery force model ---------------------------------------
def test_asteroid_force_wrapper_forces_and_falls_through():
    """_AsteroidForce turns the perturbers on for every call that accepts `asteroids=`,
    leaves every other attribute untouched (capability hasattr-checks must keep working),
    lets an explicit asteroids= win, and wraps idempotently -- a delivered ctx gets
    re-wrapped by later re-fits and must not nest."""
    calls = {}

    class _B:
        def predict_radec_jac_ng(self, *a, **k):
            calls["p"] = dict(k); return "pred"

        def helio_states_multi(self, *a, **k):
            calls["m"] = dict(k); return "st"

        def other(self):
            return 42

    inner = _B()
    b = fo._AsteroidForce(inner)
    assert b.predict_radec_jac_ng(1) == "pred" and calls["p"] == {"asteroids": True}
    assert b.helio_states_multi(1, asteroids=None) == "st" and calls["m"] == {"asteroids": None}
    assert b.other() == 42
    assert hasattr(b, "predict_radec_jac_ng") and not hasattr(b, "predict_nope")
    assert fo._AsteroidForce(b)._backend is inner


def test_nongrav_asteroids_gate(monkeypatch):
    """The delivery gate: long arc AND belt-sampling aphelion AND ASSIST + kernel present;
    PYROORB_NG_ASTEROIDS=0 kills it. No suite fixture meets it (their aphelia sit under
    ~1.5 au), which is what keeps the offline archive untouched by construction."""
    import pyroorb.force_assist as fa

    class _Assist:
        @staticmethod
        def helio_states_multi(*a, **k):
            return None

    class _F:
        def __init__(self, arc, Q, backend=_Assist()):
            self._arc, self._Q = arc, Q
            self._ctx = {"backend": backend}

        def arc_length_days(self):
            return self._arc

        def aphelion_distance(self):
            return self._Q, 0.0

    monkeypatch.setattr(fa, "asteroid_kernel_path", lambda: "/x/sb441-n16.bsp")
    monkeypatch.delenv("PYROORB_NG_ASTEROIDS", raising=False)
    assert fo._nongrav_asteroids_gate(_F(4000, 2.3)) is True
    assert fo._nongrav_asteroids_gate(_F(3000, 2.3)) is False        # arc too short
    assert fo._nongrav_asteroids_gate(_F(4000, 1.4)) is False        # aphelion inside the belt
    assert fo._nongrav_asteroids_gate(_F(4000, 2.3, backend=object())) is False   # rebound fallback
    monkeypatch.setattr(fa, "asteroid_kernel_path", lambda: None)
    assert fo._nongrav_asteroids_gate(_F(4000, 2.3)) is False        # kernel absent
    monkeypatch.setattr(fa, "asteroid_kernel_path", lambda: "/x/sb441-n16.bsp")
    monkeypatch.setenv("PYROORB_NG_ASTEROIDS", "0")
    assert fo._nongrav_asteroids_gate(_F(4000, 2.3)) is False        # kill switch


def test_asteroids_detection_recheck_arithmetic(monkeypatch):
    """The perturbed re-check: the gravity baseline is REFIT under the perturbed
    model and the chosen model must still clear the BIC bar. A justification that collapses
    was the planets-only baseline absorbing the missing perturbers -- the record says so and
    the caller reverts. Offline: the residual factory is faked, the arithmetic is real."""
    class _Res:
        def __init__(self, fun):
            self.fun = np.asarray(fun, float)

    class _Fit:
        _state = np.zeros(6)

        def __init__(self, fun):
            self._res = _Res(fun)

    def _fake_factory(chi2_grav_per_row):
        def factory(backend, t0, tt, Rs, ra, dec, cosd, sra, sdec, gr, hA,
                    a_of_p, free_cols, col_scale, m):
            r = lambda p: np.full(8, chi2_grav_per_row)
            j = lambda p: np.zeros((8, 6))
            return r, j, None
        return factory

    class _Ev:
        times_tdb = np.arange(4.0)
        Robs = [np.zeros(3)] * 4
        ra = np.zeros(4); dec = np.zeros(4); cosd = np.ones(4)
        fit_sig_ra = np.ones(4); fit_sig_dec = np.ones(4)

    ctx = {"ev": _Ev(), "backend": object(), "epoch_tdb": 2460000.5}
    mask = np.ones(4, bool)

    # gravity refit lands at chi2 = 8*2^2 = 32; the delivered yarkovsky at 8*1 = 8:
    # dBIC = 24 - 1*ln(20) ~ 21 >= 15 -> the detection SURVIVES better physics
    monkeypatch.setattr(fo, "_make_ng_residual", _fake_factory(2.0))
    rec, ok = fo._asteroids_detection_recheck(ctx, _Fit([0]), _Fit(np.ones(8)),
                                              mask, "yarkovsky", 20)
    assert ok is True and rec["dbic"] > fo.NONGRAV_BIC_MARGIN
    assert rec["chi2_grav"] == 32.0 and rec["chi2_model"] == 8.0 and rec["dp"] == 1

    # gravity under perturbers is nearly as good (chi2 10.1) -> dBIC ~ -0.9: the planets-only
    # justification WAS the perturbers; the caller must revert
    monkeypatch.setattr(fo, "_make_ng_residual",
                        _fake_factory(np.sqrt(10.1 / 8.0)))
    rec, ok = fo._asteroids_detection_recheck(ctx, _Fit([0]), _Fit(np.ones(8)),
                                              mask, "yarkovsky", 20)
    assert ok is False and rec["dbic"] < fo.NONGRAV_BIC_MARGIN


def test_final_reclaim_is_a_noop_without_a_late_mover():
    """The end-of-fit reclamation fires only when a LATE mover (delivered nongrav
    or applied radar) changed the orbit after the rejection loop closed. An offline gravity
    fixture has neither -- it must come back as the SAME object, untouched, which is what
    keeps the whole offline archive byte-identical by construction."""
    class _F:
        _ctx = {"ev": object()}
        nongrav_model = None
        radar = {"applied": False}
    f = _F()
    assert fo._final_reclaim(f, [], {}) is f
    f2 = _F(); f2.radar = None
    assert fo._final_reclaim(f2, [], {}) is f2
    f3 = _F(); f3._ctx = None
    assert fo._final_reclaim(f3, [], {}) is f3

def test_mask_recheck_skip_paths_and_arithmetic(monkeypatch):
    """The same-data re-check: Phase A re-derives the dBIC against gravity's own
    rejection fixed point, keeping a disputed obs in evidence iff the model genuinely holds
    it (the 2003 BN4 lesson); Phase B is the raw-visibility floor -- on the same mask the
    model must improve the raw arcsec rms by NONGRAV_MIN_RMS_GAIN (the tell on
    2021 LW3 / 2022 OC3). Offline: rejection + fits faked, routing and arithmetic real."""
    import math
    import pyroorb.arc_extend as ae

    N_OBS = 10

    class _Ev:
        sig_ra = np.ones(N_OBS)
        sig_dec = np.ones(N_OBS)

        def fit(self, state, mask, thin=False, polish=False):
            return np.zeros(6), None

        # the basin-escape floors evaluate the model orbit's 6-state as gravity; return values large
        # enough that the min() floors never bite in these routing tests
        def rms_norm(self, state, mask):
            return 10.0

        def rms_arcsec(self, state, mask):
            return 9.9

    class _Grav:
        _state = np.zeros(6)

    class _ModelFit:
        """Per-obs model residuals: `bad` sigma at the listed indices, 0.1 elsewhere."""
        def __init__(self, bad_at=(), bad=5.0):
            self._bad_at, self._bad = list(bad_at), bad

        def residuals(self):
            rows = [{"d_ra_arcsec": 0.1, "d_dec_arcsec": 0.0} for _ in range(N_OBS)]
            for i in self._bad_at:
                rows[i] = {"d_ra_arcsec": self._bad, "d_dec_arcsec": 0.0}
            return rows

    class _Orb:
        _state = np.zeros(6)

        def __init__(self, chi2, n=18, rms=0.40):
            self._chi2, self._n, self._rms = chi2, n, rms

        def rms_normalized(self):
            return math.sqrt(self._chi2 / self._n)

        def n_resids(self):
            return self._n

        def rms_residual(self):
            return self._rms

    def _sel2(model_fit=None, rms0=0.50, rms_model=0.40):
        return {"choice": "yarkovsky", "grav": _Grav(), "rms0": rms0,
                "models": {"yarkovsky": {"fit": model_fit or _ModelFit(),
                                         "rms": rms_model}}}

    def _queues(build, refine):
        bq, rq = [dict(t) for t in build], [dict(t) for t in refine]
        monkeypatch.setattr(fo, "_build_orbit", lambda *a, **k: _Orb(**bq.pop(0)))
        monkeypatch.setattr(fo, "refine_nongrav", lambda *a, **k: _Orb(**rq.pop(0)))

    monkeypatch.setattr(fo, "_lift_nongrav_state", lambda f, m: None)
    ctx = {"ev": _Ev()}
    mask = np.ones(N_OBS, bool)

    # env kill switch: recorded, never vetoes, nothing else runs
    monkeypatch.setenv("PYROORB_NG_MASK_RECHECK", "0")
    rec, ok = fo._ng_mask_recheck(ctx, _sel2(), mask, 200)
    assert ok is True and rec["skipped"] == "env"
    monkeypatch.delenv("PYROORB_NG_MASK_RECHECK")

    # identical fixed points -> Phase A skipped; the visibility floor reads the retest's own
    # same-mask numbers: a 20% raw gain clears it
    monkeypatch.setattr(ae, "auto_reject_within_arc", lambda ev, m, s: 0)
    rec, ok = fo._ng_mask_recheck(ctx, _sel2(rms0=0.50, rms_model=0.40), mask, 200)
    assert ok is True and rec["skipped"] == "masks agree"
    assert abs(rec["rms_gain"] - 0.20) < 1e-9

    # ...a 1.5% raw gain does not (the LW3/OC3 tell): the model explains nothing visible
    rec, ok = fo._ng_mask_recheck(ctx, _sel2(rms0=0.407, rms_model=0.401), mask, 200)
    assert ok is False and "visible" in rec["why"]
    assert rec["rms_gain"] < fo.NONGRAV_MIN_RMS_GAIN

    # the same collapse on a RADAR object records without vetoing (2003 HM: the real signal
    # is raw-invisible until the joint solve pulls A2 up): the radar joint decides
    ctx_radar = {"ev": _Ev(), "meta": {"radar": [{"kind": "delay"}]}}
    rec, ok = fo._ng_mask_recheck(ctx_radar, _sel2(rms0=0.407, rms_model=0.401), mask, 200)
    assert ok is True and rec["veto_suppressed"] == "radar pending"

    # Phase A: gravity rejects obs 0 and the model cannot hold it either (5 sigma -- the
    # LW3 shape): it leaves evidence, the shared-data dBIC falls short, caller reverts
    def _drop_one(ev, m, s):
        if m[0]:
            m[0] = False
            return 1
        return 0
    monkeypatch.setattr(ae, "auto_reject_within_arc", _drop_one)
    _queues(build=[{"chi2": 140.0}], refine=[{"chi2": 128.0}])
    rec, ok = fo._ng_mask_recheck(ctx, _sel2(_ModelFit(bad_at=[0])), mask, 200)
    assert ok is False and "mask" in rec["why"]
    assert rec["disputed"] == 1 and rec["disputed_kept"] == 0
    assert rec["dbic"] < fo.NONGRAV_BIC_MARGIN

    # Phase A passes on shared data -> the visibility floor reads the Phase A refits
    _queues(build=[{"chi2": 140.0, "rms": 0.50}], refine=[{"chi2": 120.0, "rms": 0.40}])
    rec, ok = fo._ng_mask_recheck(ctx, _sel2(_ModelFit(bad_at=[0])), mask, 200)
    assert ok is True
    assert abs(rec["dbic"] - (20.0 - math.log(18.0))) < 1e-3   # record rounds to 3dp
    assert abs(rec["rms_gain"] - 0.20) < 1e-9

    # ...and vetoes there too when the raw gain is invisible even though the dBIC clears
    _queues(build=[{"chi2": 140.0, "rms": 0.402}], refine=[{"chi2": 120.0, "rms": 0.400}])
    rec, ok = fo._ng_mask_recheck(ctx, _sel2(_ModelFit(bad_at=[0])), mask, 200)
    assert ok is False and "visible" in rec["why"] and rec["dbic"] >= fo.NONGRAV_BIC_MARGIN

    # a disputed obs the model holds stays in evidence -> comparison equals the retest,
    # and the floor then reads the retest numbers (the BN4 routing)
    rec, ok = fo._ng_mask_recheck(ctx, _sel2(_ModelFit(bad_at=[0], bad=0.1)), mask, 200)
    assert ok is True and rec["skipped"] == "model holds every disputed obs"
    assert rec["disputed"] == 1 and rec["disputed_kept"] == 1 and rec["rms_gain"] == 0.2

    # a recheck that cannot run must not veto the detection
    class _BadEv:
        def fit(self, *a, **k):
            raise RuntimeError("boom")
    rec, ok = fo._ng_mask_recheck({"ev": _BadEv()}, _sel2(), mask, 200)
    assert ok is True and rec["skipped"].startswith("RuntimeError")


def test_delivered_verification_chapters(monkeypatch):
    """The unified post-delivery verification -- necessity (pinned-refit dBIC),
    visibility (the raw-rms floor), significance (recorded), robustness (delete-1
    jackknife, consulted only in the marginal band). Offline: the pinned refit and the
    rebuild are faked, the routing and arithmetic are real."""
    import math
    import types

    N_OBS, M = 10, 20

    class _Ev:
        times_tdb = np.arange(float(N_OBS))
        Robs = [np.zeros(3)] * N_OBS
        ra = np.zeros(N_OBS); dec = np.zeros(N_OBS); cosd = np.ones(N_OBS)
        fit_sig_ra = np.ones(N_OBS); fit_sig_dec = np.ones(N_OBS)
        nongrav = ("active", {})
        rms_g = 0.50

        def rms_arcsec(self, st, mask):
            return self.rms_g

    class _F:
        nongrav_model = "yarkovsky"
        nongrav_params = {"A2": 1e-13}
        nongrav_sigmas = {"A2": 1e-14}
        history = None
        radar = None

        def __init__(self, ev, chi2_model=20.0, rms=0.40, jk=None):
            self._ctx = {"ev": ev, "backend": object(), "epoch_tdb": 0.0, "meta": {}}
            self._mask = np.ones(N_OBS, bool)
            self._state = np.zeros(7)
            self._res = types.SimpleNamespace(fun=np.full(M, math.sqrt(chi2_model / M)))
            self._jk = {} if jk is None else jk
            self._rms = rms
            self.nongrav_discovery = {"asteroid": {}}

        def rms_residual(self):
            return self._rms

        def rms_normalized(self):
            return 0.70

        def nongrav_jackknife_detail(self):
            if self._jk == "BOOM":
                raise AssertionError("jackknife consulted where it should be skipped")
            return self._jk

    def _fake_factory(chi2_grav):
        def factory(backend, t0, tt, Rs, ra, dec, cosd, sra, sdec, gr, hA,
                    a_of_p, free_cols, col_scale, m):
            r = lambda p: np.full(M, math.sqrt(chi2_grav / M))
            j = lambda p: np.zeros((M, 6))
            return r, j, None
        return factory

    class _Orb:
        def rms_normalized(self):
            return 0.69

    monkeypatch.setattr(fo, "_build_orbit", lambda *a, **k: _Orb())

    def run(chi2_grav, ev_rms_g=0.50, jk=None, radar=None):
        ev = _Ev(); ev.rms_g = ev_rms_g
        f = _F(ev, jk=jk)
        monkeypatch.setattr(fo, "_make_ng_residual", _fake_factory(chi2_grav))
        out = fo._delivered_verification(f, radar or [], verbose=False)
        ver = f.nongrav_discovery["asteroid"].get("verification") or {}
        return out is f, ver

    ln = math.log(M)
    # radar adjudicates: nothing else runs
    kept, ver = run(60.0, radar=[{"kind": "delay"}])
    assert kept and ver == {"skipped": "radar adjudicates"}
    # necessity: dBIC (30-20-ln20 ~ 7) under the margin -> reverted
    kept, ver = run(30.0)
    assert not kept and "necessity" in ver["verdict"]
    # visibility: dBIC clears (60-20-ln20 ~ 37) but raw gain 2.4% < 8% -> reverted
    kept, ver = run(60.0, ev_rms_g=0.41)
    assert not kept and "visibility" in ver["verdict"]
    # robustness: marginal band (37 < 60), every assessed axis spans zero -> reverted
    kept, ver = run(60.0, jk={"A2": {"assessed": True, "spans_zero": True, "z": 0.4}})
    assert not kept and "robustness" in ver["verdict"]
    # ...sign survives -> pass, with all three chapters on record
    kept, ver = run(60.0, jk={"A2": {"assessed": True, "spans_zero": False, "z": 9.0}})
    assert kept and ver["verdict"] == "pass"
    assert ver["necessity"]["dbic"] > fo.NONGRAV_BIC_MARGIN and ver["significance"]["z_joint"] > 6
    # unassessable jackknife is recorded, never a veto
    kept, ver = run(60.0, jk={})
    assert kept and ver["robustness"] == {"not_assessed": True}
    # overwhelming margin (dBIC >= 4x): the jackknife is not even consulted
    kept, ver = run(20.0 + 4 * fo.NONGRAV_BIC_MARGIN + ln + 1.0, jk="BOOM")
    assert kept and "skipped" in ver["robustness"] and ver["verdict"] == "pass"


def test_delivered_verification_fails_loud(monkeypatch):
    """Hardening: an exception inside the gate is RECORDED, never swallowed.

    A bare swallow is the gate failing OPEN -- any future bug in the verification would
    silently deliver every model unverified, which is the exact failure mode the gate
    exists to prevent, one door down. The model is still delivered (an erroring
    instrument is not evidence against the detection), the evaluator keeps its active
    force model, and `verification.error` carries what broke."""
    N_OBS = 4

    class _Ev:
        times_tdb = np.arange(float(N_OBS))
        Robs = [np.zeros(3)] * N_OBS
        ra = np.zeros(N_OBS); dec = np.zeros(N_OBS); cosd = np.ones(N_OBS)
        fit_sig_ra = np.ones(N_OBS); fit_sig_dec = np.ones(N_OBS)
        nongrav = ("active", {})

    class _F:
        nongrav_model = "yarkovsky"
        nongrav_params = {"A2": 1e-13}
        nongrav_sigmas = {"A2": 1e-14}
        radar = None

        def __init__(self):
            self._ctx = {"ev": _Ev(), "backend": object(), "epoch_tdb": 0.0, "meta": {}}
            self._mask = np.ones(N_OBS, bool)
            self._state = np.zeros(7)
            self.nongrav_discovery = {"asteroid": {}}

    def boom(*a, **k):
        raise RuntimeError("synthetic gate bug")

    monkeypatch.setattr(fo, "_make_ng_residual", boom)
    f = _F()
    out = fo._delivered_verification(f, [], verbose=False)
    assert out is f                                        # the model is still delivered...
    ver = f.nongrav_discovery["asteroid"]["verification"]
    assert "RuntimeError: synthetic gate bug" in ver["error"]   # ...but the hole is on record
    assert "unverified" in ver["verdict"]
    assert f._ctx["ev"].nongrav == ("active", {})          # the ev's force model survives


def test_keep_better_gravity_fails_loud():
    """Same principle one door down: the keep-the-better polish is best-effort by design
    (the original fit legitimately stands on any failure), but the failure must land on
    the discovery info rather than vanish -- a swallow would hide a real bug in the
    seeded polish forever."""
    class _Ev:
        def fit(self, *a, **k):
            raise RuntimeError("polish exploded")

    class _Fit:
        _mask = np.ones(3, bool)

        def rms_normalized(self):
            return 1.0

    info = {}
    fit = _Fit()
    out = fo._keep_better_gravity({"ev": _Ev()}, fit,
                                  {"grav_floor_state": np.zeros(6)}, info)
    assert out is fit
    assert "RuntimeError: polish exploded" in info["kept_better_gravity"]["error"]
