"""ITF recovery regression against FROZEN astrometry, with the LIVE ITF as
optional distractors.

Each fixture in ``itf_fixtures/`` carries one object's verbatim published OBS80.
The harness re-parses it (deterministic seed, no network), plants the object's
other apparitions as fake ITF, and searches.  When the SBN database is reachable
the live ITF supplies REAL tracklets as distractors beside the planted ones --
so the search must still recover the truth without being pulled off by real
neighbours.  Both modes must recover every planted apparition; the resulting
orbit must stay under the fixture's rms cap.

    pytest tests/test_itf_recovery.py           # offline: deterministic, no DB
    pytest tests/test_itf_recovery.py --live    # + the live ITF as distractors

RUN THE LIVE MODE before trusting a recall-affecting change.  Offline is
deterministic and fast, but it cannot see anything that lives on the DB side of
the search -- the sweep row budget, and (2026-07-23) the variant scan's union
sweep spending the whole no-link budget before variant #1, which failed 2015
BE559 live while offline stayed green.  A green offline run is not evidence
about the live pipeline.

Offline is the DEFAULT here.  It used to be the other way round -- the module
tried the live DB first and fell back on any exception -- which meant a plain
suite run sat for many minutes on connection timeouts before quietly testing
something other than what it said.  The live pass is now its own `live`-marked
test, so only `--live` selects it and its failures are attributable.
"""
import pytest

from pyroorb.obs_sources import load_obscodes
from pyroorb.itf.config import ItfConfig
from pyroorb.itf.harness import run_recovery

from _itf_fixtures import iter_fixtures, load_fixture, load_obs

_FIXTURES = iter_fixtures()


def _label(path):
    return load_fixture(path)["designation"]


def _live_source(cfg):
    """A live DbSource (real ITF distractors); SKIP if the DB is unreachable --
    an unreachable database is not a recovery regression."""
    try:
        from pyroorb.itf.candidates import DbSource
        from pyroorb.itf.dbio import connect_pub
        return DbSource(connect_pub(cfg), cfg, reconnect=lambda: connect_pub(cfg))
    except Exception as e:
        pytest.skip(f"SBN database unreachable: {type(e).__name__}: {e}")


def check_recovery(path, base_source, cfg_overrides=None):
    """Re-fit one frozen fixture, plant its other apparitions as fake ITF, search,
    and return (ok, report_line). Shared by the offline and live passes."""
    oc = load_obscodes()
    # The production config, deliberately: a fixture run that needs a special
    # budget is not measuring what production does.  The load-sensitivity that
    # once cut 2024 PQ7 to 0/4 in a busy suite is fixed where it belongs -- in
    # cfg.min_fruitless_fits, which makes expiry depend on WORK burned as well
    # as on the clock (see SearchBudget).
    # PYROORB_ITF_MODEL points the ranker at an alternative artifact for a
    # model A/B (see training_data/README.md); unset = the deployed default.
    import os
    cfg = ItfConfig(ml_model_path=os.environ.get("PYROORB_ITF_MODEL") or None,
                    **(cfg_overrides or {}))
    fx = load_fixture(path)
    obs, meta = load_obs(fx, oc)
    rep = run_recovery(fx["designation"], cfg=cfg, base_source=base_source,
                       obscodes=oc, obs_meta=(obs, meta),
                       # PIN the seed apparition the fixture was built on.
                       # The fixture has always recorded it, but the run
                       # used to re-derive it from run_recovery's default --
                       # so changing that default silently re-seeded every
                       # frozen fixture, with no diff to show for it.
                       seed_apparition=(None if fx.get("compact_seed")
                                        else fx.get("seed_apparition")),
                       seed_nights=fx.get("seed_nights"),
                       # the scan's compact-N-night seed, re-derived from the
                       # FROZEN astrometry (deterministic)
                       compact_seed=bool(fx.get("compact_seed")),
                       hold_out_stations=tuple(fx.get("hold_out_stations", ())),
                       max_planted=fx.get("max_planted"),
                       seed_oppositions=fx.get("seed_oppositions", 1),
                       log=(lambda *a: None))
    exp = fx["expect"]
    # apparitions a fixture explicitly allows the search to leave out (a
    # per-object decision recorded in its note); every other one is required
    planted = set(fx["planted_apparitions"]) - set(exp.get("missing_apparitions", ()))
    recovered = set(rep.recovered_apparitions)
    # Most fixtures pin a LINK.  A few pin that the search finds the planted
    # tracklets and reports them as a reviewable POSSIBLE (status 0.6) -- a
    # lone new night, or a linkage the seed check must not let be submitted.
    if exp.get("verdict") == "possible":
        apps_ok = rep.status == 0.6 and recovered >= planted
    else:
        apps_ok = rep.ok and recovered >= planted
    # Every planted tracklet, EXCEPT where the fixture's golden records fewer
    # on purpose: 2016 EZ157 plants one 3-obs tracklet that keeps 2 of its obs,
    # which acceptance is right to refuse (the 75% retention rule) and the
    # search now drops instead of failing the whole linkage over.  The bar is
    # still ">= what this object achieved when the fixture was baked", so a
    # regression that loses one more tracklet fails; a fixture simply may not
    # demand a tracklet the gates correctly reject.
    want_trk = min(exp.get("recovered_tracklets", fx["n_fake_tracklets"]),
                   fx["n_fake_tracklets"])
    trk_ok = rep.recovered_tracklets >= want_trk
    rms_ok = rep.rms <= exp["rms_max"]
    # A recovery ADDS observations; it must not throw the object's own
    # away.  2024 PQ7's linkage is real (the MPC accepted it) and the
    # search clipped three of its nine W76 observations to a flat arcsec cut
    # that ignored the station's uncertainty -- so fixtures pin the drop
    # count, defaulting to zero tolerance.
    drop_ok = rep.dropped_seed_obs <= exp.get("max_dropped_seed_obs", 0)
    ok = apps_ok and trk_ok and rms_ok and drop_ok
    why = "" if ok else (" <- " + ", ".join(
        w for w, c in (("apps", apps_ok), ("tracklets", trk_ok),
                       ("rms", rms_ok), ("dropped-seed-obs", drop_ok)) if not c))
    foreign = f" foreign={len(rep.foreign_tracklets)}" if rep.foreign_tracklets else ""
    if rep.false_foreign:
        foreign += f" FALSE={','.join(rep.false_foreign)}"
    msg = (f"{fx['designation']} U={fx['seed_U']} "
           f"apps={sorted(recovered)}/{sorted(planted)} "
           f"trk={rep.recovered_tracklets}/{fx['n_fake_tracklets']} "
           f"rms={rep.rms:.3f}(<={exp['rms_max']}){foreign}"
           + (f" DROPPED_SEED={rep.dropped_seed_obs}" if rep.dropped_seed_obs else "")
           + why + (f"  |  {rep.reason[:140]}" if not ok else ""))
    return ok, msg


@pytest.mark.slow
@pytest.mark.parametrize("path", _FIXTURES, ids=_label)
def test_recovers_offline(path):
    """Fake-only: the deterministic golden -- every planted apparition must recover
    under the fixture's rms cap. Reproducible anywhere, no DB."""
    ok, msg = check_recovery(path, base_source=None)
    assert ok, msg


# The k-clustering + barren-take backtracking flag set (clustering-idea.md /
# ladder-backtracking.md), as the live BE559 validation ran it.
_LOV_PROFILE_FLAGS = dict(lov_profile=True, lov_profile_order=True,
                       lov_profile_center_singles=2, lov_profile_cluster_trials=2,
                       backtrack_per_round=3)


@pytest.mark.slow
@pytest.mark.parametrize("path",
                         [p for p in _FIXTURES if "BE559" in str(p)],
                         ids=_label)
def test_recovers_offline_with_lov_profile_flags(path):
    """The LOV profile trial sources + barren-take backtracking flag set must keep the
    deterministic golden green on the object the flags were built for."""
    ok, msg = check_recovery(path, base_source=None,
                             cfg_overrides=_LOV_PROFILE_FLAGS)
    assert ok, msg


# The value a consumer arms ``max_night_systematic_sigma`` with (the shipped default is
# 0, off). Any consumer that arms the clause must use this value or re-run this test.
ARMED_NIGHT_SIGMA = 2.0


@pytest.mark.slow
@pytest.mark.parametrize("path", _FIXTURES, ids=_label)
def test_recovers_with_the_night_coherence_clause_armed(path):
    """Every fixture must still recover with ``max_night_systematic_sigma``
    ARMED at ``ARMED_NIGHT_SIGMA``, the value a consumer that arms it uses.

    The shipped engine default is 0 (off) and stays that way -- only an explicit
    consumer arms it -- but a gate that cannot be switched on without vetoing real
    multi-opposition recoveries is not usable in ANY population, and the first
    version of this clause was exactly that: it refused 856215, a 19-night
    5-opposition linkage of 90/91 observations fitting at 0.242", purely for
    having many nights.  This pins the fix.

    Measured 2026-08-16 at this commit: 24/24 recover armed at 2.0, and the
    worst per-fixture ``night_coherence`` the clause saw was 1.84 (2015 KJ19,
    30 nights) -- so the pass is real but the MARGIN is 8%, not the comfortable
    one the threshold comment claims.  If this test starts failing, read that
    number before touching the threshold.
    """
    ok, msg = check_recovery(
        path, base_source=None,
        cfg_overrides={"max_night_systematic_sigma": ARMED_NIGHT_SIGMA})
    assert ok, msg


@pytest.mark.live
@pytest.mark.slow
@pytest.mark.parametrize("path", _FIXTURES, ids=_label)
def test_recovers_against_live_distractors(path):
    """The live ITF merged in as REAL distractors beside the planted fakes -- the
    recovery must survive real neighbours. This is the pass that caught the
    `obs_unique_id` exposure-collision (PJ60's true tracklet was dropped when real
    ITF neighbours shared its G96 exposure)."""
    why = load_fixture(path).get("offline_only")
    if why:
        # A fixture may be a sound OFFLINE guard and still not survive the live
        # pass -- the ranker's scores are per PROCESS, so a long suite run scores
        # a marginal seed differently than a fresh one. Skipping beats both
        # alternatives: deleting a fixture that pins a real fix, and leaving a
        # flapping test that trains everyone to ignore a red live suite.
        pytest.skip(f"offline-only fixture: {why}")
    ok, msg = check_recovery(path, base_source=_live_source(ItfConfig()))
    assert ok, msg
