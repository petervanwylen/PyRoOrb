"""The property offline queue tests cannot check: a session restored from a
coalesced fit is a genuinely PRIVATE copy that reproduces the original orbit.

Marked slow -- it runs one real fit (offline, from the frozen fixture).  The
invariant is the reason single-flight is safe: one fit may serve a hundred
viewers only if none of them can reach another's evaluator, mask or state.
"""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow


DESIG = "2023 QJ3"


@pytest.fixture(scope="module", autouse=True)
def _frozen_astrometry():
    """Serve this module's designation from the frozen fixture, never the MPC.

    The fit here goes through the whole web route (`run_fit_queued`), so the
    fetch is several layers down in `fitcache` / `engine_api` -- both of which
    reach it as an `obs_sources` attribute, so pinning it there covers both.
    Without this the module is network-dependent despite calling itself offline:
    the fetch is TTL-gated, and an unreachable MPC returns no rows, so all four
    restore tests error out in `job` with a fit that never had any astrometry.
    See _fixtures.fixture_obs.
    """
    from pyroorb import obs_sources
    from _fixtures import fixture_obs
    real = obs_sources.fetch_observations

    def _fetch(desigs, obscodes, *a, **kw):
        want = [desigs] if isinstance(desigs, str) else list(desigs)
        if want == [DESIG]:
            obs, meta = fixture_obs("2023_QJ3")
            return [dict(o) for o in obs], dict(meta)
        return real(desigs, obscodes, *a, **kw)

    obs_sources.fetch_observations = _fetch
    yield
    obs_sources.fetch_observations = real


@pytest.fixture(scope="module", autouse=True)
def _isolated_disk_cache(tmp_path_factory):
    """Point the disk tier at a directory of this module's own.

    Every test here needs a LIVE job with a template session behind it, and a
    disk hit deliberately produces neither -- it answers with a payload and a
    `refit`, no job id.  Sharing $TMPDIR with the real cache therefore turns the
    whole module into `wait(None)` the moment any earlier run has fitted this
    designation, which is exactly how it failed: four errors, all
    "fit never finished", all because the fit never STARTED.
    """
    from pyroorb.web import fitdisk
    prev_env = os.environ.get("PYROORB_FIT_CACHE_DIR")
    os.environ["PYROORB_FIT_CACHE_DIR"] = str(
        tmp_path_factory.mktemp("restore-fitcache"))
    fitdisk._DIR = None
    fitdisk._DISABLED_REASON = None
    yield
    if prev_env is None:
        os.environ.pop("PYROORB_FIT_CACHE_DIR", None)
    else:
        os.environ["PYROORB_FIT_CACHE_DIR"] = prev_env
    fitdisk._DIR = None
    fitdisk._DISABLED_REASON = None


@pytest.fixture(scope="module")
def job():
    import pyroorb.web.engine_api as E
    import pyroorb.web.fitqueue as fq
    started = E.run_fit_queued(DESIG)
    assert started.get("job"), (
        "no job id -- the fit was answered from cache, so there is no template "
        f"to restore from: {started}")
    assert started.get("ok"), started
    assert fq.wait(started["job"], 900), "fit never finished"
    out = E.fit_job_status(started["job"])
    assert out["result"].get("ok"), out["result"].get("error")
    return started["job"], out["result"]


def test_payload_carries_a_restore_token_and_no_session(job):
    _jid, payload = job
    assert "session" not in payload, "the shared session leaked to a client"
    assert payload.get("restore"), "no restore token -- interaction impossible"


def test_concurrent_requests_coalesce_to_one_job():
    import pyroorb.web.engine_api as E
    jobs = {E.run_fit_queued(DESIG)["job"] for _ in range(5)}
    assert len(jobs) == 1, f"single-flight failed: {len(jobs)} jobs"


def test_each_claim_is_a_distinct_private_session(job):
    import pyroorb.web.engine_api as E
    import pyroorb.web.fitqueue as fq
    jid, _ = job
    sids = []
    for _ in range(3):
        out = E.session_from_job(jid)
        assert out.get("ok"), out
        sids.append(out["session"])
    assert len(set(sids)) == 3
    tmpl = fq.retained(jid)
    for sid in sids:
        s = E._SESSIONS[sid]
        assert s["grav_fit"] is not tmpl["grav_fit"]
        assert s["grav_fit"]._ctx is not tmpl["grav_fit"]._ctx
        assert s["grav_fit"]._ctx["ev"] is not tmpl["grav_fit"]._ctx["ev"]
        assert s["history"] is not tmpl["history"]


def test_restored_session_reproduces_the_template_orbit(job):
    """A rebuilt evaluator is only safe if it is the SAME evaluator numerically
    -- otherwise a restored session would quietly show a different fit."""
    import pyroorb.web.engine_api as E
    import pyroorb.web.fitqueue as fq
    jid, _ = job
    tmpl = fq.retained(jid)
    s = E._SESSIONS[E.session_from_job(jid)["session"]]
    ra_t, dec_t = tmpl["grav_fit"]._ctx["ev"].resid_arcsec(tmpl["state"])
    ra_p, dec_p = s["grav_fit"]._ctx["ev"].resid_arcsec(s["state"])
    assert np.allclose(ra_t, ra_p, atol=1e-9) and np.allclose(dec_t, dec_p, atol=1e-9)


def test_a_step_in_one_session_does_not_touch_another_or_the_template(job):
    import pyroorb.web.engine_api as E
    import pyroorb.web.fitqueue as fq
    jid, _ = job
    tmpl = fq.retained(jid)
    before = np.asarray(tmpl["state"], dtype=float).copy()
    a = E.session_from_job(jid)["session"]
    b = E.session_from_job(jid)["session"]
    mask = [bool(m) for m in E._SESSIONS[a]["grav_fit"]._mask]
    mask[0] = False                                  # exclude an observation and re-solve
    out = E.full_step(a, mask, single=True)
    assert out.get("ok") is not False, out
    assert np.allclose(np.asarray(tmpl["state"], dtype=float), before), \
        "the template was mutated by a client's step"
    assert np.allclose(E._SESSIONS[b]["state"], before), \
        "one client's step changed another client's session"
