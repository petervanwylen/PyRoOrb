"""The delivered verification's OUTCOME ROUTING and how an unverified delivery is announced.
Offline and pure -- the three chapters themselves need a converged fit and live in
the fit-driven tests; what is here is the decision they feed and the channel it speaks through.

The bug this pins: the gate wrote its failing verdict, then tried to rebuild the pinned-A=0
gravity orbit, and if that rebuild came back materially WORSE it fell out of the function
keeping the model -- with `applied` still True. The record then read "necessity: dBIC 3.2 < 15"
next to "applied: true", and nothing anywhere named the refusal, so a delivery that failed
verification was indistinguishable from one that passed.
"""
import numpy as np
import pytest

import pyroorb.fit_object as fo


# --- the routing ---------------------------------------------------------------------------
def test_a_clean_pass_says_nothing():
    outcome, note = fo._verification_outcome(None, 1.7, 1.0)
    assert outcome == "pass" and note is None      # gravity worse is irrelevant when nothing failed


def test_a_failed_chapter_with_a_sane_gravity_orbit_reverts():
    outcome, note = fo._verification_outcome("necessity: dBIC 3.2 < 15", 1.01, 1.00)
    assert outcome == "revert"
    assert "reverted to gravity" in note and "dBIC 3.2" in note


def test_the_1_05_net_is_noise_tolerance_not_a_second_verdict():
    """Two orbits within 5% of each other in normalized rms are the same orbit; the net only
    exists to refuse a CATASTROPHIC rebuild (the wrong-basin class)."""
    why = "necessity: dBIC 3.2 < 15"
    assert fo._verification_outcome(why, 1.0499, 1.0)[0] == "revert"
    assert fo._verification_outcome(why, 1.05, 1.0)[0] == "revert"        # inclusive
    assert fo._verification_outcome(why, 1.0501, 1.0)[0] == "unverified"
    assert fo.VERIFICATION_REVERT_MAX_FACTOR == 1.05


def test_a_refused_revert_is_named_unverified_not_left_silent():
    """THE REGRESSION. A wrong-basin rebuild (6 sigma against the model's 1) must not revert --
    but it must not deliver the model looking clean either."""
    outcome, note = fo._verification_outcome("necessity: dBIC 3.2 < 15", 6.0, 1.0)
    assert outcome == "unverified"
    assert "UNVERIFIED" in note                    # the state has a name
    assert "dBIC 3.2" in note                      # ...and still says WHICH chapter failed
    assert "6.00" in note and "1.00" in note       # ...and why reverting was refused
    assert "provisional" in note


def test_a_rebuild_that_produced_no_number_is_unverified_never_a_revert():
    """A NaN rms is not a demonstration of anything. The bare `<=` comparison this replaced
    already behaved this way (any comparison with NaN is False); it is kept deliberately, so
    the model is delivered under caution rather than replaced by an orbit with no rms."""
    why = "robustness: every assessed coefficient's cloud spans zero"
    assert fo._verification_outcome(why, float("nan"), 1.0)[0] == "unverified"
    assert fo._verification_outcome(why, 1.0, float("nan"))[0] == "unverified"
    assert fo._verification_outcome(why, np.inf, 1.0)[0] == "unverified"


# --- the channel it speaks through -----------------------------------------------------------
# `nongrav_discovery` is a diagnostic record NOTHING renders (grep: it is written in
# fit_object and read nowhere else), so an unverified delivery that only landed there would
# reach the phase-2 scorer and no human. `nongrav_warning` is the advisory string the CLI
# prints and the web force-model panel shows, and it may already hold refine_nongrav's
# physical-plausibility text -- so the two have to coexist.
def test_a_note_lands_on_an_empty_warning_as_a_sentence():
    assert fo._ng_warn_append(None, "the model is UNVERIFIED") == "The model is UNVERIFIED."
    assert fo._ng_warn_append("", "the model is UNVERIFIED") == "The model is UNVERIFIED."


def test_a_note_never_displaces_the_physical_plausibility_text():
    existing = "The selected non-gravitational model has values worth a second look: A1 is negative."
    out = fo._ng_warn_append(existing, "the model is UNVERIFIED")
    assert out.startswith(existing) and out.endswith("The model is UNVERIFIED.")


def test_appending_the_same_note_twice_does_not_repeat_it():
    """The tail can run more than once on one orbit (the CLI's _finalize and, on a restored
    session, a re-display); the banner must not grow a copy each time."""
    once = fo._ng_warn_append("Existing.", "the model is UNVERIFIED")
    assert fo._ng_warn_append(once, "the model is UNVERIFIED") == once


def test_an_empty_note_is_a_no_op():
    assert fo._ng_warn_append("Existing.", "   ") == "Existing."
    assert fo._ng_warn_append(None, "") is None


def test_the_banner_writer_cannot_itself_raise():
    """One of its two callers IS the gate's exception handler. A handler that can raise on
    its own account turns "fail loud" into "die silent" -- which is what happened the first
    time: it read `f.nongrav_warning` directly and an object without the attribute (a test
    double, and by construction anything that is not a FittedOrbit) blew up INSIDE the
    except-branch, escaping the gate entirely."""
    class _NoAttr:
        __slots__ = ()

    fo._ng_warn_on(_NoAttr(), "the model is UNVERIFIED")     # must not raise

    class _Plain:
        pass

    p = _Plain()                                             # no nongrav_warning yet
    fo._ng_warn_on(p, "the model is UNVERIFIED")
    assert p.nongrav_warning == "The model is UNVERIFIED."


# --- the web surface's copy of the tail --------------------------------------------------------
class _Orbit:
    """Just enough of a FittedOrbit for the tail: the model it carries and the advisory."""
    def __init__(self, model="yarkovsky", warning=None):
        self.nongrav_model = model
        self.nongrav_warning = warning


def _tail(monkeypatch, reclaim, verify):
    import pyroorb.web.engine_api as api
    monkeypatch.setattr(api.fo, "_final_reclaim", reclaim, raising=False)
    monkeypatch.setattr(api.fo, "_delivered_verification", verify, raising=False)
    return api


def test_web_tail_reports_the_model_the_delivered_orbit_actually_carries(monkeypatch):
    """The verification may REVERT to gravity; `auto_model` drives the model dropdown, so it
    is read off the orbit that comes back -- never left holding the pre-tail model."""
    api = _tail(monkeypatch,
                lambda f, rows, oc: f,
                lambda f, rows: _Orbit(model=None))
    display, auto = api.post_discovery_tail(_Orbit("yarkovsky"), {})
    assert auto is None and display.nongrav_model is None


def test_web_tail_does_not_swallow_a_verification_failure(monkeypatch):
    """THE REGRESSION. A blanket `except Exception: pass` around the pair put back, on the
    surface most people look at, exactly the silent swallow _delivered_verification's
    docstring is emphatic about."""
    api = _tail(monkeypatch,
                lambda f, rows, oc: f,
                lambda f, rows: (_ for _ in ()).throw(ValueError("boom")))
    display, auto = api.post_discovery_tail(_Orbit("yarkovsky"), {})
    assert auto == "yarkovsky"                         # an erroring instrument is not evidence
    assert "UNVERIFIED" in display.nongrav_warning     # ...but it is not a clean pass either
    assert "ValueError: boom" in display.nongrav_warning


def test_web_tail_reports_a_failed_reclaim_and_still_verifies(monkeypatch):
    """A reclaim that raises must not skip the verification with it -- and the orbit on screen
    is then the pre-reclaim one, which the user is told."""
    seen = {}

    def _verify(f, rows):
        seen["ran"] = True
        return f

    api = _tail(monkeypatch,
                lambda f, rows, oc: (_ for _ in ()).throw(RuntimeError("nope")),
                _verify)
    display, auto = api.post_discovery_tail(_Orbit("yarkovsky"), {})
    assert seen.get("ran") is True
    assert auto == "yarkovsky"
    assert "pre-reclaim" in display.nongrav_warning and "RuntimeError: nope" in display.nongrav_warning


def test_web_tail_is_quiet_when_both_steps_succeed(monkeypatch):
    api = _tail(monkeypatch, lambda f, rows, oc: f, lambda f, rows: f)
    display, auto = api.post_discovery_tail(_Orbit("yarkovsky"), {})
    assert auto == "yarkovsky" and display.nongrav_warning is None


def test_the_tail_keeps_the_radar_the_fit_used(monkeypatch):
    """The discovered orbit is radar-refined and the reclaim and verification see the rows,
    as in _finalize. Without them Bennu's Yarkovsky was judged optically and reverted (dBIC
    11.4 < 15) where the radar solve keeps it (A2 -4.40e-14)."""
    import pyroorb.radar as radar
    seen = {}
    rows = [{"kind": "delay"}]
    monkeypatch.setattr(radar, "refine_radar",
                        lambda f, r, oc, verbose=False: (seen.setdefault("radar", r), f)[1])
    api = _tail(monkeypatch,
                lambda f, r, oc: (seen.setdefault("reclaim", r), f)[1],
                lambda f, r: (seen.setdefault("verify", r), f)[1])
    display, auto = api.post_discovery_tail(_Orbit("yarkovsky"), {}, rows)
    assert seen == {"radar": rows, "reclaim": rows, "verify": rows} and auto == "yarkovsky"
    seen.clear()
    api.post_discovery_tail(_Orbit("yarkovsky"), {})              # no radar: none applied
    assert "radar" not in seen and seen["verify"] == []
