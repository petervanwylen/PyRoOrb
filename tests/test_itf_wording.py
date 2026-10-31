"""The web UI's plain-English rendering of the ITF search's shorthands
(src/pyroorb/web/itf_wording.py), offline.

The engine counts oppositions fractionally (1.5, 1.75) and reports an outcome
as a number (0, 0.6, 1) so code can compare them; neither means anything to a
person reading a banner or the live trace.  These tests pin both halves of the
contract: every shorthand is spelled out on its way to the browser, and no
OTHER number in the same sentence -- an rms, a U, a night count -- is touched.

    pytest tests/test_itf_wording.py
"""
import os

import re
import sys

import pyroorb.web.itf_wording as W


# every number the engine uses as a code rather than as a measurement
_SHORTHAND = re.compile(r"(?<![\d.])(1\.75|1\.5|0\.6)(?![\d])")


def _clean(text):
    """The property under test: no bare shorthand reaches a reader."""
    return _SHORTHAND.search(text) is None


def test_outcome_labels_are_sentences_not_codes():
    assert W.outcome_label(1) == "submittable linkage"
    assert W.outcome_label(0.6) == "possible linkage (not submittable)"
    assert W.outcome_label(0) == "no linkage"
    # a status the engine never emits still reads as something, never a crash
    assert W.outcome_label(None) == "no linkage"
    assert all(_clean(W.outcome_label(s)) for s in (1, 0.6, 0))


def test_opp_counts_whole_pass_through_fractional_spelled_out():
    assert W.opp_text(1) == "1"
    assert W.opp_text(3) == "3"
    assert W.opp_text(1.5) == "2 (1 single-night)"
    assert W.opp_text(1.75) == "3 (2 single-night)"
    assert W.opp_text(1.5, compact=False) == "2 oppositions, one of them a single night"


def test_reasons_and_trace_lines_lose_the_shorthand():
    for raw in [
        "1.5 opps without progress, abandoning",
        "1.75 opps -- too fragile to submit, abandoning",
        "abandoned (opps 1 -> 1.5, nights 3 -> 4)",
        "opps 1 -> 1.75, nights 4, opposition profile  FAIL",
        'rms 0.593", opps 1.5, nights 3',
        "CONFIRMED: 1.75 opps -> 3",
        "treating 1.75 opps as 3",
        "arc makes the 1.75-opposition profile submittable",
        "[depth 1] 1.75 refused -> falling back",
        "reached 1.75 oppositions at depth 2",
        "status 0.6 -- 1.5 opps without progress",
        "POSSIBLE but unconfirmed linkage (1.75 oppositions): review",
    ]:
        out = W.humanize(raw)
        assert _clean(out), f"{raw!r} -> {out!r}"
        assert "opposition" in out or "single-night" in out or "linkage" in out


def test_measurements_that_merely_look_like_shorthand_are_untouched():
    # an rms, a U and a sigma are DATA -- rewriting them would be a lie
    for raw in [
        'residual rms 1.5" over 12 obs',
        "(U 1.75), 1 opp(s), arc 12 d",
        "seed U 1.50 vs cap 2.00, orig opps 1",
        "candidate at 0.6 sigma of the corridor centre",
        "mag limit 23.5, sweep 1.5 s",
    ]:
        assert W.humanize(raw) == raw


def test_rewriting_is_idempotent():
    # payload warnings are humanized when stashed AND when re-served from the
    # pending block; a second pass must not garble the first
    for raw in ["1.5 opps without progress", "opps 1 -> 1.75, nights 4",
                "status 0.6 -- abandoning"]:
        once = W.humanize(raw)
        assert W.humanize(once) == once


def test_vetting_dict_counts_and_text():
    out = W.humanize_vetting({"opps_pre": 1, "opps_post": 1.75,
                              "rms_post": 0.6, "note": "1.5 opps, one night"})
    assert out["opps_pre"] == "1"
    assert out["opps_post"] == "3 (2 single-night)"
    assert out["rms_post"] == 0.6          # a measurement, left numeric
    assert _clean(out["note"])


def test_humanize_all_preserves_shape():
    assert W.humanize_all(None) == []
    assert W.humanize_all(["a", "1.5 opps"])[0] == "a"
    assert len(W.humanize_all(["1.5 opps", "1.75 opps"])) == 2


def test_the_rejection_ledgers_named_causes_humanize_cleanly():
    """Since the ledger names the clause that refused (search._acceptance_
    failures) rather than blaming rms_max, its reasons carry chi2s, obs counts
    and tracklet tags alongside the opposition shorthand.  None of that may be
    mistaken for a shorthand, and the shorthand must still be spelled out."""
    for raw in [
        'too few nights/oppositions to submit; seed chi2 5.3 -> 34.0 '
        '(Delta 28.7 vs 21.1, 6 dof + 25% of seed) '
        '(overall rms 0.553", reported, not gated)',
        'seed chi2 5.3 -> 34.0 (Delta 28.7 vs 21.1, 6 dof + 25% of seed) '
        '(opps 1 -> 1.5, nights 3, rms 0.553")',
        'adds nothing to the seed fit (used 13 of 13, seed 13 of 13) '
        '(opps 1 -> 1.75, nights 4, rms 0.401")',
        'tracklet H326348@T09/2016-04-02(9obs): 3/4 obs on (75% vs 75%) '
        '(overall rms 0.610", reported, not gated)',
    ]:
        out = W.humanize(raw)
        assert _clean(out), f"{raw!r} -> {out!r}"
        assert W.humanize(out) == out, "not idempotent"
    # the measurements inside those reasons survive untouched
    out = W.humanize('seed chi2 5.3 -> 34.0 (Delta 28.7 vs 21.1, 6 dof + 25% '
                     'of seed) (overall rms 0.553", reported, not gated)')
    assert '0.553' in out and '28.7' in out and 'not gated' in out
