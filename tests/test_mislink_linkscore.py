"""The mislinkage model as an ITF link scorer (mislink/linkscore.py): it objects at a score
of 0.1 or more, says so with a caveat outside the model's domain, and is silent when the
model cannot score the linkage."""
import pytest

from pyroorb.mislink import linkscore as L
from pyroorb.mislink import score as S


@pytest.fixture
def scored(monkeypatch):
    import pyroorb.obs_sources as O
    monkeypatch.setattr(O, "read_obs_file", lambda text, oc, in_format=None: ([{"x": 1}], "obs80"))
    box = {}
    monkeypatch.setattr(S, "score_observations", lambda obs, desig, oc: dict(box))
    return box


def test_below_the_cut_is_silent(scored):
    scored.update(status="ok", score=0.0999, in_domain=True)
    assert L.objection("text", {}) is None


def test_at_the_cut_objects(scored):
    scored.update(status="ok", score=0.1, in_domain=True, verdict="small chance of mislinkage")
    why = L.objection("text", {})
    assert why and "0.10" in why and "below 0.1" in why and "spurious" not in why


def test_out_of_domain_still_objects_with_a_caveat(scored):
    scored.update(status="ok", score=0.97, in_domain=False, verdict="very likely mislinkage")
    why = L.objection("text", {})
    assert "0.97" in why and "may be spurious" in why and "q >= 1.3 au" in why


@pytest.mark.parametrize("status", ["out-of-scope-nights", "multi-opposition",
                                    "single-detection-nights", "fit-failed", "too-few-obs"])
def test_no_score_no_objection(scored, status):
    """No score: no objection, and no verdict either (a linkage nothing else judged then
    gets the chimera guard)."""
    from pyroorb.itf import linkscore
    scored.update(status=status)
    why = L.objection("text", {})
    assert why is linkscore.NO_VERDICT and not why


def test_ships_as_a_builtin_link_scorer(monkeypatch, tmp_path):
    from pyroorb.itf import linkscore
    assert "pyroorb.mislink.linkscore:SCORER" in linkscore.BUILTIN
    assert L.SCORER is L.objection
    # found with no registry file and no entry point: a plain install judges with it
    monkeypatch.setenv(linkscore.FILE_ENV, str(tmp_path / "none.txt"))
    assert L.objection in linkscore.discover(log=lambda *_: None)
