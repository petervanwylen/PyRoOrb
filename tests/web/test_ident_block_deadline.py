"""The identification block is the last thing a web fit's payload builds. When the fit's own bound
(fitqueue's "this fit" deadline) runs out INSIDE the identification check -- a solo baseline fit
hands the caller's deadline on rather than absorbing it -- the finished fit must still be shown,
without a verdict, not thrown away as a timed-out fit."""
from pyroorb import deadline, ident_check
from pyroorb.web import engine_api


class _Fit:
    n_requested = 2
    _obs = [{"designation": "2014 AA1"}, {"designation": "2015 BB2"}]


def test_a_deadline_inside_the_ident_check_drops_only_the_block(monkeypatch):
    def out_of_time(fit):
        raise deadline.DeadlineExceeded("this fit", 180.0, 181.0)

    monkeypatch.setattr(engine_api, "_itf_live_linkage", lambda s: (None, "unused"))
    monkeypatch.setattr(ident_check, "linkage_verdict", out_of_time)
    block = engine_api._ident_linkage_block({"fit": _Fit()})
    assert block["found"] is False and block["submittable"] is False
    assert "ran out of this fit's time" in block["reason"]


def test_a_cancel_is_not_absorbed(monkeypatch):
    """Only the deadline is the block's to absorb; a cancel must still unwind the work."""
    def cancelled(fit):
        raise deadline.Cancelled("this fit")

    monkeypatch.setattr(engine_api, "_itf_live_linkage", lambda s: (None, "unused"))
    monkeypatch.setattr(ident_check, "linkage_verdict", cancelled)
    try:
        engine_api._ident_linkage_block({"fit": _Fit()})
    except deadline.Cancelled:
        return
    raise AssertionError("a cancel was absorbed")
