"""The web "Submit linkage" routes: the default sends the
document to the identification queue over HTTP (pyroorb.net posts to itself),
marked ``noextend`` when it came out of an ITF search; "mpc" goes straight to
the MPC only where DIRECT_SUBMIT_ALLOWED is set; the header is always the
person's own name / e-mail -- there is no built-in submitter any more."""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
import pyroorb.web.engine_api as api  # noqa: E402

DOC = {"header": {"name": "", "email": "", "comment": "REPLY_EMAIL"},
       "links": {"link_0": {"designations": ["K16G96F"],
                            "trksubs": [["PR00001", "2020 01 06.1", "F51"]]}}}


@pytest.fixture
def session(monkeypatch):
    """A session carrying a submittable linkage, without a fit behind it."""
    block = {"found": True, "kind": "itf", "submittable": True, "warnings": []}
    monkeypatch.setattr(api, "_linkage_block", lambda s: dict(block))
    monkeypatch.setattr(api, "_itf_live_linkage", lambda s, *a, **k: (
        {"header": dict(DOC["header"]), "links": {"link_0": dict(DOC["links"]["link_0"])}},
        None))
    monkeypatch.setattr(api, "_live_lines", lambda s: [])
    sid = "websubmit-test"
    api._SESSIONS[sid] = {"fit": object(), "itf_pending": {"primaries": []},
                          "itf_warnings": []}
    yield sid
    api._SESSIONS.pop(sid, None)


def test_the_default_route_posts_to_the_queue_with_noextend(session, monkeypatch):
    monkeypatch.setenv("IDQUEUE_URL", "https://example.test/identifications/")
    posted = []

    def poster(doc):
        posted.append(doc)
        return 200, {"ok": True, "document": 7, "token": "tok", "message": "Thank you",
                     "status_url": "/identifications/?document=7&token=tok"}
    got = api.itf_submit_web(session, [], name="J. Smith", email="js@example.org",
                             _poster=poster)
    assert got["ok"] and got["accepted"] and got["route"] == "queue", got
    doc = posted[0]
    assert doc["header"]["name"] == "J. Smith" and doc["header"]["email"] == "js@example.org"
    assert doc["noextend"] is True                 # the ITF search already ran here
    assert doc["noreject"] is True                 # the user clicked through every caution
    assert got["queued"]["status_url"] == \
        "https://example.test/identifications/?document=7&token=tok"
    assert got["queued"]["dashboard"] == "https://example.test/identifications/dashboard"
    assert api._SESSIONS[session]["itf_pending"] is None       # one-shot


def test_an_identification_without_a_search_is_not_marked_noextend(session):
    api._SESSIONS[session]["itf_pending"] = None
    posted = []
    got = api.itf_submit_web(session, [], name="J. Smith", email="js@example.org",
                             _poster=lambda d: posted.append(d) or (200, {"ok": True}))
    assert got["ok"] and "noextend" not in posted[0]
    assert posted[0]["noreject"] is True           # every web submission


def test_a_queue_refusal_is_reported_and_nothing_is_cleared(session):
    got = api.itf_submit_web(session, [], name="J. Smith", email="js@example.org",
                             _poster=lambda d: (503, {"ok": False, "error": "db down"}))
    assert not got["ok"] and "db down" in got["error"]
    assert api._SESSIONS[session]["itf_pending"] is not None


def test_a_submission_needs_a_name_and_email(session):
    got = api.itf_submit_web(session, [], name="", email="js@example.org",
                             _poster=lambda d: (200, {"ok": True}))
    assert not got["ok"] and got.get("need_identity")


def test_direct_to_mpc_needs_the_switch(session, monkeypatch):
    monkeypatch.delenv("DIRECT_SUBMIT_ALLOWED", raising=False)
    sent = []
    got = api.itf_submit_web(session, [], name="J. Smith", email="js@example.org",
                             route="mpc", _submitter=lambda *a: sent.append(a))
    assert not got["ok"] and "DIRECT_SUBMIT_ALLOWED" in got["error"] and not sent
    monkeypatch.setenv("DIRECT_SUBMIT_ALLOWED", "True")
    got = api.itf_submit_web(session, [], name="J. Smith", email="js@example.org",
                             route="mpc",
                             _submitter=lambda doc, cfg, log: sent.append(doc) or
                             (True, "200 OK", "Thank you"))
    assert got["ok"] and got["accepted"] and got["route"] == "mpc"
    assert sent[0]["header"]["name"] == "J. Smith"
    assert "noextend" not in sent[0]               # never sent to the MPC


def test_no_built_in_submitter(monkeypatch):
    for var in ("ITF_SUBMIT_NAME", "ITF_SUBMIT_EMAIL"):
        monkeypatch.delenv(var, raising=False)
    from pyroorb.itf.config import ItfConfig
    from pyroorb.itf import dbio
    monkeypatch.setattr(dbio, "env_default", lambda name, fallback: os.environ.get(name, fallback))
    monkeypatch.setattr(dbio, "load_dotenv", lambda *a, **k: None)
    cfg = ItfConfig()
    assert cfg.submit_name == "" and cfg.submit_email == ""


def test_submit_linkage_refuses_an_anonymous_document(monkeypatch):
    from pyroorb.itf.config import ItfConfig
    from pyroorb.itf import linkage
    monkeypatch.setattr(linkage, "_family_map", lambda conn, unp: {d: d for d in unp})
    import pyroorb.itf.dbio as dbio

    class _C:
        def close(self):
            pass
    monkeypatch.setattr(dbio, "connect_pub", lambda cfg: _C())
    doc = {"header": {"name": "", "email": ""},
           "links": {"link_0": {"designations": ["K16G96F", "K15A53T"]}}}
    sent, status, _m = linkage.submit_linkage(doc, ItfConfig(submit=True), do_submit=True,
                                              log=lambda *_: None)
    assert not sent and "no submitter" in status


def test_a_redirecting_queue_address_is_refused_not_followed(monkeypatch):
    # pyroorb.net 301s to https://www.pyroorb.net (path dropped); requests would
    # re-send the POST as a body-less GET and get the home page back
    import requests
    seen = {}

    class R:
        status_code, headers, text = 301, {"Location": "https://www.pyroorb.net"}, ""

    def post(url, **kw):
        seen.update(kw)
        return R()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setenv("IDQUEUE_URL", "https://pyroorb.net/identifications/")
    code, body = api._post_to_queue({"header": {"name": "J. Doe", "email": "j@example.org"},
                                     "links": {}})
    assert seen["allow_redirects"] is False
    assert code == 301 and not body["ok"]
    assert "redirects to https://www.pyroorb.net" in body["error"]


def test_an_unsigned_document_never_leaves_for_the_queue(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("it was sent"))
    code, body = api._post_to_queue({"header": {"name": "", "email": ""},
                                     "links": {"link_0": {"designations": ["K16G96F"]}}})
    assert code == 0 and not body["ok"] and "no submitter" in body["error"]


def test_the_default_queue_address_is_the_one_that_answers(monkeypatch):
    monkeypatch.delenv("IDQUEUE_URL", raising=False)
    assert api._queue_url() == "https://www.pyroorb.net/identifications/"


def test_the_pending_linkage_keeps_the_database_trksub():
    """The live rebuild names each ITF tracklet by the database trksub the
    search saw, not columns 6-12 of its line -- also for a possible, which has
    no ``itf_trksubs`` of its own but whose registry banked the same rows."""
    from types import SimpleNamespace
    from pyroorb.itf.obs80_fields import obs_unique_id
    from pyroorb.itf.obsreg import AdesRegistry
    line = ("     M00000   C2006 10 11.39981 10 00 00.00 +10 00 00.0"
            + " " * 9 + " 21.0G" + " " * 7 + "F51")
    assert len(line) == 80
    reg = AdesRegistry()
    reg.add_records([{"obs80": line, "trksub": "MB00000"}])
    possible = SimpleNamespace(itf_obs80=line + "\n", registry=reg)
    assert api._itf_trksubs(possible) == {obs_unique_id(line): "MB00000"}
    linked = SimpleNamespace(itf_obs80=line + "\n", registry=None,
                             itf_trksubs={obs_unique_id(line): "MB00000"})
    assert api._itf_trksubs(linked) == {obs_unique_id(line): "MB00000"}
