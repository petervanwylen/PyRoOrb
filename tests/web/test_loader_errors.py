"""The paste loader's error boundary: whose fault was it?

`loader.fit_pasted` / `loader.add_to_fit` used to wrap the parse call in `except
Exception` and report everything as "could not parse pasted obs: <TypeError>: ...".
That conflates two very different failures:

  * MALFORMED INPUT -- the user pasted something we can't read. A user-facing error
    is exactly right.
  * OUR BUG -- an internal call that no longer matches its signature, a bad attribute,
    anything else. Reporting it as a parse failure blames the user for our defect, and
    it hides the fault in tests: tests/test_arc_order.py's stale `_parse` stub raised
    TypeError for months and the test failed with the unrelated message "fit_obs_list
    was never called".

So the parsing boundary (loader._parse) names the first kind -- obs_sources.ObsParseError
-- and the handlers catch only that. Everything else propagates to the caller's error
boundary (web/server.py already wraps every route, so production still answers with an
error payload rather than an empty 500 -- it just says the truth about what happened).

    python web/tests/test_loader_errors.py    # plain runner, exits 1 on failure
    python -m pytest web/tests/test_loader_errors.py
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
from pyroorb import obs_sources
from pyroorb.web import loader
from pyroorb.web import engine_api


def _obscodes():
    return engine_api._obscodes()


def test_malformed_xml_is_a_user_facing_parse_error():
    """Genuinely unreadable input still reads as the user's problem."""
    broken = '<?xml version="1.0"?>\n<ades><optical><stn>691</stn></optic'
    try:
        loader._parse(broken, _obscodes())
        assert False, "malformed XML should raise"
    except obs_sources.ObsParseError as ex:
        assert "ParseError" in str(ex) or "mismatch" in str(ex).lower(), str(ex)
    r = loader.fit_pasted(broken, _obscodes())
    assert r.get("ok") is False and "could not parse" in r["error"], r


def test_internal_error_propagates_instead_of_blaming_the_paste():
    """An internal TypeError -- the stale-stub / drifted-signature class -- must NOT be
    dressed up as a parse failure. It propagates; server.py's per-route handler turns it
    into an error payload in production, and a test sees the real exception."""
    orig = loader._parse
    def broken_parse(text, obscodes):        # drifted signature: no radar_out
        return [], "OBS80"
    loader._parse = broken_parse
    try:
        try:
            loader.fit_pasted("whatever", _obscodes())
            assert False, "an internal TypeError should propagate, not be swallowed"
        except TypeError as ex:
            assert "radar_out" in str(ex), str(ex)
        try:
            loader.add_to_fit("whatever", _obscodes(), fit_key="nope")
            # add_to_fit resolves the base fit FIRST; an unknown fit_key returns an
            # error before parsing, which is fine -- what matters is it never reports
            # a TypeError as a parse failure.
        except TypeError as ex:
            assert "radar_out" in str(ex), str(ex)
    finally:
        loader._parse = orig


def test_unparseable_but_well_formed_text_is_empty_not_an_error():
    """Text we can read but that carries no observations is not an exception -- it is
    the 'no usable observations' payload, with a format hint."""
    r = loader.fit_pasted("hello world\nnothing to see here\n", _obscodes())
    assert r.get("ok") is False, r
    assert "no usable observations" in r["error"], r
    assert "hint" in r, r


def test_obsparseerror_is_a_valueerror():
    """It stays a ValueError subclass so any caller that already handled ValueError from
    the parsers keeps working."""
    assert issubclass(obs_sources.ObsParseError, ValueError)


def main():
    fails = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  PASS  {name}")
            except Exception as ex:
                print(f"  FAIL  {name}: {type(ex).__name__}: {ex}"); fails.append(name)
    print(f"\n{len(fails)} failed" if fails else "\nall loader-error tests passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
