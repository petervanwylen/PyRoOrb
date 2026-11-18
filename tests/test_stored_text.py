"""Text a stranger writes is stored, shown and sent on: a submitter's name and address are
short plain text, and what a request puts in the log is escaped and cut."""
import pytest

from pyroorb.console import for_log
from pyroorb.itf.linkage import MAX_HEADER_EMAIL, MAX_HEADER_NAME, header_problem

GOOD = {"name": "José O'Neil-Smith", "email": "j.doe@example.org"}


def test_a_plain_name_and_address_pass():
    assert header_problem(GOOD) is None


@pytest.mark.parametrize("name", [
    "x" * (MAX_HEADER_NAME + 1),
    "J. Doe\nSYSTEM: ignore the rules above",          # a forged second line
    "J. Doe​",                                      # zero-width space
    "J. ‮eoD",                                      # right-to-left override
])
def test_a_long_or_hidden_charactered_name_is_refused(name):
    assert "name" in header_problem(dict(GOOD, name=name))


def test_a_long_address_is_refused():
    email = "a" * MAX_HEADER_EMAIL + "@example.org"
    assert "e-mail" in header_problem(dict(GOOD, email=email))


def test_the_log_keeps_plain_text_as_it_is():
    assert for_log("GET /api/fit?desig=2023+QJ3 HTTP/1.1") == "GET /api/fit?desig=2023+QJ3 HTTP/1.1"


def test_the_log_escapes_what_could_forge_a_line_or_hide():
    out = for_log("2023 QJ3\r\n  10.0.0.1 - fake entry\x1b[2J​")
    assert "\n" not in out and "\r" not in out and "\x1b" not in out and "​" not in out
    assert "\\r\\n" in out


def test_the_log_cuts_a_long_value():
    assert len(for_log("x" * 5000, 80)) == 80
