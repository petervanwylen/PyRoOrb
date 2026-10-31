"""Settings for the whole test suite (every test tree), applied before any engine module
is imported.

The default run is truly offline -- what continuous integration sees:

* **No network.** Every connection to anything but this machine is refused, so a test
  that quietly fetches fails at once instead of passing on a developer's box and failing
  in CI (or changing its answer when the MPC changes its data). Tests that genuinely need
  live data are marked ``live`` and run only with ``--live``, which lifts the block.
* **No developer cache.** Downloads and fetch caches go to a fresh temporary folder, so a
  test cannot lean on an object someone happened to fetch earlier; a test's inputs are its
  tracked fixtures. The fixed reference tables (DE440, the debias tables) are not
  astrometry and never change: they stay where they are (pinned below before the cache
  moves), and tests that need the 645 MB asteroid kernel skip without it.
* **The station table** is the copy bundled with the package, and is never downloaded.
"""
import ipaddress
import os
import socket
import sys
import tempfile

from pyroorb import datadirs  # noqa: E402

# Pin the reference tables where they already resolve, THEN move the cache.
os.environ.setdefault("PYROORB_EPHEM_DIR", str(datadirs.table_dir("ephem")))
os.environ.setdefault("PYROORB_DEBIAS_DIR", str(datadirs.table_dir("debias")))
_CACHE = tempfile.mkdtemp(prefix="pyroorb-test-cache-")
os.environ["PYROORB_CACHE_DIR"] = _CACHE
os.environ["PYROORB_OBSCODES_REFRESH"] = "never"
os.environ["PYROORB_OBSCODES_PATH"] = os.path.join(_CACHE, "no-obscodes-cache", "ObsCodes.htm")
os.environ["PYROORB_DEBIAS_AUTOFETCH"] = "0"

_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex


class NetworkBlocked(OSError):
    """A test reached for the network in the offline suite."""


def _is_local(address):
    if not isinstance(address, tuple) or not address:
        return True                                  # AF_UNIX and friends
    host = str(address[0])
    if host in ("localhost", ""):
        return True
    try:
        ip = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return False
    return ip.is_loopback or ip.is_unspecified


def _refuse(address):
    raise NetworkBlocked(
        f"network access to {address!r} is blocked in the offline test suite "
        "(repo-root conftest.py): freeze the input as a tracked fixture, or mark the test "
        "`live` (run with --live)")


def _guarded_connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(address):
        _refuse(address)
    return _REAL_CONNECT(self, address)


def _guarded_connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(address):
        _refuse(address)
    return _REAL_CONNECT_EX(self, address)


def block_network():
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    os.environ["PYROORB_TEST_NETWORK_BLOCKED"] = "1"


def unblock_network():
    socket.socket.connect = _REAL_CONNECT
    socket.socket.connect_ex = _REAL_CONNECT_EX
    os.environ.pop("PYROORB_TEST_NETWORK_BLOCKED", None)


def _live_requested(config):
    try:
        return bool(config.getoption("--live"))
    except (ValueError, AttributeError):
        return "--live" in sys.argv


def pytest_configure(config):
    if not _live_requested(config):
        block_network()
