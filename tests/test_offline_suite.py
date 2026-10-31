"""The default test run is offline (repo-root conftest.py): no network, no developer cache.

A test that reaches for the network must fail loudly, never pass on one machine and fail
on another; the local loopback (the web tests' own servers) stays reachable."""
import os
import socket
import threading

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PYROORB_TEST_NETWORK_BLOCKED") != "1",
    reason="network block lifted (--live)")


def test_a_remote_connection_is_refused():
    with pytest.raises(OSError, match="blocked in the offline test suite"):
        socket.create_connection(("192.0.2.1", 80), timeout=2)      # TEST-NET-1
    with pytest.raises(OSError, match="blocked"):
        socket.create_connection(("2001:db8::1", 80), timeout=2)


def test_loopback_still_works():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    t = threading.Thread(target=lambda: srv.accept()[0].close(), daemon=True)
    t.start()
    c = socket.create_connection(("127.0.0.1", port), timeout=5)
    c.close()
    t.join(timeout=5)
    srv.close()


def test_the_cache_is_a_fresh_temporary_folder():
    assert "pyroorb-test-cache-" in os.environ["PYROORB_CACHE_DIR"]
    assert os.environ["PYROORB_OBSCODES_REFRESH"] == "never"
