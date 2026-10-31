"""force_assist.get_ephem must build each ASSIST ephemeris exactly ONCE.

An ``assist.Ephem`` is a small Python object wrapping a C mmap of the DE440 /
sb441-n16 kernels, so a duplicate is cheap to construct and then quietly pays
its OWN resident pages for the very same files as it gets used -- measured at
+6 MB for one Ephem sweeping 40 years and another +6 MB for a second sweeping
the identical span, while re-sweeping the first adds nothing.  Unlocked, the
lazy "if None: build" was a plain race: six threads through it returned FIVE
distinct Ephem objects, so a cold web server whose fit workers all start at
once could map the ~747 MB of kernels several times over -- invisible to the
/memory page, because the Python object is tiny and the cost is the mapping's.
"""
import threading

import pytest

import pyroorb.force_assist as fa


class _CountingEphem:
    """Stands in for assist.Ephem: counts constructions and is slow enough that
    an unlocked racer reliably loses."""

    calls = 0
    lock = threading.Lock()

    def __init__(self, *args):
        with _CountingEphem.lock:
            _CountingEphem.calls += 1
        # Wide enough that every racer is inside the window, not a timing fluke.
        threading.Event().wait(0.05)
        self.args = args


@pytest.fixture
def counting_ephem(monkeypatch):
    _CountingEphem.calls = 0
    monkeypatch.setattr(fa.assist, "Ephem", _CountingEphem)
    monkeypatch.setattr(fa, "ensure_planets_kernel", lambda *a, **k: "planets.bsp")
    monkeypatch.setattr(fa, "asteroid_kernel_path", lambda *a, **k: "ast.bsp")
    monkeypatch.setattr(fa, "_EPHEM", None)
    monkeypatch.setattr(fa, "_EPHEM_AST", None)
    return _CountingEphem


@pytest.mark.parametrize("asteroids", [False, True])
def test_concurrent_get_ephem_builds_one(counting_ephem, asteroids):
    """N threads racing get_ephem get ONE object, built once -- for both caches."""
    n = 8
    barrier = threading.Barrier(n)
    got, errors = [], []
    guard = threading.Lock()

    def racer():
        try:
            barrier.wait(timeout=10)
            e = fa.get_ephem(asteroids)
        except Exception as ex:                      # pragma: no cover - failure path
            with guard:
                errors.append(ex)
            return
        with guard:
            got.append(e)

    threads = [threading.Thread(target=racer) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    assert len(got) == n
    assert len({id(e) for e in got}) == 1, "get_ephem handed out duplicate Ephem objects"
    assert counting_ephem.calls == 1, (
        f"built {counting_ephem.calls} Ephem objects; each one maps the kernels again")


def test_get_ephem_caches_across_calls(counting_ephem):
    """The cached read is a plain hit -- the lock must not rebuild anything."""
    first = fa.get_ephem(False)
    assert fa.get_ephem(False) is first
    assert fa.get_ephem(False) is first
    assert counting_ephem.calls == 1


def test_gravity_and_asteroid_caches_are_separate(counting_ephem):
    """Two caches, two ephemerides -- the lock must not collapse them into one."""
    grav = fa.get_ephem(False)
    ast = fa.get_ephem(True)
    assert grav is not ast
    assert counting_ephem.calls == 2
    assert fa.get_ephem(False) is grav
    assert fa.get_ephem(True) is ast
    assert counting_ephem.calls == 2
