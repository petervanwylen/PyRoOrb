"""The force model an orbit was fitted in, carried as STATE on the fit's backend.

A fit's context holds one backend object (``ctx["backend"]``), and every later step that
integrates the orbit -- the evaluator's residuals and Jacobian, a re-anchor, the LOV, the range
profile, an ephemeris, the public products -- goes through it. Two thin wrappers make that object
say which field it integrates in, so nothing has to be reconstructed later from process-wide
settings or from the display label ``fit.force_model``:

* :class:`PinnedForce` fixes the asteroid perturbers ON or OFF for every call that takes an
  ``asteroids=`` flag. A call that leaves the flag out (or passes None, "the global default")
  gets the pinned value; an explicit True/False still wins, so a caller that deliberately asks
  for the other model (the +16ast keep-gate's comparison) gets it.
* ``force_assist.SelfExcludedForce`` takes ONE of the 16 perturber bodies out of the field, for
  an object that is itself that body. It marks itself with the ``_self_excluded`` attribute,
  which :func:`excluded_body` reads.

This module imports nothing heavy (force_assist imports ASSIST at import time, and the engine
must import on a rebound-only box), so the call list lives here and force_assist reads it.
"""

# Every backend call that accepts an `asteroids=` flag. force_assist re-exports it as
# ASTEROID_AWARE_CALLS, and tests/test_force_model_durable.py checks it against every
# force_assist function whose signature has the parameter, so a new entry point cannot be
# missed by the pin.
ASTEROID_AWARE_CALLS = frozenset((
    "predict_radec_jac_ng", "predict_radec_jac_batch", "predict_radec_batch",
    "helio_states_multi", "helio_positions_batch", "predict_state", "_jac_batch"))


class PinnedForce:
    """A force backend whose asteroid perturbers are fixed ON or OFF.

    Attribute access falls through to the wrapped backend, so hasattr-style capability checks
    are unchanged. Wrapping a PinnedForce replaces its pin (the inner one is unwrapped); any
    other wrapper -- above all a SelfExcludedForce -- is kept inside, so the exclusion survives.
    """

    def __init__(self, backend, asteroids):
        if isinstance(backend, PinnedForce):
            backend = backend._backend
        self._backend = backend
        self._asteroids = bool(asteroids)

    def __getattr__(self, name):
        f = getattr(self._backend, name)
        if name in ASTEROID_AWARE_CALLS and callable(f):
            pin = self._asteroids

            def _pinned(*a, **k):
                if k.get("asteroids") is None:
                    k["asteroids"] = pin
                return f(*a, **k)
            return _pinned
        return f

    def __repr__(self):
        return f"PinnedForce({self._backend!r}, asteroids={self._asteroids})"


def is_assist(backend):
    """True for the ASSIST backend (the only one with asteroid perturbers), wrapped or not."""
    return hasattr(backend, "ensure_asteroid_kernel")


def pin(backend, asteroids):
    """`backend` with its asteroid perturbers fixed to `asteroids`; the rebound 8-planet
    fallback (no perturbers, no `asteroids=` parameter) is returned unchanged, and so is a
    backend already pinned to the same value (an identification fit's forcing wrapper)."""
    if not is_assist(backend):
        return backend
    if isinstance(backend, PinnedForce) and backend._asteroids == bool(asteroids):
        return backend
    return PinnedForce(backend, asteroids)


def _chain(backend):
    """`backend` and every backend it wraps, outermost first."""
    seen = 0
    while backend is not None and seen < 16:
        yield backend
        backend = (vars(backend).get("_backend") if hasattr(backend, "__dict__") else None)
        seen += 1


def pinned_asteroids(backend):
    """The asteroid pin of `backend` (True / False), or None when it carries none."""
    for b in _chain(backend):
        if isinstance(b, PinnedForce):
            return b._asteroids
    return None


def excluded_body(backend):
    """Display name ('4 Vesta') of the perturber body `backend` excludes from its own field
    (force_assist.SelfExcludedForce, at any depth), or None."""
    for b in _chain(backend):
        name = vars(b).get("_self_excluded") if hasattr(b, "__dict__") else None
        if name is not None:
            return name
    return None
