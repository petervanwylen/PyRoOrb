"""Link scorers: optional judges of a whole linkage that the ITF learns about at run time
instead of importing by name.

A link scorer is a callable ``scorer(obs80_text, obscodes) -> str | None``: given the whole
linked astrometry (80-column lines), it returns one sentence when it objects to the linkage as
a possible chimera (tracklets of more than one body), else None.  An objection is a WARNING on
the linkage -- shown with it, never a refusal by itself (see ``srsearch.chimera_warnings``).
A scorer that cannot score this linkage at all (outside what it judges) returns
:data:`NO_VERDICT`: no objection, and no confirmation either -- a linkage nothing else has
judged then gets the chimera guard instead (``srsearch.chimera_warnings``, ``unvouched``).

Discovery, in this order (duplicates ignored):

0. The scorers that ship with PyRoOrb (:data:`BUILTIN`): the mislinkage model
   (``pyroorb.mislink.linkscore``).  Its model runtime (xgboost) comes with the ``itf`` extra;
   without it the scorer does not load, and every linkage says the check was not run.
1. Python entry points in the group ``pyroorb.link_scorers`` -- a scorer installed as its own
   package; each entry point names the callable.
2. A registry file, ``link_scorers.txt`` at the checkout root (or the path in
   ``PYROORB_LINK_SCORERS_FILE``): one ``module:attribute`` per line, ``#`` comments allowed.

A scorer that cannot be loaded, or raises while scoring, never objects (its failure is not
evidence about the linkage) -- but it has not confirmed the linkage either: :func:`judge`
names it, and the caller says so (``srsearch.CHECK_NOT_RUN``).
"""
import importlib
import os
import threading
from pathlib import Path

GROUP = "pyroorb.link_scorers"
#: The link scorers that ship with the package, loaded before any discovered one.
BUILTIN = ("pyroorb.mislink.linkscore:SCORER",)
FILE_ENV = "PYROORB_LINK_SCORERS_FILE"
_CHECKOUT = Path(__file__).resolve().parents[3]
_LOCK = threading.Lock()


class _NoVerdict:
    """A scorer's answer when it cannot score the linkage: false, like "no objection"."""

    def __bool__(self):
        return False

    def __repr__(self):
        return "NO_VERDICT"


#: What a scorer returns for a linkage it cannot score (no objection, no confirmation).
NO_VERDICT = _NoVerdict()
_LOADED = None
_UNLOADED = ()


def _from_spec(spec):
    mod, _, attr = spec.strip().partition(":")
    obj = importlib.import_module(mod)
    for part in (attr or "SCORER").split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{spec} is not callable")
    return obj


def _registry_file():
    want = os.environ.get(FILE_ENV, "").strip()
    return Path(want) if want else _CHECKOUT / "link_scorers.txt"


def discover(log=print, failed=None):
    """Every link scorer this process can find (:data:`BUILTIN`, entry points, then the
    registry file); the spec of each one that would not load is appended to ``failed``."""
    found, seen = [], set()

    def add(spec):
        if spec in seen:
            return
        seen.add(spec)
        try:
            found.append(_from_spec(spec))
        except Exception as ex:                              # noqa: BLE001
            log(f"link scorer {spec}: not loaded ({type(ex).__name__}: {ex})")
            if failed is not None:
                failed.append(spec)

    for spec in BUILTIN:
        add(spec)
    try:
        from importlib.metadata import entry_points
        for ep in entry_points(group=GROUP):
            add(ep.value)
    except Exception as ex:                                  # pragma: no cover - defensive
        log(f"link scorers: entry points unreadable ({ex})")
    path = _registry_file()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            spec = line.split("#", 1)[0].strip()
            if spec:
                add(spec)
    return found


def installed():
    """The loaded link scorers (discovered once per process)."""
    global _LOADED, _UNLOADED
    if _LOADED is None:
        with _LOCK:
            if _LOADED is None:
                failed = []
                _LOADED, _UNLOADED = tuple(discover(failed=failed)), tuple(failed)
    return _LOADED


def set_for_tests(scorers):
    """Replace the loaded scorers (tests only); None re-discovers on next use."""
    global _LOADED, _UNLOADED
    _LOADED, _UNLOADED = (None if scorers is None else tuple(scorers)), ()


def judge(obs80_text, obscodes, scorers=None, verdicts=None):
    """``(objections, not_run)`` for the linkage ``obs80_text``: every scorer's objection
    (sentences), and every scorer that could not judge it -- one that would not load, or
    raised.  ``scorers`` replaces the installed ones.  ``verdicts``, a list, receives the
    name of every scorer that gave a verdict (objected or did not), i.e. neither raised nor
    answered :data:`NO_VERDICT`."""
    loaded = installed() if scorers is None else scorers
    objections = []
    not_run = [f"link scorer {spec} did not load" for spec in (_UNLOADED if scorers is None
                                                                else ())]
    for fn in loaded:
        try:
            why = fn(obs80_text, obscodes)
        except Exception as ex:                              # noqa: BLE001 -- no verdict
            name = getattr(fn, "__qualname__", None) or type(fn).__name__
            not_run.append(f"link scorer {name} failed ({type(ex).__name__})")
            continue
        if why is NO_VERDICT:
            continue
        if verdicts is not None:
            verdicts.append(getattr(fn, "__qualname__", None) or type(fn).__name__)
        if why:
            objections.append(str(why))
    return objections, not_run
