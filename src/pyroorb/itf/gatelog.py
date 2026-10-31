"""Opt-in instrumentation for the two gates that decide whether a THIN
multi-opposition profile is shown or submitted:

* :func:`itf.search._show_lone_night` -- the 1.5 profile (one lone new night),
  shown for review or withheld;
* :func:`itf.search._confirm_top_pair` -- the three-thin upgrade, which turns
  two lone single-night oppositions into a submittable linkage.

Both were assembled from case law rather than measurement, and
re-calibrating them needs the CLAUSE
STATISTICS of every attempt -- including the attempts an earlier clause
short-circuited, which the search itself never computes.  This module is how a
study run collects them, and it is **off** unless ``PYROORB_ITF_GATELOG``
names a file: no environment variable, no work, no behaviour change.

Environment
-----------
``PYROORB_ITF_GATELOG``
    Path to a JSONL file.  Every row is one gate attempt.  Set = ON.
``PYROORB_ITF_GATELOG_TAG``
    A JSON object merged into every row (the study driver stamps the object
    and which arm of the experiment is running).
``PYROORB_ITF_GATELOG_EXHAUST``
    ``0`` to skip the exclusivity measurement when the real gate would not
    have reached it.  Default on: the exhaustion clause spends trial fits, but
    whether it is what withholds correct recoveries is precisely the question.
``PYROORB_ITF_GATELOG_RIVALK``
    ``K > 0`` to COUNT how many of the top-K rivals also link, on every show
    the gate makes without having measured exclusivity.  Off by default
    because it costs K trial fits: it exists to answer "is the thing we are
    showing one of two, or one of ten?", which the search never asks in the
    no-cliff case (it spends zero fits there and shows on P alone).

Rows carry MEASUREMENTS and the search's own verdict -- never a judgement
about whether the tracklet was the right one.  Truth is the driver's business
(it planted it), and keeping the engine ignorant of it is what makes the
resulting calibration honest.
"""

import json
import os
import threading

__all__ = ["enabled", "want_exhaustion", "n_rivals", "record", "tag"]

_LOCK = threading.Lock()


def _path():
    return os.environ.get("PYROORB_ITF_GATELOG") or None


def enabled():
    """True when a study run asked for the gate feed."""
    return bool(_path())


def want_exhaustion():
    """Should the exclusivity clause be MEASURED even when the real gate
    refused before reaching it?  On by default (see the module docstring)."""
    return os.environ.get("PYROORB_ITF_GATELOG_EXHAUST", "1") not in ("0", "")


def n_rivals():
    """How many rivals to fit-count per non-exclusive show (0 = don't)."""
    try:
        return max(0, int(os.environ.get("PYROORB_ITF_GATELOG_RIVALK", "0")))
    except ValueError:
        return 0


def tag():
    """The driver's per-run stamp (object, arm, ...), or ``{}``."""
    raw = os.environ.get("PYROORB_ITF_GATELOG_TAG")
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {"tag": val}
    except ValueError:
        return {"tag": raw}


def record(kind, **fields):
    """Append one row.  Never raises: instrumentation must not be able to
    fail a search, and a study that loses a row loses a row."""
    path = _path()
    if not path:
        return
    try:
        row = {"kind": kind, **tag(), **fields}
        line = json.dumps(row, default=_jsonable)
        with _LOCK:
            # one flushed append per row: rows are small, runs are long, and a
            # study that dies mid-object must keep everything it had (the
            # torn-append hazard that bit the training harness is a
            # large-record problem -- the reader still skips bad lines)
            with open(path, "a") as fh:
                fh.write(line + "\n")
                fh.flush()
    except Exception:
        pass


def _jsonable(obj):
    if isinstance(obj, (set, frozenset)):
        return sorted(str(x) for x in obj)
    return str(obj)


def read(path):
    """Every well-formed row of a feed, skipping any torn tail."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out
