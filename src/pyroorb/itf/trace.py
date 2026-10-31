"""Live text trace of an ITF search (debugging / spectating / performance).

A :class:`Tracer` is a timestamped line sink threaded through the search
(:func:`itf.search.itf_search` ``trace=``) and the DB candidate source: every
consequential decision -- corridor size, database hits, candidate scoring,
which tracklet subsets each backoff trial tried and why they were kept or
skipped, recursion, gate values -- reaches it, prefixed with the elapsed
seconds since the search began.  The CLI prints them as they happen (``python
-m itf search`` by default, ``--quiet`` to suppress); the web UI streams them
to the "see trace" live feed (GET /api/itf_trace polling).

FOUR LEVELS, because a wide-corridor sweep emits tens of thousands of lines
and the story is not readable inside them.  Every line declares the level it
belongs to and a Tracer only writes lines at or below its own:

  1 HEADLINE  the shape of the search: seed, one line per round, every TAKEN /
              recursion, the final verdict, and any warning.  "Is this worth
              waiting for?"
  2 STORY     + the narrative, rolled up per round: what the sweep found, the
     (default)  P(match) distribution and its top few, which trials the trial queue
              tried and how each died, the acceptance gates, why a linkage was
              or was not submittable/displayable.  "What is it doing?"
  3 DETAIL    + every trial, candidate, gate and rival verbatim -- the level
              this trace has always run at.  "Why didn't THAT tracklet link?"
  4 FIREHOSE  + per-window corridor rows and per-chunk DB queries (rolled up
              into one line each at levels 1-3).  "Where is the time going?"

Levels 1-3 are cumulative narration of the same search, so a level-2 read is
a complete story with no gaps -- not level 3 with holes in it.

ROUND STATE.  The narrative lines are rollups, so somebody has to count.  The
Tracer does (:meth:`round_begin` / :meth:`count` / :meth:`field`): it is
already threaded through both the search and the candidate source, which are
the two places the counts come from, and holding the tally here keeps the
counters out of the search's own control flow.
"""

import time

L_HEAD = 1          # the shape of the search
L_STORY = 2         # the narrative (default)
L_DETAIL = 3        # every decision verbatim (the historical default)
L_FIREHOSE = 4      # per-window / per-DB-query rows

DEFAULT_LEVEL = L_STORY

_FIELD_W = 12       # label column of a round's field lines


class Tracer:
    """Elapsed-stamped, levelled trace sink.  ``sink`` is any
    ``callable(str)`` (print, ``list.append``, a streaming buffer); a None sink
    disables tracing entirely.

    ``level`` is the most verbose level this tracer will write (see the module
    docstring).  Call sites either name their level -- ``trc(msg, L_HEAD)``,
    or the ``head``/``story``/``field`` helpers -- or say nothing and land on
    ``L_DETAIL``, which is where every line written before levels existed
    belongs.

    Hot loops that would build an expensive string only to have it dropped
    should guard with :meth:`wants`; ``enabled`` remains the cheap "is anyone
    listening at all" test."""

    def __init__(self, sink=None, level=DEFAULT_LEVEL):
        self.sink = sink
        self.level = level
        self.t0 = time.monotonic()
        self._round = {}
        self._round_open = False
        self._last = {}         # say-it-once memory (see field_changed)

    @property
    def enabled(self):
        return self.sink is not None

    def wants(self, level):
        """True if a line at ``level`` would actually be written."""
        return self.sink is not None and level <= self.level

    def __call__(self, msg, level=L_DETAIL):
        if self.sink is not None and level <= self.level:
            self.sink(f"[{time.monotonic() - self.t0:7.2f}s] {msg}")

    # -- shorthands -----------------------------------------------------------
    def head(self, msg):
        """A headline: seed, round, take, verdict, warning (level 1)."""
        self(msg, L_HEAD)

    def story(self, msg):
        """A narrative line (level 2)."""
        self(msg, L_STORY)

    def warn(self, msg):
        """A warning -- headline level, and prefixed '!' so a reader scanning
        the trace and a reader grepping it find them the same way."""
        self(f"! {msg}", L_HEAD)

    def blank(self, level=L_HEAD):
        """A bare separator line (no elapsed stamp).  The trace is read by eye
        in a scrolling pane, so the round boundaries have to be visible
        without parsing anything."""
        if self.sink is not None and level <= self.level:
            self.sink("")

    def field(self, name, text, level=L_STORY):
        """One ``name``-labelled line of the current round's rollup, indented
        under its header and column-aligned so the labels scan vertically."""
        self(f"  {name:<{_FIELD_W}} {text}", level)

    def cont(self, text, level=L_STORY):
        """A continuation line under the previous :meth:`field` (no label)."""
        self(f"  {'':<{_FIELD_W}} {text}", level)

    # -- say-it-once ----------------------------------------------------------
    # A 31-variant LOV scan re-states the same parent orbit and the same likely
    # gate 31 times.  Repetition is what makes a trace unscannable, so these
    # emit only on CHANGE and stay silent while a fact holds -- the reader
    # learns it once and may assume it until told otherwise.  The memory is
    # per-tracer, deliberately NOT reset by :meth:`round_begin`.
    def field_changed(self, name, text, level=L_STORY):
        if self._last.get(name) != text:
            self._last[name] = text
            self.field(name, text, level)

    def cont_changed(self, key, text, level=L_STORY):
        if self._last.get(key) != text:
            self._last[key] = text
            self.cont(text, level)

    # -- round tally ----------------------------------------------------------
    def round_begin(self, title):
        """Open a round: blank separator + header, and a fresh tally."""
        self._round = {}
        self._round_open = True
        self.blank()
        self.head(f"=== {title}")

    def round_end(self):
        self._round_open = False

    def count(self, key, n=1):
        """Add to the current round's tally of ``key``."""
        if self.sink is not None:
            self._round[key] = self._round.get(key, 0) + n

    def note(self, key, value):
        """Record a non-additive value (a max, a label) on the tally."""
        if self.sink is not None:
            self._round[key] = value

    def get(self, key, default=0):
        return self._round.get(key, default)

    def pop(self, key, default=0):
        return self._round.pop(key, default)


def as_tracer(trace, level=None):
    """Coerce a ``trace`` argument (None | callable sink | Tracer) to a Tracer.

    A Tracer passed through keeps its own level unless ``level`` overrides it,
    so a caller that built the sink (the web feed) and a caller that chose the
    verbosity (the CLI flag) do not have to be the same caller."""
    if isinstance(trace, Tracer):
        if level is not None:
            trace.level = level
        return trace
    return Tracer(trace, DEFAULT_LEVEL if level is None else level)


def day_of(epoch_sec):
    """UTC yyyy-mm-dd of a unix epoch (trace labels)."""
    return time.strftime("%Y-%m-%d", time.gmtime(epoch_sec))


def span_of(epochs):
    """"yyyy-mm .. yyyy-mm" over an iterable of unix epochs, for a rolled-up
    sweep line that has to say WHEN it looked without listing every window."""
    lo = hi = None
    for e in epochs:
        if lo is None or e < lo:
            lo = e
        if hi is None or e > hi:
            hi = e
    if lo is None:
        return "-"
    a, b = day_of(lo)[:7], day_of(hi)[:7]
    return a if a == b else f"{a} .. {b}"


def histogram(probs, edges=(0.5, 0.1)):
    """"P>=0.5: 3, 0.1-0.5: 12, <0.1: 940" -- the ML field's shape in one
    phrase.  The old trace printed a top-12 list and "and 9453 more", which
    answers "what was best" but never "were any of them promising"."""
    vals = [p for p in probs if p is not None]
    if not vals:
        return "no ML scores"
    hi, lo = edges
    n_hi = sum(1 for p in vals if p >= hi)
    n_mid = sum(1 for p in vals if lo <= p < hi)
    return (f"P>={hi:g}: {n_hi}, {lo:g}-{hi:g}: {n_mid}, "
            f"<{lo:g}: {len(vals) - n_hi - n_mid}")


def row_label(row):
    """What to call a candidate's tracklet: the submitter's own ``trksub``
    when the row carries one, else the packed designation core the 80-column
    line holds in columns 6-12 (`batch.tracklet_summary` reads the same
    field).  Many ITF rows have no trksub at all -- a Pan-STARRS row carrying
    ``/25K01d`` used to label its whole tracklet ``?``, which names nothing
    and hides that the material is a comet fragment."""
    try:
        sub = (row.get("trksub") or "").strip()
        if sub:
            return sub
        return (row.get("obs80") or "")[5:12].strip() or "?"
    except Exception:
        return "?"


def tracklet_tag(tr):
    """Short human tag for a candidate tracklet dict: trksub (or the packed
    id, see `row_label`), UTC day, station, obs count.  Never raises (trace
    labels only)."""
    try:
        r0 = tr["rows"][0]
        return (f"{row_label(r0)}"
                f"@{r0.get('stn', '?')}"
                f"/{day_of(r0['epoch_sec'])}"
                f"({len(tr['rows'])}obs)")
    except Exception:
        return "?"
