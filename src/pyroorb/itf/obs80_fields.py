"""Column-level helpers for raw 80-column observation lines (the ITF search's
bookkeeping is defined on the published line format: obs identity, duplicate
detection, wrap handling, epoch conversion).

Split out of the former ``itf/obsutil.py``, which mixed this with
the night/opposition counters (now :mod:`itf.opp_profile`) and the astrometry
jitter hack (now :mod:`itf.jitter`). Ported from the proven notebook pipeline
(``ift outline/itflib/obs.py``); the parsed-obs side of the engine uses
``src/pyroorb/obs_sources.py`` as usual.
"""

EARTH_RAD_AU = 4.26352e-5


def unix_from_jd(jd):
    return (jd - 2440587.5) * 86400.0


def unix_from_mjd(mjd):
    return (mjd - 40587.0) * 86400.0


def obs_unique_id(line):
    """Identity of one observation for duplicate detection: date+time cols, the
    SKY POSITION (RA+Dec) and the station code (cols 16-30 + 33-56 + 78-80).

    The position is load-bearing, not cosmetic: one exposure (same station, same
    timestamp) routinely carries two different objects (a crowded survey frame),
    so a station+time-only id collides them.  In the ITF search that collision
    silently DROPPED a true tracklet -- the `by_key` sweep dedup collapsed a
    planted tracklet with a real ITF neighbour from the same G96 exposure and
    kept the (lower-score) neighbour -- and it also aliases residual/seed-obs
    maps.  Same fix the identification path already made (RA/Dec, not just
    obscode+time)."""
    return line[15:30] + line[32:56] + line[77:80]


def is_companion(oid):
    """True when ``oid`` (an :func:`obs_unique_id`) names the SECOND line of a
    two-line record -- the `s` satellite-position / `v` roving-site line that
    follows a space-based or roving observation -- rather than an observation.

    The id keeps cols 33-34: on an observation line they are the RA hours, two
    digits; on a companion line they are the units flag and a blank (`s`) or
    blank-led longitude (`v`), never two digits.  The companion travels with
    its observation through every id-keyed filter, so it stays IN the id sets;
    it must not COUNT: an 8-obs NEOWISE (C51) tracklet is 16 lines, and
    counting those read every one at 50% retained."""
    return not oid[15:17].isdigit()


def n_observations(ids):
    """How many observations the id set ``ids`` holds (companion lines are
    not observations, see :func:`is_companion`)."""
    return sum(1 for i in ids if not is_companion(i))


def obs_ids(astrometry):
    """The :func:`obs_unique_id` set of a block of 80-column ``astrometry``.

    The one-liner this replaces was written out a dozen times in ``search.py``
    alone, each spelling out the ``if l.strip()`` that keeps a trailing newline
    from becoming an id -- worth a name, because "the identities in this text"
    is a thing the search talks about constantly and forgetting the guard is
    silent (a blank line yields the id ``""``, which then matches nothing and
    inflates every count by one)."""
    return {obs_unique_id(l) for l in astrometry.splitlines() if l.strip()}


def drop_obs_ids(astrometry, drop):
    """``astrometry`` with every line whose :func:`obs_unique_id` is in ``drop``
    removed, blank lines dropped, one trailing newline per line kept.

    The companion to :func:`obs_ids`: the search removes tracklets from a block
    of text by identity in several places (a refused tracklet, a clipped
    poor-fitter), and each did it with its own generator expression."""
    return "".join(l + "\n" for l in astrometry.splitlines()
                   if l.strip() and obs_unique_id(l) not in drop)


def count_obs(astrometry):
    """Distinct observation instants (date+time cols) in an OBS80 blob."""
    return len({x[15:32] for x in astrometry.splitlines() if x.strip()})


def shift_above_180(x):
    """Map RA to (-180, 180] so arithmetic near the 0/360 seam is continuous."""
    return x - 360 if x > 180 else x


def is_crossing_360(a, b, c):
    """True when three successive RAs straddle the 0/360 seam (the shifted
    linear extrapolation predicts the third point better than the raw one)."""
    a1, b1, c1 = shift_above_180(a), shift_above_180(b), shift_above_180(c)
    regular_proj = (b - a) + b
    crossing_proj = (b1 - a1) + b1
    return abs(c1 - crossing_proj) < abs(c - regular_proj)


def split_obs80_long_lines(txt):
    """Split any 160-char two-line records (satellite obs fetched as one row)
    into proper 80-col line pairs."""
    out = []
    for line in txt.splitlines():
        if len(line) == 160:
            out.append(line[:80])
            out.append(line[80:160])
        elif line.strip():
            out.append(line)
    return "\n".join(out) + ("\n" if out else "")
