"""Recovery test harness: can the search re-assemble a known multi-opposition
object from one seeded apparition?

For a chosen object the harness fetches its published astrometry, keeps ONE
apparition as the seed, and turns every OTHER apparition's tracklets into
**fake ITF tracklets**: obs80 lines re-labeled under synthetic trksubs, with
tracklet-index feature rows computed by the very same
:func:`pyroorb.itf.index.compute_tracklet_features` the real table uses.  An
:class:`~pyroorb.itf.candidates.OverlaySource` then serves them alongside whatever
the base source returns (the live ITF, or nothing when offline), admitting a
fake ONLY through the same db-side predicates a real row would face -- so a
planted tracklet is discoverable exactly iff its real ITF twin would have
been.  Success = the search recovers the planted apparitions.

This validates the whole loop end to end (seed fit -> corridor -> candidate
retrieval -> trial fits -> recursion -> gates) without writing a byte to the
shared database.
"""

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .candidates import OverlaySource
from .config import ItfConfig
from .index import IndexConfig, compute_tracklet_features
from .obs80_fields import obs_unique_id, unix_from_jd
from .search import itf_search
from .verdict import SearchVerdict

HARNESS_OPPOSITION_GAP_DAYS = 275.0          # same gap the opposition counter uses
FAKE_RUN_GAP_DAYS = 0.5              # max spacing inside a planted fake-ITF tracklet run: a
                                     # same-station run up to 12 h long is planted as ONE
                                     # tracklet.  Its own name: it shared a constant
                                     # with the night boundary below, which is now the tree's.
SEED_MIN_NIGHTS = 3                  # a seed opposition must have this many nights
SEED_MIN_OBS = 6                     # ...and this many observations:
                                     # 3 observations are 6
                                     # measurements against 6 orbital elements,
                                     # exactly determined, so the fit has ZERO
                                     # degrees of freedom -- rms identically
                                     # 0.000", covariance collapsing to ~1e-21
                                     # and U reading 0.00, the BEST possible
                                     # certainty, for the weakest possible arc.
                                     # 2013 UL49 is the case: seeded on its
                                     # 3-obs 2018 apparition it served 0 of 5
                                     # planted tracklets behind a pencil-thin
                                     # corridor, with the weak-seed variant
                                     # fallback gated off (U 0.00 < 4) -- while
                                     # its 8-obs/4-night 2017 apparition sat
                                     # right there unused.

# ...but 275 d is calibrated for objects that spend part of the year behind the
# Sun.  A distant object does not: at 44 AU it is observable nearly year-round,
# so its gap BETWEEN oppositions is only ~7 months and the 275 d rule merges
# them into one "apparition".  Measured on 2015 VV167 -- 27 obs, two real
# oppositions, largest internal gap 210 d, and split_oppositions returns ONE
# group spanning 533.8 d.  Everything downstream then goes wrong quietly: the
# multi-apparition seed cannot engage (there is only one group to take), the
# seed collapses to the first N nights inside a single opposition (116.8 d,
# U 6.26), and the corridor comes out wider than max_uncert_deg so the sweep
# builds ZERO windows and serves nothing -- not even the planted truth.
DISTANT_OPPOSITION_GAP_DAYS = 150.0


# How many nights of one apparition become the SEED during ML collection, by
# dynamical class.  v2 used 3-4 for everything, which is a deliberately weak
# seed -- the point is to make the search work for its links.
#
# It is too weak for a distant object.  A 3-4 night arc of a TNO barely
# constrains the orbit at all, so the corridor windows come out wider than
# cfg.max_uncert_deg (then 19.4 deg) and the sweep skips every one of them: the two
# pilot TNOs served ZERO candidates and returned "no likely matches" with their
# tracklets sitting right there in the overlay.  Distant objects need a longer
# seed before the corridor means anything -- which is equally true of a real
# search for one, so this is the honest seed shape rather than a concession.
SEED_NIGHTS = {
    None: (3, 4),           # default: MBA, NEO -- the v2 shape
    "TNO": (6, 7, 8),
    "Centaur": (5, 6),
}

# ...and how many APPARITIONS the seed spans, same reasoning taken further: no
# number of nights inside a single TNO apparition constrains the orbit enough
# to build a corridor, because the apparition is ~3 months of a 250-year orbit.
SEED_OPPOSITIONS = {None: 1, "TNO": 2, "Centaur": 2}

# Floor and target for the adaptive seed (see run_recovery): never seed with
# fewer than two nights -- one night gives no direction at all -- and stop
# shrinking once at least this many tracklets remain to be recovered.
MIN_SEED_NIGHTS = 2
MIN_PLANTED_TRACKLETS = 2

# ...and the gap that DEFINES an apparition for each, since the 275 d default
# cannot see a distant object's oppositions as separate at all.
OPPOSITION_GAP = {None: HARNESS_OPPOSITION_GAP_DAYS,
                  "TNO": DISTANT_OPPOSITION_GAP_DAYS,
                  "Centaur": DISTANT_OPPOSITION_GAP_DAYS}


class SearchDeadline(Exception):
    """The collection deadline fired mid-search (see :func:`deadline`)."""


def _watchdog(seconds, rescue):
    """Daemon thread that calls ``rescue()`` and hard-exits after ``seconds``.

    The SIGALRM deadline is a floor, not a guarantee: Python delivers signals
    between bytecodes, so a long call inside a C extension (an ASSIST
    integration, a psycopg2 fetch) defers it indefinitely.  Measured on
    2000 PL8, which sailed 3.5 minutes past a 120 s alarm.  Left to the
    driver's hard timeout that object is SIGKILLed and every row it harvested
    dies with it, because the child only writes at the end.

    A thread is the way out: the GIL is released across exactly those long C
    calls, so this runs when the alarm cannot.  It writes what was harvested
    and then ``os._exit`` -- deliberately not a clean shutdown, because the
    main thread is wedged in C and cannot be unwound from here.
    """
    import threading

    def _fire():
        try:
            rescue()
        finally:
            import os
            os._exit(0)

    t = threading.Timer(seconds, _fire)
    t.daemon = True
    t.start()
    return t


@contextmanager
def deadline(seconds):
    """Raise :class:`SearchDeadline` inside the search after ``seconds``.

    SIGALRM rather than a subprocess kill, because the point is to keep the
    rows: the alarm unwinds the stack through :func:`run_recovery`'s ``finally``,
    which labels and returns everything harvested so far.  A kill would throw
    the object's whole yield away.

    The signal can only be delivered between bytecodes, so a long call inside a
    C extension (an ASSIST integration, a numpy reduction) defers it until it
    returns -- the deadline is therefore a floor, not a guarantee, and the
    collection driver still keeps a hard subprocess timeout above it as the
    backstop.  Main thread only; a no-op if ``seconds`` is falsy.

    Windows has no SIGALRM: there a timer thread raises SIGINT in the process instead,
    delivered between bytecodes exactly as the alarm is (:func:`_deadline_by_sigint`)."""
    import signal
    if not seconds:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        with _deadline_by_sigint(seconds):
            yield
        return

    def _fire(_signum, _frame):
        raise SearchDeadline(f"collection deadline: {seconds:g}s")

    old = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


@contextmanager
def _deadline_by_sigint(seconds):
    """:func:`deadline` where there is no SIGALRM (Windows): a timer thread raises SIGINT,
    whose handler runs on the main thread between bytecodes, as the alarm's does, and also
    wakes a ``time.sleep``. The handler raises :class:`SearchDeadline` only once the timer
    has rung; a real Ctrl-C before then keeps its own meaning. The lock orders the timer
    against the exit, so a SIGINT is never raised after the previous handler is back.

    The signal is aimed at the main thread: ``raise_signal`` delivers to the CALLING thread,
    which on Windows still wakes the main thread's sleep (Python's SIGINT event), but on POSIX
    leaves it asleep until the sleep ends (measured: a 0.3 s deadline fired after 5 s)."""
    import signal
    import threading

    lock = threading.Lock()
    state = {"rang": False, "done": False}
    main = threading.main_thread().ident

    def _ring():
        with lock:
            if state["done"]:
                return
            state["rang"] = True
            if hasattr(signal, "pthread_kill"):
                signal.pthread_kill(main, signal.SIGINT)
            else:
                signal.raise_signal(signal.SIGINT)

    def _fire(signum, frame):
        if state["rang"]:
            raise SearchDeadline(f"collection deadline: {seconds:g}s")
        if callable(old):
            return old(signum, frame)
        raise KeyboardInterrupt

    old = signal.signal(signal.SIGINT, _fire)
    timer = threading.Timer(float(seconds), _ring)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        with lock:
            state["done"] = True
        timer.cancel()
        try:
            # signal.signal runs a still-pending ring first (raising SearchDeadline, as a
            # SIGALRM landing on the way out would), so restore again behind it
            signal.signal(signal.SIGINT, old)
        finally:
            signal.signal(signal.SIGINT, old)


def fetch_object_obs(desig, obscodes):
    """Published obs for one designation, time-sorted, every one carrying an
    80-column line.

    ADES is fetched (the engine default), so the seed the harness hands the
    search is the same ADES-quality data a live search seeds with -- otherwise
    the rig would exercise a weaker fit than the thing it is validating.  The
    LINE is still what the harness manipulates (apparitions are planted by
    re-labelling lines under synthetic trksubs), so an ADES-only obs gets one
    synthesized here, exactly as the search's own seed path does."""
    from pyroorb.obs_sources import fetch_observations
    from pyroorb.obs80 import obs80_base_line
    obs, meta = fetch_observations(desig, obscodes)
    out = []
    for o in obs:
        if not o.get("obs80_line"):
            try:
                parts = obs80_base_line(o).splitlines()
            except Exception:
                continue                 # unrenderable -> not plantable
            o = {**o, "obs80_line": parts[0]}
            if len(parts) > 1:
                o["obs80_line2"] = parts[1]
        out.append(o)
    out.sort(key=lambda o: o["jd_utc"])
    return out, meta


def split_oppositions(obs, gap_days=HARNESS_OPPOSITION_GAP_DAYS):
    """Group time-sorted obs into apparitions (a gap over ``gap_days`` starts a
    new one).  Distant objects need a shorter gap -- see
    :data:`DISTANT_OPPOSITION_GAP_DAYS`."""
    groups = []
    run = []
    last = None
    for o in obs:
        if last is not None and o["jd_utc"] - last > gap_days:
            groups.append(run)
            run = []
        run.append(o)
        last = o["jd_utc"]
    if run:
        groups.append(run)
    return groups


def obs_lines(group):
    out = []
    for o in group:
        out.append(o["obs80_line"])
        if o.get("obs80_line2"):
            out.append(o["obs80_line2"])
    return "\n".join(out) + "\n"


def _fake_line(line, trksub):
    """Re-label an obs80 line as an ITF line under a synthetic trksub
    (number columns blanked, provisional columns replaced)."""
    return ("     " + f"{trksub:<7.7}" + line[12:]).ljust(80)[:80]


def fake_from_run(run, trksub, opposition, index_cfg):
    """ONE fake-ITF tracklet from ``run`` (time-sorted published obs of one
    station), under ``trksub``, through the same admission the index applies
    to a real row.  Returns ``(fake, None)`` or ``(None, reason)``."""
    if len(run) < 2:
        return None, "single-obs tracklet (no direction)"
    feat_rows = []
    obs_rows = []
    for o in run:
        es = unix_from_jd(o["jd_utc"])
        line = _fake_line(o["obs80_line"], trksub)
        feat_rows.append({
            "tracklet_id": trksub, "es": es,
            "ra": o["ra_deg"], "dec": o["dec_deg"],
            # the REPORTED rms, as the index reads obs_sbn.rmsra/rmsdec --
            # NOT sigma_ra, the fit's weighting sigma (a Veres2017 model value
            # wherever no rms was reported).  Measured on 112 ex-ITF tracklets
            # built both ways: the weighting sigma halved sig_pos/sig_vel/
            # rate_sigma on 68% of them and quadrupled chi2_dof, so a planted
            # tracklet's own features said "planted" (itf/exitf.py)
            "rms_ra": o.get("ades_rms_ra"), "rms_dec": o.get("ades_rms_dec"),
            "mag": o.get("mag"), "band": o.get("band"),
            "trksub": trksub, "stn": o["obscode"], "provid": None,
        })
        obs_rows.append({
            "obs80": line, "ra": o["ra_deg"], "dec": o["dec_deg"],
            "epoch_sec": es, "trksub": trksub, "stn": o["obscode"],
            "trkid": trksub,
            # the ADES payload a REAL obs_sbn row carries, under the
            # same column names, so a planted tracklet exercises the
            # registry path (itf/obsreg.py) exactly like a live one
            "obsid": o.get("obs_id"), "mode": o.get("mode"),
            "astcat": o.get("astcat"),
            "rmsra": o.get("ades_rms_ra"),
            "rmsdec": o.get("ades_rms_dec"),
            "rmsmag": o.get("mag_sigma"), "logsnr": o.get("log_snr"),
            "rmstime": o.get("sigma_time_sec"),
        })
    feat = compute_tracklet_features(feat_rows, index_cfg)
    if feat is None:
        return None, "degenerate (single epoch)"
    return {"feat": feat, "obs": obs_rows, "apparition": opposition}, None


def make_fake_itf(opposition_groups, index_cfg=None, trk_prefix="PR"):
    """Fake-ITF tracklets from published apparitions.

    Tracklets are formed per station with <= 0.5 d internal spacing (the
    single-night shape of real ITF tracklets).  Each candidate must pass the
    same admission criteria the index applies to real rows -- >= 2 obs and a
    non-degenerate epoch spread, enforced by running the actual feature
    computation.  Two-line satellite obs are skipped (the corridor search is
    ground-based geometry).

    Returns ``(fakes, rejected)``: overlay-ready ``{"feat", "obs",
    "apparition"}`` dicts, and ``[(trksub, reason)]`` for candidates that a
    real ITF ingest would also have dropped.
    """
    index_cfg = index_cfg or IndexConfig()
    fakes = []
    rejected = []
    seq = 0
    for app_i, group in enumerate(opposition_groups):
        singles = [o for o in group
                   if not o.get("obs80_line2") and o.get("obs80_line")]
        runs = []
        by_stn = sorted(singles, key=lambda o: (o["obscode"], o["jd_utc"]))
        run = []
        last = None
        for o in by_stn:
            if run and (o["obscode"] != last["obscode"]
                        or o["jd_utc"] - last["jd_utc"] > FAKE_RUN_GAP_DAYS):
                runs.append(run)
                run = []
            run.append(o)
            last = o
        if run:
            runs.append(run)

        for run in runs:
            # NB the counter restarts per CALL, so trksubs collide across
            # objects: object A's ZT00001 and object B's ZT00001 are unrelated
            # tracklets.  That is harmless to the search (a trksub only has to
            # be unique within one run) but any ANALYSIS across objects must
            # key on (designation, trksub) -- counting trksub alone undercounted
            # a collection's distinct truths 17-fold.  Not changed to a globally
            # unique id because this string is written into the obs80 line's
            # designation columns, where it has to keep looking like a real ITF
            # trksub.
            trksub = f"{trk_prefix}{seq:05d}"[:7]
            seq += 1
            fk, why = fake_from_run(run, trksub, app_i, index_cfg)
            if fk is None:
                rejected.append((trksub, why))
            else:
                fakes.append(fk)
    return fakes, rejected


@dataclass
class RecoveryReport:
    designation: str
    ok: bool = False
    status: float = 0.0
    reason: str = ""
    n_apparitions: int = 0
    seed_apparition: int = -1
    n_fake_tracklets: int = 0
    n_rejected_fakes: int = 0
    planted_apparitions: set = field(default_factory=set)
    recovered_apparitions: set = field(default_factory=set)
    recovered_tracklets: int = 0
    # real-ITF tracklets that ended up in the linkage (live mode): each is a
    # potential genuine find -- or a contamination
    foreign_tracklets: list = field(default_factory=list)
    # each foreign tracklet judged against the object's own full-arc orbit
    # (:func:`label_foreign`): trksub -> {"miss_arcsec", "label"}
    foreign_labels: dict = field(default_factory=dict)
    seed_obs: int = 0
    linked_obs: int = 0
    seed_apparitions_used: int = 1
    seed_nights_used: int = 0
    # the collection deadline fired: rows were kept, the verdict was not reached
    truncated: bool = False
    # seed observations the recovery THREW AWAY -- clipped as poor fitters, or
    # missing from the accepted astrometry altogether.  A recovery is supposed
    # to ADD observations; every one of the object's own that it drops is a
    # claim that the published astrometry is wrong, and 2024 PQ7 (a real
    # linkage, MPC-accepted) had three dropped by a flat arcsec cut that took
    # no account of the station's own uncertainty.  Fixtures pin this at 0.
    dropped_seed_obs: int = 0
    # (first, last) JD of the seed actually used, so a caller can rebuild it
    seed_jd: tuple = None
    rms: float = 0.0
    # the search's raw telemetry, so a harness run can be measured and not just
    # scored -- notably telemetry["memory"], the peak-RSS profile needed to turn
    # ItfConfig.sweep_min_free_mb from an anecdote into a number
    telemetry: dict = field(default_factory=dict)

    @property
    def false_foreign(self):
        """Foreign tracklets the object's full-arc orbit rules out: a false link."""
        return sorted(k for k, v in self.foreign_labels.items() if v["label"] == "false")

    @property
    def full_recovery(self):
        return (self.ok and self.planted_apparitions
                and self.recovered_apparitions == self.planted_apparitions)

    def summary(self):
        foreign = (f" foreign={len(self.foreign_tracklets)}"
                   if self.foreign_tracklets else "")
        if self.false_foreign:
            foreign += f" FALSE={len(self.false_foreign)}"
        return (f"{self.designation:>12}  status={self.status:<3} "
                f"apps={len(self.recovered_apparitions)}/{len(self.planted_apparitions)} "
                f"trk={self.recovered_tracklets}/{self.n_fake_tracklets}"
                + (f" DROPPED_SEED={self.dropped_seed_obs}"
                   if self.dropped_seed_obs else "")
                + f"{foreign} obs+{self.linked_obs} "
                + f"rms={self.rms:.3f}\"  {self.reason}")


def observing_nights(group):
    """Split a time-sorted obs group into actual OBSERVING nights (a gap of
    more than ``orbit_common.NIGHT_GAP_DAYS`` between consecutive observations
    opens a new one).

    NB this is deliberately NOT :func:`itf.opp_profile.spans_of`'s 36-hour SPAN, and
    the difference is not an oversight.  That unit is the ported notebook
    convention the ACCEPTANCE GATES are calibrated on (``spans >= 8 / 6 / 4``),
    it measures from the span's START, and it reads observations 1.0 d apart as
    ONE span on purpose.  Sound for gating; wrong for building a seed, because
    it collapses a 3-consecutive-night discovery arc into a single span --
    which is exactly how a 2-day, 3-night arc came to look unfittable.  Seed
    construction wants real nights; the gates keep their counter.  Two rules,
    each named for what it does.
    """
    from pyroorb.orbit_common import NIGHT_GAP_DAYS
    out = [[group[0]]]
    for o in group[1:]:
        if o["jd_utc"] - out[-1][-1]["jd_utc"] > NIGHT_GAP_DAYS:
            out.append([])
        out[-1].append(o)
    return out


def designation_year(desig):
    """Discovery year of a provisional designation (``2024 NH2`` -> 2024), or
    None for anything else (a number, a trksub, a survey designation)."""
    m = re.match(r"\s*(1[89]|20)(\d\d)\s+[A-Z]{2}", str(desig or ""))
    return int(m.group(1) + m.group(2)) if m else None


def _opposition_years(group):
    return {datetime.fromtimestamp(unix_from_jd(o["jd_utc"]), timezone.utc).year
            for o in group}


def pick_seed_opposition(apps, min_nights=SEED_MIN_NIGHTS, desig=None,
                         min_obs=SEED_MIN_OBS):
    """Index of the apparition to seed a recovery with: the object's DISCOVERY
    apparition when it carries at least ``min_nights`` observing nights,
    otherwise the most compact apparition that does.

    A recovery harness is only interesting when the seed is the arc a real
    discovery would have had -- a handful of nights over a few days, from which
    everything else must be found.  Two wrong answers this has already given,
    both worth keeping named:

    * the FATTEST apparition (``max(apps, key=len)``, the original) hands the
      search a well-observed arc no recovery ever gets;
    * the most COMPACT >= 3-night apparition, which replaced it, minimises
      SPAN -- and span says nothing about whether an arc carries information.
      On 2024 NH2 it preferred a 7-obs/7-night/36 d *precovery* apparition
      (1.0 obs per night) over the real 28-obs/10-night/57 d 2024 discovery
      arc, and its first 3 nights are then 3 single-observation nights: six
      measurements against six orbital elements, exactly determined, fitting
      itself to 0.011" at U 15.25.  Such a seed cannot accept ANY extension --
      ``extension_degrades``' floor sits below real astrometric noise -- so the
      object is unrecoverable by arithmetic before the search starts.

    The observation floor is the third wrong answer, and the quietest: an
    apparition of 3 observations clears "3 nights" while carrying exactly as
    many measurements as there are orbital elements.  Such a fit is EXACTLY
    determined -- it reproduces its own astrometry to 0.000", its covariance
    collapses to numerical zero, and U comes out 0.00, which every downstream
    consumer reads as a superbly determined orbit.  In the search that switches
    OFF the under-determined-seed fallback (gated on U >= 4) and pencils the
    corridor to nothing: 2013 UL49 served 0 of 5 planted tracklets that way,
    seeded on its 3-obs 2018 apparition while an 8-obs/4-night 2017 one sat
    unused.  ``min_obs`` (6, so 12 measurements against 6 elements) keeps a
    seed that can actually carry information.

    The discovery apparition is the one the designation names (``2024 NH2`` ->
    the apparition holding 2024 observations), which is what "seed it with the
    discovery arc" means and what a precovery-bearing NEO needs.  ``desig``
    None, unparseable, or naming no qualifying apparition falls back to
    most-compact; no apparition reaching ``min_nights`` falls back to the one
    with the most nights (ties to the more compact), so the caller always gets
    the best seed the object can support.
    """
    scored = [(i, len(observing_nights(g)), g[-1]["jd_utc"] - g[0]["jd_utc"],
               len(g)) for i, g in enumerate(apps)]
    qualified = [s for s in scored if s[1] >= min_nights and s[3] >= min_obs]
    if not qualified:
        # nothing qualifies: the most informative arc available -- most nights,
        # then most observations, then the more compact of the ties
        return max(scored, key=lambda s: (s[1], s[3], -s[2]))[0]
    year = designation_year(desig)
    if year is not None:
        for i, _n, _span, _obs in qualified:
            if year in _opposition_years(apps[i]):
                return i
    return min(qualified, key=lambda s: s[2])[0]


def _first_n_nights(group, n_nights):
    """Split a time-sorted obs group into (first ``n_nights`` observing nights,
    remainder). See :func:`observing_nights` for why the night rule here is not
    the gates' 1.6 d one."""
    head = [o for night in observing_nights(group)[:n_nights] for o in night]
    return head, group[len(head):]


def compact_seed_window(apps, n_nights, min_obs=SEED_MIN_OBS):
    """The most compact ``n_nights``-night seed the object offers: over every
    apparition, every run of ``n_nights`` CONSECUTIVE observing nights, the one
    with the shortest time span.

    Returns ``(apparition_index, head, before, after)`` -- the seed
    observations and the parts of that apparition on either side of it, which
    are recoverable truth like every other apparition -- or None when no
    apparition has ``n_nights`` nights.

    Shortest span is the hardest honest seed: a 3-night arc over 2 days
    extrapolates far worse than the same nights spread over a month, and it is
    the shape a real discovery hands the search.  Windows carrying fewer than
    ``min_obs`` observations are passed over while any window has enough (see
    SEED_MIN_OBS: fewer than 6 observations is an exactly-determined fit that
    reports a perfect orbit); if none does, the floor is dropped rather than
    the object.  Ties on span go to the window with more observations."""
    windows = []
    for i, g in enumerate(apps):
        nights = observing_nights(g)
        for k in range(len(nights) - n_nights + 1):
            win = nights[k:k + n_nights]
            span = win[-1][-1]["jd_utc"] - win[0][0]["jd_utc"]
            windows.append((i, k, span, sum(len(n) for n in win)))
    if not windows:
        return None
    rich = [w for w in windows if w[3] >= min_obs] or windows
    i, k, _span, _n = min(rich, key=lambda w: (w[2], -w[3]))
    nights = observing_nights(apps[i])
    flat = lambda ns: [o for n in ns for o in n]
    return (i, flat(nights[k:k + n_nights]), flat(nights[:k]),
            flat(nights[k + n_nights:]))


def thin_fakes(fakes, max_planted):
    """Keep at most ``max_planted`` planted tracklets, spread ROUND-ROBIN across
    apparitions so every apparition keeps representation.

    A real ITF linkage is 10-15 tracklets at the very most:
    an object with 50+ unattributed tracklets sitting in the file would have
    been linked years ago.  But the harness plants EVERY tracklet of every
    non-seed apparition, so a well-observed object arrives with 55, 71, 108 --
    a shape the ITF never actually presents, and one that runs into find_orb's
    50-id document cap (MAX_LINKAGE_IDS), which then truncates the SEARCH:
    2008 KV2 had a 72-id linkage in hand at depth 5, was refused, and fell back
    to depth 4's 46 tracklets.  Thinning asks the question the file would
    really ask instead of engineering around a cap that guards a case that
    cannot occur.

    Deterministic: apparitions in index order, tracklets in epoch order within
    each, taken one apparition at a time until the budget is spent."""
    if not max_planted or len(fakes) <= max_planted:
        return fakes
    by_app = {}
    for fk in sorted(fakes, key=lambda f: f["feat"]["epoch_start_sec"]):
        by_app.setdefault(fk["apparition"], []).append(fk)
    order = sorted(by_app)
    kept, i = [], 0
    while len(kept) < max_planted and any(by_app[a] for a in order):
        app = order[i % len(order)]
        if by_app[app]:
            kept.append(by_app[app].pop(0))
        i += 1
    return sorted(kept, key=lambda f: f["feat"]["epoch_start_sec"])


#: A foreign tracklet whose every observation lies within this of the object's full-arc orbit is
#: the object's own (a genuine find); one with any observation past FOREIGN_FALSE_ARCSEC is
#: another body; between the two it is left "unclear" for a person.  Measured on live weak-seed
#: scans: genuine finds missed by 0.3-2.4", false tracklets by 7,800-580,000", one case at 44".
FOREIGN_GENUINE_ARCSEC = 5.0
FOREIGN_FALSE_ARCSEC = 60.0


def label_foreign(full_fit, itf_obs80, foreign, obscodes):
    """Judge each foreign tracklet of a linkage against the object's own full-arc orbit.

    ``itf_obs80`` is the linkage's ITF lines (``ItfSearchResult.itf_obs80``), ``foreign`` the
    trksubs (columns 6-12) to judge.  Returns trksub -> ``{"miss_arcsec": the largest miss of
    its observations, "label": "genuine" | "false" | "unclear"}``.  A recovery scan otherwise
    only lists foreign tracklets, and a false link reads like a find until someone checks it.
    """
    import math
    from pyroorb.ephemerides import compute_ephemeris
    from pyroorb.obs_sources import read_obs_file
    wanted = set(foreign)
    miss = {}
    for line in (itf_obs80 or "").splitlines():
        sub = line[5:12].strip()
        if not line.strip() or sub not in wanted:
            continue
        parsed, _ = read_obs_file(line + "\n", obscodes, in_format="obs80")
        if not parsed:
            continue
        o = parsed[0]
        rows = compute_ephemeris(full_fit, obscodes, o["jd_utc"], o["jd_utc"], 1.0,
                                 site=o["obscode"], full=False)
        if not rows:
            continue
        e = rows[0]
        dra = ((o["ra_deg"] - e["ra_deg"] + 180.0) % 360.0 - 180.0) \
            * math.cos(math.radians(o["dec_deg"])) * 3600.0
        ddec = (o["dec_deg"] - e["dec_deg"]) * 3600.0
        miss[sub] = max(miss.get(sub, 0.0), math.hypot(dra, ddec))
    return {k: {"miss_arcsec": v,
                "label": ("genuine" if v < FOREIGN_GENUINE_ARCSEC
                          else "false" if v > FOREIGN_FALSE_ARCSEC else "unclear")}
            for k, v in miss.items()}


def run_recovery(desig, cfg=None, index_cfg=None, base_source=None, obscodes=None,
                 seed_apparition=None, seed_nights=None, do_submit=False,
                 log=print, verbose=False, collector_rows=None, obs_meta=None,
                 trace=None, max_planted=None, deadline_sec=None,
                 seed_oppositions=1, rescue_write=None,
                 opposition_gap_days=HARNESS_OPPOSITION_GAP_DAYS,
                 adaptive_seed=False, hold_out_stations=(), dlog=None,
                 compact_seed=False):
    """One recovery case: seed with a single apparition, plant the rest as
    fake ITF, search the merged (real + fake) ITF, score the recovery.

    ``base_source`` -- optional live :class:`~pyroorb.itf.candidates.DbSource`;
    None = offline (fake-only).  ``seed_apparition`` -- index of the
    apparition to seed with (default: :func:`pick_seed_opposition` -- the most
    compact one with >= 3 observing nights).  A caller that must pin the seed
    across code changes (a frozen fixture) passes the index EXPLICITLY rather
    than inheriting whatever the default currently resolves to.
    ``seed_nights`` -- seed with only the first N nights of that apparition
    and plant its REMAINING tracklets as recoverable fakes too (the ML
    data-generation shape).  ``hold_out_stations`` -- observatory codes
    whose obs in the seeded apparition(s) are planted instead of seeded: a
    tracklet INSIDE the seed's own arc (2015 KK542's W84 night sits between
    its T14 nights), which ``seed_nights``' contiguous split cannot reach.
    ``compact_seed`` -- with ``seed_nights``, seed with the most compact run of
    that many consecutive nights in ANY apparition (:func:`compact_seed_window`)
    instead of the first nights of the picked one; the rest of that apparition,
    before and after the window, is planted.
    ``collector_rows`` -- a list to receive labeled ML feature rows (see :mod:`pyroorb.itf.training`).  ``trace`` -- optional
    decision-feed sink (:mod:`pyroorb.itf.trace`), so a harness run can be read the
    same way a CLI or web search can.  ``obs_meta`` -- a pre-fetched
    ``(obs, meta)`` to use INSTEAD of fetching from the MPC: a regression
    fixture feeds its FROZEN published astrometry here so the seed is
    deterministic while the (optional) live ``base_source`` still supplies real
    ITF distractors (see tests/build_itf_fixtures.py).
    Submission is forced into dry-run regardless of flags.
    """
    cfg = cfg or ItfConfig()
    index_cfg = index_cfg or IndexConfig()
    if obscodes is None:
        from pyroorb.obs_sources import load_obscodes
        obscodes = load_obscodes()

    rep = RecoveryReport(designation=str(desig))
    try:
        obs, meta = obs_meta if obs_meta is not None else \
            fetch_object_obs(desig, obscodes)
    except Exception as e:
        rep.reason = f"fetch failed: {e}"
        return rep
    if not obs:
        rep.reason = "fetch failed: no observations returned"
        return rep
    apps = split_oppositions(obs, opposition_gap_days)
    rep.n_apparitions = len(apps)
    # one apparition is enough when part of it is held back to plant
    if len(apps) < 2 and seed_nights is None and not hold_out_stations:
        rep.reason = f"only {len(apps)} apparition(s) fetched"
        return rep

    compact = (compact_seed_window(apps, seed_nights)
               if compact_seed and seed_nights and seed_apparition is None
               else None)
    if compact is not None:
        seed_apparition = compact[0]
    if seed_apparition is None:
        seed_apparition = pick_seed_opposition(apps, desig=desig)
    rep.seed_apparition = seed_apparition
    seed_group = apps[seed_apparition]
    others = [g for i, g in enumerate(apps) if i != seed_apparition]
    # A DISTANT object cannot be seeded from ONE apparition.  Measured on
    # 2018 FL76: a 72-day single-apparition TNO arc (10 obs, rms 0.104",
    # U 7.40) turns 3611 ephemeris nodes into ZERO windows, because every one
    # comes out wider than cfg.max_uncert_deg -- the search has nothing to
    # sweep and reports "no likely matches" with the planted tracklets sitting
    # right there in the overlay.  A real ITF search for a TNO is seeded from a
    # multi-apparition orbit for exactly that reason, so seeding more than one
    # here makes the rig MORE like the thing it stands in for, not less.
    # Apparitions adjacent to the seed are the ones consumed, leaving the
    # temporally DISTANT ones to be planted -- the hard recoveries.
    seeded = [seed_group]
    if seed_oppositions and seed_oppositions > 1 and len(others) > 1:
        take = min(seed_oppositions - 1, len(others) - 1)
        nxt = sorted(others, key=lambda g: abs(g[0]["jd_utc"]
                                               - seed_group[0]["jd_utc"]))[:take]
        for g in nxt:
            others.remove(g)
        seeded += nxt
    rep.seed_apparitions_used = len(seeded)
    if hold_out_stations:
        held = [o for g in seeded for o in g
                if o["obscode"] in hold_out_stations]
        seeded = [[o for o in g if o["obscode"] not in hold_out_stations]
                  for g in seeded]
        if held:
            others.append(held)
    def _build(n):
        """(seed_group, others) for a seed of ``n`` nights per apparition.

        Pure -- it never mutates the caller's ``others`` -- because the
        adaptive loop below calls it repeatedly and a leaked append would
        accumulate plants across attempts."""
        if n is None:
            return sorted((o for g in seeded for o in g),
                          key=lambda o: o["jd_utc"]), list(others)
        # the night limit is PER seeded apparition -- applying it to the
        # concatenation would take every night from the earliest one and leave
        # the others unrepresented, which is the single-apparition seed again
        kept, rest_groups = [], list(others)
        for g in seeded:
            head, rest = _first_n_nights(g, n)
            kept.extend(head)
            if len(rest) >= 2:
                # the rest of a seeded apparition is recoverable truth as well
                rest_groups.append(rest)
        return sorted(kept, key=lambda o: o["jd_utc"]), rest_groups

    # ADAPTIVE seed: the largest seed that still leaves something to recover.
    #
    # A fixed night count decides how much of the object to keep without
    # looking at how much there IS, so on a short arc it eats the whole object
    # and the run yields nothing -- four NEOs died that way, not because their
    # arcs were short but because the count happened to draw 4 instead of 3.
    # Measured on those four, dropping to a 2-night seed rescued all of them
    # AND produced more positives than 3 nights did on every one (12 vs 5),
    # because a smaller seed leaves more truth to plant.  Shrinking is safe for
    # near objects; it is NOT unbounded, since a seed too loose to constrain a
    # corridor builds zero windows (see DISTANT_OPPOSITION_GAP_DAYS) -- hence
    # the floor, and hence taking the LARGEST workable seed rather than the
    # smallest.
    if compact is not None:
        _i, seed_group, before, after = compact
        others = list(others) + [g for g in (before, after) if len(g) >= 2]
        rep.seed_nights_used = seed_nights
    else:
        seed_group, others = _build(seed_nights)
    rep.seed_jd = ((seed_group[0]["jd_utc"], seed_group[-1]["jd_utc"])
                   if seed_group else None)
    if seed_nights is not None and adaptive_seed and compact is None:
        for n in range(seed_nights, MIN_SEED_NIGHTS - 1, -1):
            cand_seed, cand_others = _build(n)
            if len(cand_seed) < 3:
                continue
            n_fakes = len(make_fake_itf(cand_others, index_cfg)[0])
            seed_group, others = cand_seed, cand_others
            rep.seed_nights_used = n
            if n_fakes >= MIN_PLANTED_TRACKLETS:
                break
    if seed_nights is not None and len(seed_group) < 3:
        rep.reason = f"seed too small ({len(seed_group)} obs)"
        return rep

    fakes, rejected = make_fake_itf(others, index_cfg)
    fakes = thin_fakes(fakes, max_planted)
    rep.n_fake_tracklets = len(fakes)
    rep.n_rejected_fakes = len(rejected)
    rep.planted_apparitions = {fk["apparition"] for fk in fakes}
    rep.seed_obs = len(seed_group)
    if not fakes:
        rep.reason = "no plantable tracklets in the other apparitions"
        return rep

    source = OverlaySource(base_source, fakes, cfg, index_cfg)
    seed_text = obs_lines(seed_group)
    telemetry = {}
    # The collector list is handed to the search DIRECTLY rather than filled
    # from a scratch list at the end: a collection run puts a deadline on each
    # object, and rows already harvested when it fires are exactly as valid as
    # rows from a search that ran to completion.  Accumulating into the
    # caller's list means an interrupted search still yields its data.
    raw_rows = collector_rows
    result, truncated = None, False

    def _label(result, was_truncated):
        """Label whatever was harvested: planted (MPC-linked truth) -> 1; a
        real ITF tracklet that ended up linked -> presumed positive but
        flagged; every other served distractor -> 0.

        planted/distractor do NOT depend on the search's outcome, so they stand
        on a truncated run.  linked_foreign DOES (it reads the accepted
        astrometry), so on truncation it is unavailable and those rows stay
        labelled 'distractor' -- flagged by the `truncated` column so a retrain
        can drop them if that presumed positive matters."""
        if raw_rows is None:
            return
        planted = {fk["feat"]["trksub"] for fk in fakes}
        linked = ({l[5:12].strip() for l in result.itf_obs80.splitlines()
                   if l.strip()}
                  if result is not None and result.itf_obs80 else set())
        for row in raw_rows:
            row["designation"] = str(desig)
            row["truncated"] = int(was_truncated)
            if row["trksub"] in planted:
                row["label"] = 1
                row["label_source"] = "planted"
            elif row["trksub"] in linked:
                row["label"] = 1
                row["label_source"] = "linked_foreign"
            else:
                row["label"] = 0
                row["label_source"] = "distractor"

    # Backstop for a deadline the signal cannot deliver (see _watchdog): label
    # and hand back what we have, then exit the process.  Armed only when the
    # caller supplied a way to persist rows from another thread.
    if deadline_sec and rescue_write is not None and raw_rows is not None:
        def _rescue():
            _label(None, True)
            rescue_write(raw_rows)
        _watchdog(deadline_sec + 45.0, _rescue)

    try:
        with deadline(deadline_sec):
            result = itf_search(source, seed_obs80=seed_text,
                                seed_obs=seed_group,
                                primary_desigs=[meta.get("designation",
                                                         str(desig))],
                                cfg=cfg, obscodes=obscodes,
                                telemetry=telemetry,
                                do_submit=False,  # harness NEVER submits
                                verbose=verbose, log=log, collector=raw_rows,
                                trace=trace, dlog=dlog)
    except SearchDeadline:
        truncated = True
        log(f"{desig}: deadline hit -- keeping {len(raw_rows or [])} "
            f"harvested rows, stopping the search")
    finally:
        _label(result, truncated)

    rep.truncated = truncated
    if result is None:
        rep.reason = "search truncated at the collection deadline"
        rep.status = 0
        return rep

    rep.ok = result.verdict is SearchVerdict.LINKED
    rep.status = result.verdict.legacy_status   # the report JSON keeps the float
    rep.reason = result.reason
    rep.linked_obs = telemetry.get("obs_count", 0)
    rep.telemetry = telemetry
    if result.fit is not None:
        rep.rms = float(result.fit.rms_residual())
    # what became of the seed's OWN observations: present in the accepted
    # astrometry AND included in the final fit, or dropped
    if (result.verdict in (SearchVerdict.LINKED, SearchVerdict.POSSIBLE)
            and result.fit is not None):
        seed_ids = {obs_unique_id(l) for l in seed_text.splitlines() if l.strip()}
        kept = set()
        fit_lines = [o.get("obs80_line") for o in result.fit._obs]
        inc = result.fit.is_included()
        for line, ok in zip(fit_lines, inc):
            if line and ok:
                kept.add(obs_unique_id(line))
        rep.dropped_seed_obs = len(seed_ids - kept)
    if result.itf_obs80:
        got = {l[5:12].strip() for l in result.itf_obs80.splitlines()
               if l.strip()}
        fake_subs = set()
        for fk in fakes:
            fake_subs.add(fk["feat"]["trksub"])
            if fk["feat"]["trksub"] in got:
                rep.recovered_tracklets += 1
                rep.recovered_apparitions.add(fk["apparition"])
        rep.foreign_tracklets = sorted(got - fake_subs)
        if rep.foreign_tracklets:
            # The object's own observations, every apparition, are the truth to judge by.
            try:
                from pyroorb.fit_object import fit_orbit
                full = fit_orbit(obs, obscodes, desig=meta.get("designation", str(desig)),
                                 meta=meta)
                rep.foreign_labels = label_foreign(full, result.itf_obs80,
                                                   rep.foreign_tracklets, obscodes)
            except Exception as e:                    # noqa: BLE001 -- a label, not a verdict
                log(f"{desig}: foreign tracklets left unlabelled ({type(e).__name__}: {e})")
    return rep


def run_suite(designations, cfg=None, base_source=None, log=print,
              verbose=False, report_path=None, seed_nights=None):
    """Run the recovery case over a list of designations; returns the report
    list and prints a summary table.

    ``seed_nights`` -- seed each object with only the first N nights of its
    seed opposition (``--seed-nights`` on the CLI).  With ``SEED_MIN_NIGHTS``
    that is the discovery-arc stress shape: a ~2-day, 3-night seed, everything
    else planted as recoverable ITF.
    """
    cfg = cfg or ItfConfig()
    from pyroorb.obs_sources import load_obscodes
    obscodes = load_obscodes()
    reports = []
    for d in designations:
        log(f"=== {d} ===")
        rep = run_recovery(d, cfg=cfg, base_source=base_source,
                           obscodes=obscodes, log=log, verbose=verbose,
                           seed_nights=seed_nights)
        log(rep.summary())
        reports.append(rep)
    n_full = sum(1 for r in reports if r.full_recovery)
    n_found = sum(1 for r in reports if r.ok)
    log("\n=== ITF recovery suite ===")
    for r in reports:
        log(("FULL " if r.full_recovery else ("part " if r.ok else "FAIL "))
            + r.summary())
    log(f"{n_full}/{len(reports)} full recoveries, {n_found} linkages found")
    if report_path:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump([{**r.__dict__,
                        "planted_apparitions": sorted(r.planted_apparitions),
                        "recovered_apparitions": sorted(r.recovered_apparitions),
                        "full_recovery": r.full_recovery} for r in reports],
                      fh, indent=2, default=str)
        log(f"report written to {report_path}")
    return reports


def sample_multi_opposition_objects(n=10, seed=42, n_oppositions=(5, 5)):
    """n random objects from MPCORB (cached download) whose opposition count is
    in ``n_oppositions`` (inclusive; five exactly by default -- what the
    recovery suite has always drawn)."""
    from pyroorb.mpcorb import sample_asteroids
    return sample_asteroids(n, seed=seed, nopp=tuple(n_oppositions))


def generate_training_data(n_objects, out_csv, seed=42, nopp=(1, 6),
                           base_source=None, cfg=None, log=print):
    """ML dataset generation: run the synthetic recovery on ``n_objects``
    random 1-6-opposition objects, seeding each with the first 3-4 nights of
    one opposition, and append one labeled feature row per candidate tracklet
    per recursion level to ``out_csv`` (crash-safe incremental writes).

    Positives = the object's own (MPC-linked) planted tracklets; negatives =
    real ITF distractors served into the same corridors -- so this is only
    meaningful with a LIVE ``base_source``.
    """
    import random as _random

    from pyroorb.mpcorb import sample_asteroids
    from pyroorb.obs_sources import load_obscodes

    from .training import write_rows

    cfg = cfg or ItfConfig()
    obscodes = load_obscodes()
    rng = _random.Random(seed)
    desigs = sample_asteroids(n_objects, seed=seed, nopp=nopp)
    tot_rows = tot_pos = tot_neg = n_ok = 0
    for i, d in enumerate(desigs, 1):
        rows = []
        try:
            rep = run_recovery(d, cfg=cfg, base_source=base_source,
                               obscodes=obscodes,
                               seed_nights=rng.choice((3, 4)),
                               collector_rows=rows,
                               log=lambda *a: None)   # per-object noise; the
                                                      # summary below is the log
        except Exception as e:
            log(f"[{i}/{len(desigs)}] {d}: CRASH {e}")
            continue
        write_rows(out_csv, rows)
        pos = sum(1 for r in rows if r["label"] == 1)
        tot_rows += len(rows); tot_pos += pos; tot_neg += len(rows) - pos
        n_ok += 1
        log(f"[{i}/{len(desigs)}] {d}: {rep.summary()}")
        log(f"    +{len(rows)} rows ({pos} pos) -> totals {tot_rows} rows, "
            f"{tot_pos} pos / {tot_neg} neg")
    log(f"\ndataset done: {tot_rows} rows ({tot_pos} pos / {tot_neg} neg) "
        f"from {n_ok}/{len(desigs)} objects -> {out_csv}")
    return tot_rows


def collect_one(obj, cfg=None, base_source=None, obscodes=None, seed=777,
                deadline_sec=120.0, log=print, rescue_write=None):
    """Collect one sampled object's labeled rows (see :mod:`itf.collect`).

    ``obj`` is a :class:`itf.sampling.SampledObject`; its stratum and split ride
    onto every row so the CSV is self-describing.  Returns ``(report, rows)``;
    rows come back labeled even when the deadline truncated the search."""
    import random as _random

    from pyroorb.obs_sources import load_obscodes

    from .training import split_key

    cfg = cfg or ItfConfig()
    obscodes = obscodes if obscodes is not None else load_obscodes()
    # the seed-night count is drawn per OBJECT from the object's own name, so a
    # re-run (or a resume) gives that object the same seed shape it had before
    rng = _random.Random(f"{seed}:{obj.designation}")
    nights = rng.choice(SEED_NIGHTS.get(obj.stratum, SEED_NIGHTS[None]))
    key = split_key(obj.designation)

    def _stamp(rs):
        """Object-level columns onto every row.  Also used by the watchdog
        rescue, which writes from another thread mid-search."""
        for r in rs:
            r["split"] = obj.split
            r["split_key"] = key
            r["stratum"] = obj.stratum
        return rs
    rows = []
    rep = run_recovery(obj.designation, cfg=cfg, base_source=base_source,
                       obscodes=obscodes, seed_nights=nights,
                       seed_oppositions=SEED_OPPOSITIONS.get(
                           obj.stratum, SEED_OPPOSITIONS[None]),
                       opposition_gap_days=OPPOSITION_GAP.get(
                           obj.stratum, OPPOSITION_GAP[None]),
                       adaptive_seed=True,
                       collector_rows=rows, deadline_sec=deadline_sec,
                       rescue_write=(
                           (lambda rs: rescue_write(_stamp(rs)))
                           if rescue_write else None),
                       log=log)
    return rep, _stamp(rows)
