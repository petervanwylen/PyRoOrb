"""Scoring driver: motion vector -> orbit search -> class percentages.

Ports ``d2math.c``'s ``score()`` -- the function that sets up the two
observation vectors, runs the adaptive orbit search over distance and
angle, and turns the resulting tagged-bin population sums into the raw
and no-ID percentages.

The search itself -- the ~95% of runtime that is ``dRange`` /
``searchDistance`` / ``aRange`` / ``tagAngle`` and friends -- lives in
:mod:`pydigest2._search`, written once as a buffer-based kernel. This
module picks how to run it:

* If **numba** is installed, the kernel is JIT-compiled (``njit``,
  ``nogil``) and fed NumPy buffers. This is roughly an order of
  magnitude faster than interpreting it, and because it releases the
  GIL, :mod:`pydigest2.engine` can then parallelize with threads
  instead of processes.
* Otherwise the exact same function runs as ordinary Python, fed plain
  lists (which index faster than NumPy scalars do in the interpreter).

numba is an optional extra, never required: ``pip install
pydigest2[fast]``. Both paths run the same source, so they cannot
drift; ``tests/test_kernel.py`` asserts they produce identical scores.

Why the search can't just be vectorized with NumPy instead
-----------------------------------------------------------
It is a *data-dependent recursive* search: at each node, whether (and
in what order) the two halves get explored depends on whether the
midpoint orbit lands in a previously untagged model bin -- and that
midpoint is jittered by an LCG whose draws must be consumed in exactly
this traversal order to reproduce the reference program's bin
selection. There is no fixed set of "steps" to batch across, and
reordering the calls would desync the RNG. The parallelism that *is*
available is across tracklets, which are fully independent of one
another; see :mod:`pydigest2.engine`.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt
from typing import Sequence

import numpy as np

from ._search import search_all as _search_all_py  # noqa: F401  (used by tests/test_kernel.py)
from .constants import (
    ARCSEC_RAD, EPART, EX, HPART, HX, IPART, IX, QPART, QX,
)
from .geometry import ec_rotate, lst, se2000, sub3
from .model import Model
from .obscodes import SiteTable, parse_cod3
from .observations import Observation
from .tracklet import clip_err, two_obs

N_BINS = QX * EX * IX * HX

# Stack depths for the two explicit search stacks. Both recursions halve
# their interval each level and stop once it stops yielding new bins, so
# real depths are small (the C implementation runs its worker threads on
# 8 KB stacks); these are generous and the kernel raises rather than
# silently truncating if one were ever hit.
_ANGLE_STACK = 4096
_DISTANCE_STACK = 512

try:  # optional accelerator
    from numba import njit as _njit
except ImportError:  # pragma: no cover - exercised by whichever env lacks numba
    _search_all = _search_all_py
    KERNEL_BACKEND = "python"
else:
    _search_all = _njit(cache=True, nogil=True)(_search_all_py)
    KERNEL_BACKEND = "numba"

_NUMBA = KERNEL_BACKEND == "numba"

# Partition tables: NumPy arrays for the compiled kernel, plain tuples for
# the interpreted one (tuple indexing is much cheaper than NumPy scalar
# indexing, and these are read inside the innermost bin-lookup loops).
if _NUMBA:
    _QPART = np.asarray(QPART, dtype=np.float64)
    _EPART = np.asarray(EPART, dtype=np.float64)
    _IPART = np.asarray(IPART, dtype=np.float64)
    _HPART = np.asarray(HPART, dtype=np.float64)
else:
    _QPART, _EPART, _IPART, _HPART = QPART, EPART, IPART, HPART


class _Buffers:
    """Scratch buffers for one :func:`score_tracklet` call.

    Allocated per call rather than shared, so the kernel is re-entrant
    and safe to run from several threads at once (which is the point of
    compiling it ``nogil``).
    """

    __slots__ = ("d_in", "d_out", "g_in", "g_out", "d_keys",
                 "st_a1", "st_a2", "st_age", "dst_lo", "dst_hi", "dst_age",
                 "sum_all_in", "sum_unk_in", "sum_all_out", "sum_unk_out",
                 "pair_ra", "pair_dec", "pair_rms_ra", "pair_rms_dec",
                 "obs_err", "sun_obs")

    def __init__(self) -> None:
        if _NUMBA:
            zi = lambda n: np.zeros(n, dtype=np.int64)      # noqa: E731
            ei = lambda n: np.empty(n, dtype=np.int64)      # noqa: E731
            ef = lambda n: np.empty(n, dtype=np.float64)    # noqa: E731
            zf = lambda n: np.zeros(n, dtype=np.float64)    # noqa: E731
        else:
            zi = lambda n: [0] * n                          # noqa: E731
            ei = lambda n: [0] * n                          # noqa: E731
            ef = lambda n: [0.0] * n                        # noqa: E731
            zf = lambda n: [0.0] * n                        # noqa: E731

        self.d_in = zi(N_BINS)
        self.d_out = zi(N_BINS)
        self.g_in = zi(N_BINS)
        self.g_out = zi(N_BINS)
        self.d_keys = ei(N_BINS)
        self.st_a1 = ef(_ANGLE_STACK)
        self.st_a2 = ef(_ANGLE_STACK)
        self.st_age = ei(_ANGLE_STACK)
        self.dst_lo = ef(_DISTANCE_STACK)
        self.dst_hi = ef(_DISTANCE_STACK)
        self.dst_age = ei(_DISTANCE_STACK)
        self.sum_all_in = zf(15)
        self.sum_unk_in = zf(15)
        self.sum_all_out = zf(15)
        self.sum_unk_out = zf(15)
        self.pair_ra = ef(2)
        self.pair_dec = ef(2)
        self.pair_rms_ra = ef(2)
        self.pair_rms_dec = ef(2)
        self.obs_err = ef(2)
        self.sun_obs = ef(6)


class _FlatModel:
    """Flat (1-D per bin) views of a :class:`~pydigest2.model.Model`.

    Reshaping is a view, not a copy, so this is free; it is cached on the
    Model instance so repeated tracklets don't redo it.
    """

    __slots__ = ("all_ss", "unk_ss", "all_class", "unk_class")

    def __init__(self, model: Model) -> None:
        self.all_ss = np.ascontiguousarray(model.all_ss.reshape(-1))
        self.unk_ss = np.ascontiguousarray(model.unk_ss.reshape(-1))
        self.all_class = np.ascontiguousarray(model.all_class.reshape(-1, N_BINS))
        self.unk_class = np.ascontiguousarray(model.unk_class.reshape(-1, N_BINS))


_flat_cache: dict = {}


def _flat_model(model: Model) -> _FlatModel:
    flat = _flat_cache.get(id(model))
    if flat is None:
        flat = _FlatModel(model)
        # keyed by identity, with a reference to the model kept alive by the
        # caller; bounded because a process holds one or two models at most
        _flat_cache[id(model)] = flat
    return flat


def gc_rms_prime_ades(olist: Sequence[Observation], obs_err0_rad: float,
                       no_threshold: bool) -> float:
    """d2math.c: gcRmsPrimeAdes -- RMS from the ADES-reported per-obs
    uncertainties, over *all* observations (not just the motion-vector pair)."""
    efc = obs_err0_rad / ARCSEC_RAD
    if not efc:
        efc = 1.0
    s = 0.0
    for obs in olist:
        rms_ra = obs.rms_ra / ARCSEC_RAD
        rms_dec = obs.rms_dec / ARCSEC_RAD
        if not no_threshold:
            min_t, max_t = 0.7 * efc, 5.0 * efc
            rms_ra = min(max(rms_ra, min_t), max_t)
            rms_dec = min(max(rms_dec, min_t), max_t)
        s += rms_ra * rms_ra + rms_dec * rms_dec
    return sqrt(s / len(olist))


def gc_rms_prime_mpc(obs_err0_rad: float) -> float:
    """d2math.c: gcRmsPrimeMPC. The per-obs term is a constant (the pair's
    obsErr[0]) repeated ``len(olist)`` times, so the RMS reduces exactly
    to ``sqrt(2) * obsErr[0]`` independent of tracklet length."""
    er = obs_err0_rad / ARCSEC_RAD
    return sqrt(2.0) * er


@dataclass
class ScoreResult:
    rms: float          # great-circle RMS of the tracklet, arcsec (0 if only 2 obs)
    rms_prime: float     # ADES rmsPrime, arcsec (0 if not ADES / not available)
    raw_scores: dict     # {class_index: score}
    noid_scores: dict    # {class_index: score}


def score_tracklet(olist: Sequence[Observation], *, vmag: float, is_ades: bool,
                    class_indices: Sequence[int], model: Model, site_table: SiteTable,
                    default_obserr_rad: float, no_threshold: bool,
                    repeatable: bool, seed: int = 3) -> ScoreResult:
    """Score one tracklet. Direct port of d2math.c's ``score()`` driver."""
    mv = two_obs(olist, site_table)
    obs_pair = mv.obs_pair

    dt = obs_pair[1].mjd - obs_pair[0].mjd
    invdt = 1.0 / dt
    invdtsq = invdt * invdt

    buf = _Buffers()

    # solve the sun->observer vectors at the two observation times
    soe = coe = 0.0
    for i in (0, 1):
        obs = obs_pair[i]
        site = site_table[parse_cod3(obs.obscode)]
        buf.obs_err[i] = clip_err(mv.obs_err_rms[i], site, default_obserr_rad)
        # tk->soe/coe end up holding pair[1]'s values, as in the C source
        sun_earth, soe, coe = se2000(obs.mjd)
        if obs.spacebased:
            v = obs.earth_observer
        else:
            th = lst(obs.mjd, site.longitude)
            v = (site.rho_cos_phi * cos(th), site.rho_cos_phi * sin(th), site.rho_sin_phi)
        v = ec_rotate(sub3(v, sun_earth), soe, coe)
        buf.sun_obs[i * 3] = v[0]
        buf.sun_obs[i * 3 + 1] = v[1]
        buf.sun_obs[i * 3 + 2] = v[2]
        buf.pair_ra[i] = obs.ra
        buf.pair_dec[i] = obs.dec
        buf.pair_rms_ra[i] = obs.rms_ra
        buf.pair_rms_dec[i] = obs.rms_dec

    no_obs_err = buf.obs_err[0] == 0.0 and buf.obs_err[1] == 0.0

    class_bits = 0
    for c in class_indices:
        class_bits |= 1 << c

    flat = _flat_model(model)

    _search_all(
        buf.pair_ra, buf.pair_dec, buf.pair_rms_ra, buf.pair_rms_dec, buf.obs_err,
        soe, coe, is_ades, no_threshold, no_obs_err,
        buf.sun_obs, invdt, invdtsq, vmag, class_bits,
        flat.all_ss, flat.unk_ss, flat.all_class, flat.unk_class,
        _QPART, _EPART, _IPART, _HPART,
        (seed if repeatable else _random_odd_seed()),
        buf.d_in, buf.d_out, buf.g_in, buf.g_out, buf.d_keys,
        buf.st_a1, buf.st_a2, buf.st_age, buf.dst_lo, buf.dst_hi, buf.dst_age,
        buf.sum_all_in, buf.sum_unk_in, buf.sum_all_out, buf.sum_unk_out,
    )

    rms_prime = 0.0
    if is_ades:
        rms_prime = gc_rms_prime_ades(olist, buf.obs_err[0], no_threshold)
        if rms_prime == 0.0:
            rms_prime = gc_rms_prime_mpc(buf.obs_err[0])

    raw_scores, noid_scores = {}, {}
    for cls_id in class_indices:
        denom_all = buf.sum_all_in[cls_id] + buf.sum_all_out[cls_id]
        raw = (100.0 * buf.sum_all_in[cls_id] / denom_all if denom_all > 0.0
               else (100.0 if cls_id < 2 else 0.0))
        denom_unk = buf.sum_unk_in[cls_id] + buf.sum_unk_out[cls_id]
        noid = (100.0 * buf.sum_unk_in[cls_id] / denom_unk if denom_unk > 0.0
                else (100.0 if cls_id < 2 else 0.0))
        raw_scores[cls_id] = float(raw)
        noid_scores[cls_id] = float(noid)

    return ScoreResult(rms=mv.rms, rms_prime=rms_prime, raw_scores=raw_scores,
                        noid_scores=noid_scores)


def _random_odd_seed() -> int:
    import os
    return int.from_bytes(os.urandom(7), "little") | 1
