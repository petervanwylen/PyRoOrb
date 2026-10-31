"""High-level scoring engine: loads the model/obscodes/config once and
scores tracklets against them, optionally fanning out across processes.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import Config, load_config_file
from .constants import DEFAULT_V_MAG
from .model import Model, load_model
from .obscodes import SiteTable
from .observations import Observation
from .paths import find_model_path, find_obscodes_path
from .ranging import KERNEL_BACKEND, ScoreResult, score_tracklet


def compute_vmag(olist: Sequence[Observation]) -> float:
    """Composite tracklet magnitude: mean of reported mags, default 21.0
    (digest2.c: eval(), same rule as d2lib.c's lib_alloc_tracklet)."""
    vals = [o.vmag for o in olist if o.vmag > 0.0]
    return sum(vals) / len(vals) if vals else DEFAULT_V_MAG


@dataclass
class TrackletProblem:
    """Why a tracklet can't be scored (digest2.c: eval()'s guard checks)."""
    reason: str


def validate_tracklet(olist: Sequence[Observation]) -> Optional[TrackletProblem]:
    """Same three guard checks as digest2.c's eval(); reason strings match
    the reference program's messages verbatim (desig prefix is the
    caller's job -- see pydigest2.cli)."""
    if len(olist) < 2:
        return TrackletProblem("single observation. skipped.")
    if olist[-1].mjd < olist[0].mjd:
        return TrackletProblem("observations out of order.")
    if olist[0].ra == olist[-1].ra and olist[0].dec == olist[-1].dec:
        return TrackletProblem("tracklet shows no motion.")
    return None


class Digest2Engine:
    """Loaded model + obscodes + config, ready to score tracklets."""

    def __init__(self, model: Model, site_table: SiteTable, config: Optional[Config] = None):
        self.model = model
        self.site_table = site_table
        self.config = config or Config()

    @classmethod
    def load(cls, model_path: Optional[str] = None, obscodes_path: Optional[str] = None,
              config_path: Optional[str] = None) -> "Digest2Engine":
        model_path = model_path or find_model_path()
        obscodes_path = obscodes_path or find_obscodes_path()
        site_table = SiteTable.from_file(obscodes_path)
        config = load_config_file(config_path, site_table) if config_path else Config()
        model = load_model(model_path)
        return cls(model, site_table, config)

    def score(self, olist: Sequence[Observation], *, is_ades: bool = False,
              class_indices: Optional[Sequence[int]] = None,
              repeatable: Optional[bool] = None) -> ScoreResult:
        problem = validate_tracklet(olist)
        if problem is not None:
            raise ValueError(problem.reason)
        vmag = compute_vmag(olist)
        return score_tracklet(
            olist, vmag=vmag, is_ades=is_ades,
            class_indices=class_indices if class_indices is not None else self.config.class_compute,
            model=self.model, site_table=self.site_table,
            default_obserr_rad=self.config.default_obserr_rad,
            no_threshold=self.config.no_threshold,
            repeatable=self.config.repeatable if repeatable is None else repeatable,
        )


# --- parallel fan-out for scoring many tracklets ---------------------------
#
# Tracklets are fully independent of one another (each has its own RNG
# stream and its own per-class tag sets), so the batch is embarrassingly
# parallel. That is this port's throughput lever, since an individual
# tracklet's adaptive search is inherently sequential -- see
# :mod:`pydigest2.ranging`.
#
# Which pool to use depends on the kernel backend:
#
# * **numba present** -> threads. The compiled kernel is built ``nogil``,
#   so it genuinely runs in parallel, and threads share the loaded model
#   outright: no pickling, no per-worker reload, no memory multiplication.
# * **pure Python** -> processes, because the interpreted kernel holds
#   the GIL and threads would serialize.
#
# Correctness note for the process path: a worker MUST score against the
# exact same Config and SiteTable as the calling engine -- including any
# obserr overrides or repeatable/class-set choices made programmatically
# after load(), which a config *file* path can't reconstruct. So the
# initializer is handed the already-built Config/SiteTable objects (small,
# cheap to pickle) directly, rather than a path to re-derive them from.

_worker_engine: Optional[Digest2Engine] = None


def _init_worker(model_path: Optional[str], site_table: SiteTable, config: Config) -> None:
    global _worker_engine
    if _worker_engine is not None:
        return  # already inherited a live engine via fork/copy-on-write
    _worker_engine = Digest2Engine(load_model(model_path), site_table, config)


def _score_one(args) -> Tuple[str, object]:
    desig, olist, is_ades = args
    assert _worker_engine is not None
    return _score_sequential(desig, olist, is_ades, _worker_engine)


def score_many(
    tracklets: Sequence[Tuple[str, Sequence[Observation], bool]],
    engine: Digest2Engine,
    model_path: Optional[str] = None,
    obscodes_path: Optional[str] = None,
    n_jobs: Optional[int] = None,
) -> List[Tuple[str, object]]:
    """Score many (designation, observations, is_ades) tracklets in parallel.

    Workers always score against ``engine``'s exact ``config`` and
    ``site_table``. ``model_path`` is only consulted by process workers
    that could not inherit the loaded model (i.e. spawn-based platforms);
    it is unused on the thread path and on fork.
    """
    n_jobs = n_jobs or os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, len(tracklets) or 1))

    if n_jobs <= 1 or len(tracklets) < 8:
        return [_score_sequential(desig, olist, is_ades, engine)
                for desig, olist, is_ades in tracklets]

    chunksize = max(1, len(tracklets) // (n_jobs * 4))

    if KERNEL_BACKEND == "numba":
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            return list(pool.map(
                lambda t: _score_sequential(t[0], t[1], t[2], engine),
                tracklets, chunksize=chunksize))

    global _worker_engine
    _worker_engine = engine  # visible to forked children immediately

    with ProcessPoolExecutor(
        max_workers=n_jobs, initializer=_init_worker,
        initargs=(model_path, engine.site_table, engine.config),
    ) as pool:
        return list(pool.map(_score_one, tracklets, chunksize=chunksize))


def _score_sequential(desig, olist, is_ades, engine: Digest2Engine):
    problem = validate_tracklet(olist)
    if problem is not None:
        return desig, problem
    try:
        return desig, engine.score(olist, is_ades=is_ades)
    except ArithmeticError as e:
        return desig, TrackletProblem(str(e))
