"""pydigest2 -- a pure-Python port of digest2, the MPC's short-arc
asteroid orbit classification tool.

See README.md for background and usage. The public, stable surface is:

    from pydigest2 import Digest2Engine
    engine = Digest2Engine.load()
    result = engine.score(observations)
"""
from .engine import Digest2Engine, compute_vmag, score_many, validate_tracklet
from .observations import Observation
from .ranging import ScoreResult

__all__ = [
    "Digest2Engine", "Observation", "ScoreResult",
    "compute_vmag", "score_many", "validate_tracklet",
]

__version__ = "0.1.0"
