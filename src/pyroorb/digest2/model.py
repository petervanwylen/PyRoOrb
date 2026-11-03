"""Population model (q, e, i, H histogram) loading.

The model file, ``digest2.model.csv``, tabulates two "flavors" of
population count -- ``All`` (the full modeled Solar System population)
and ``Unk`` (the subset not yet reliably discovered/identified) -- for
the whole Solar System (``SS``) and for each of the 15 orbit classes,
each binned over (q, e, i, H). This mirrors ``d2modelio.c: readCSV``
exactly, including its rigid record layout: for each of 32 (model,
class) blocks, one line per (iq, ie, ii) triple, in nested iq/ie/ii
order, with 18 comma-separated H-bin population counts per line.

Because the parse+reshape is the same nested order NumPy uses for a
C-contiguous array, an entire block can be reshaped straight into
(QX, EX, IX, HX) with no per-cell bin math.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .classes import CLASS_ABBR, N_CLASSES
from .constants import EPART, EX, HX, IPART, IX, QPART, QX

_HEADER_PREFIX = "Model,Class,Q,e,i"
_BLOCK_ROWS = QX * EX * IX
_CACHE_MAGIC = "pydigest2-model-cache-v1"


@dataclass
class Model:
    """Loaded population model, as four float64 NumPy arrays."""

    all_ss: np.ndarray     # (QX, EX, IX, HX)
    unk_ss: np.ndarray     # (QX, EX, IX, HX)
    all_class: np.ndarray  # (N_CLASSES, QX, EX, IX, HX)
    unk_class: np.ndarray  # (N_CLASSES, QX, EX, IX, HX)

    def __post_init__(self) -> None:
        expect3 = (QX, EX, IX, HX)
        expect4 = (N_CLASSES, QX, EX, IX, HX)
        if self.all_ss.shape != expect3 or self.unk_ss.shape != expect3:
            raise ValueError("model SS arrays have unexpected shape")
        if self.all_class.shape != expect4 or self.unk_class.shape != expect4:
            raise ValueError("model class arrays have unexpected shape")


class ModelError(ValueError):
    """Raised when a model CSV/cache file is missing or malformed."""


def _read_block(lines, start: int, mod: str, cls: str) -> np.ndarray:
    """Parse QX*EX*IX CSV rows into a (QX, EX, IX, HX) float64 array."""
    out = np.empty(_BLOCK_ROWS * HX, dtype=np.float64)
    row = start
    pos = 0
    for iq in range(QX):
        for ie in range(EX):
            for ii in range(IX):
                fields = lines[row].rstrip("\n").split(",")
                row += 1
                if len(fields) != 5 + HX:
                    raise ModelError(
                        f"malformed model row {row}: expected {5 + HX} fields, "
                        f"got {len(fields)}"
                    )
                if fields[0] != mod or fields[1] != cls:
                    raise ModelError(
                        f"model row {row}: expected {mod},{cls}, "
                        f"got {fields[0]},{fields[1]}"
                    )
                # Sanity-check q/e/i against the partition tables, as the
                # C reader does, to catch a misaligned/corrupt CSV early.
                if abs(float(fields[2]) - QPART[iq]) > 1e-9:
                    raise ModelError(f"model row {row}: Q mismatch")
                if abs(float(fields[3]) - EPART[ie]) > 1e-9:
                    raise ModelError(f"model row {row}: e mismatch")
                if abs(float(fields[4]) - IPART[ii]) > 1e-9:
                    raise ModelError(f"model row {row}: i mismatch")
                for h in fields[5:]:
                    out[pos] = float(h) if h else 0.0
                    pos += 1
    return out.reshape(QX, EX, IX, HX), row


def _parse_csv_text(text: str) -> Model:
    lines = text.split("\n")
    if not lines[0].rstrip("\r").startswith(_HEADER_PREFIX):
        raise ModelError("invalid or missing model CSV header")

    row = 1
    all_ss, row = _read_block(lines, row, "All", "SS")
    unk_ss, row = _read_block(lines, row, "Unk", "SS")

    all_class = np.empty((N_CLASSES, QX, EX, IX, HX), dtype=np.float64)
    unk_class = np.empty((N_CLASSES, QX, EX, IX, HX), dtype=np.float64)
    for c, abbr in enumerate(CLASS_ABBR):
        all_class[c], row = _read_block(lines, row, "All", abbr)
        unk_class[c], row = _read_block(lines, row, "Unk", abbr)

    return Model(all_ss=all_ss, unk_ss=unk_ss, all_class=all_class, unk_class=unk_class)


def _cache_path_for(csv_path: Path) -> Path:
    """Beside the CSV when that directory is writable (a source checkout); otherwise --
    an installed package is read-only -- in the user cache (``PYROORB_CACHE_DIR``, else
    ``$XDG_CACHE_HOME`` / ``~/.cache``), so the parse is still paid only once."""
    beside = csv_path.with_suffix(csv_path.suffix + ".npz")
    if beside.is_file() or os.access(csv_path.parent, os.W_OK):
        return beside
    base = (os.environ.get("PYROORB_CACHE_DIR")
            or os.path.join(os.environ.get("XDG_CACHE_HOME")
                            or os.path.join(os.path.expanduser("~"), ".cache"), "pyroorb"))
    return Path(base) / "digest2" / beside.name


def _csv_fingerprint(csv_path: Path) -> str:
    st = csv_path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def load_model_csv(csv_path: os.PathLike | str) -> Model:
    """Parse a ``digest2.model.csv`` file directly (no caching)."""
    csv_path = Path(csv_path)
    text = csv_path.read_text(encoding="utf-8")
    return _parse_csv_text(text)


def load_model(path: os.PathLike | str, use_cache: bool = True) -> Model:
    """Load a population model from ``path``.

    ``path`` may name the CSV file directly, or a cache (``.npz``) file
    previously written by :func:`save_model_cache`. When it is a CSV and
    ``use_cache`` is true, a sibling ``<name>.csv.npz`` is read/written
    to speed up subsequent loads, mirroring the CSV/binary caching in
    ``d2modelio.c`` (``mustReadModelStatCSV``): the cache is trusted only
    if its recorded (size, mtime) fingerprint still matches the CSV.
    """
    path = Path(path)

    if path.suffix == ".npz":
        return _load_cache(path)

    if not use_cache:
        return load_model_csv(path)

    cache_path = _cache_path_for(path)
    if cache_path.is_file():
        try:
            fingerprint = _csv_fingerprint(path)
        except OSError:
            fingerprint = None
        model = _try_load_cache(cache_path, fingerprint)
        if model is not None:
            return model

    model = load_model_csv(path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_model_cache(model, cache_path, source_fingerprint=_csv_fingerprint(path))
    except OSError:
        pass  # cache is a pure speed optimization; failing to write it is not fatal
    return model


def _try_load_cache(cache_path: Path, fingerprint: Optional[str]) -> Optional[Model]:
    try:
        with np.load(cache_path) as npz:
            if "magic" not in npz.files or str(npz["magic"]) != _CACHE_MAGIC:
                return None
            if fingerprint is not None and str(npz["fingerprint"]) != fingerprint:
                return None
            return Model(
                all_ss=npz["all_ss"], unk_ss=npz["unk_ss"],
                all_class=npz["all_class"], unk_class=npz["unk_class"],
            )
    except (OSError, KeyError, ValueError):
        return None


def _load_cache(cache_path: Path) -> Model:
    model = _try_load_cache(cache_path, fingerprint=None)
    if model is None:
        raise ModelError(f"could not read model cache {cache_path}")
    return model


def save_model_cache(model: Model, cache_path: os.PathLike | str,
                      source_fingerprint: str = "") -> None:
    """Write ``model`` to a fast-loading ``.npz`` cache file.

    This is the Python-package analogue of digest2's ``-m`` option
    ("generate binary model from CSV"); the on-disk layout is specific
    to this package (a compressed NumPy archive), not byte-compatible
    with the original C tool's raw ``fwrite`` binary format, since the
    cache is purely an internal speed optimization.
    """
    cache_path = Path(cache_path)
    # np.savez appends .npz if the target name doesn't already end with it;
    # write to the exact literal path by handling the extension ourselves.
    target = cache_path if cache_path.suffix == ".npz" else Path(str(cache_path) + ".npz")
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(
            f,
            magic=np.array(_CACHE_MAGIC),
            fingerprint=np.array(source_fingerprint),
            all_ss=model.all_ss, unk_ss=model.unk_ss,
            all_class=model.all_class, unk_class=model.unk_class,
        )
    os.replace(tmp, target)
