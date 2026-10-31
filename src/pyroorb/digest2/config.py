"""``digest2.config`` parsing and CLI-facing configuration state.

Ported from ``d2cli.c``: ``readConfig`` (config file keywords) and
``mustParseLimit`` (the ``--limit`` option's little
``class/raw|noid=NN`` grammar). Also holds the same defaults
``digest2.c: setup()`` establishes before applying a config file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .classes import N_CLASSES, class_index
from .constants import DEFAULT_OBSERR_RAD
from .obscodes import SiteTable, parse_cod3

_OBSERR_RE = re.compile(r"^[ \t]*([^ \t=]*)[ \t]*=[ \t]*(.+)$")
_LIMIT_RE = re.compile(r"^(.+)/(raw|noid)=([0-9]+)$")


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    headings: bool = True
    rms: bool = True
    rms_prime: bool = False
    no_threshold: bool = False
    raw: bool = False
    noid: bool = True
    repeatable: bool = False
    class_possible: bool = True
    # class columns explicitly listed (in config-file order); other classes
    # are appended as "(Abbr NN)" possibilities when class_possible is True.
    class_columns: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    # the full set of classes actually computed (== all 15 unless the
    # config restricted to specific classes *and* suppressed "poss").
    class_compute: List[int] = field(default_factory=lambda: list(range(N_CLASSES)))
    default_obserr_rad: float = DEFAULT_OBSERR_RAD

    limit_class: Optional[int] = None
    limit_raw: bool = False
    limit_value: Optional[int] = None

    def validate_limit(self) -> None:
        if self.limit_class is None:
            return
        if self.limit_class not in self.class_compute:
            raise ConfigError("--limit orbit class not configured")
        if self.limit_raw and not self.raw:
            raise ConfigError("--limit score not configured")
        if not self.limit_raw and not self.noid:
            raise ConfigError("--limit score not configured")


def parse_limit_spec(spec: str) -> tuple:
    """Parse a ``--limit`` argument, e.g. ``NEO/raw=50``. Returns
    (class_index, limit_raw, limit_value)."""
    m = _LIMIT_RE.match(spec)
    if not m:
        raise ConfigError(f"--limit invalid syntax: {spec}")
    class_name, score_kind, value_s = m.group(1), m.group(2), m.group(3)
    idx = class_index(class_name)
    if idx is None:
        raise ConfigError(f"--limit invalid orbit class: {spec} (see --help)")
    value = int(value_s)
    if not (1 <= value <= 100):
        raise ConfigError(f"--limit value must be in range [1,100]: {spec}")
    return idx, score_kind == "raw", value


def apply_config_lines(cfg: Config, lines, site_table: SiteTable) -> None:
    """Mutate cfg (and site_table's per-site obserr) according to config
    file keywords, in file order (d2cli.c: readConfig)."""
    raw_spec = False
    class_spec = False

    for raw_line in lines:
        line = raw_line.rstrip("\n").rstrip("\r")
        if not line or line.startswith("#"):
            continue

        if line == "headings":
            cfg.headings = True
        elif line == "noheadings":
            cfg.headings = False
        elif line == "rms":
            cfg.rms = True
        elif line == "rmsPrime":
            cfg.rms_prime = True
        elif line == "noThreshold":
            cfg.no_threshold = True
        elif line == "norms":
            cfg.rms = False
        elif line == "raw":
            if not raw_spec:
                raw_spec = True
                cfg.noid = False
            cfg.raw = True
        elif line == "noid":
            if not raw_spec:
                raw_spec = True
                cfg.raw = False
            cfg.noid = True
        elif line == "poss":
            if not class_spec:
                class_spec = True
                cfg.class_columns = []
            cfg.class_possible = True
        elif line == "repeatable":
            cfg.repeatable = True
        elif line == "random":
            cfg.repeatable = False
        elif line.startswith("obserr"):
            _apply_obserr(line[6:], line, site_table, cfg)
        else:
            idx = class_index(line)
            if idx is None:
                raise ConfigError(f"Unrecognized line in config file: {line}")
            if not class_spec:
                class_spec = True
                cfg.class_columns = []
                cfg.class_possible = False
            cfg.class_columns.append(idx)

    if class_spec and not cfg.class_possible:
        cfg.class_compute = list(cfg.class_columns)
    else:
        cfg.class_compute = list(range(N_CLASSES))


def _apply_obserr(rest: str, whole_line: str, site_table: SiteTable, cfg: Config) -> None:
    m = _OBSERR_RE.match(rest)
    if not m:
        raise ConfigError(f"Invalid format for obserr.\nConfig file line: {whole_line}")
    obscode, value_s = m.group(1), m.group(2)
    try:
        oe = float(value_s)
    except ValueError:
        raise ConfigError(f"Invalid obserr value.\nConfig file line: {whole_line}")
    if oe > 10:
        raise ConfigError(
            f"Observational error > 10 arc seconds not allowed.\nConfig file line: {whole_line}")
    from .constants import ARCSEC_RAD
    if obscode == "":
        cfg.default_obserr_rad = oe * ARCSEC_RAD
        return
    idx = parse_cod3(obscode)
    if idx < 0:
        raise ConfigError(f"Obscode not recognized.\nConfig file line: {whole_line}")
    site_table.set_obserr(idx, oe * ARCSEC_RAD)


def load_config_file(path, site_table: SiteTable) -> Config:
    cfg = Config()
    text = Path(path).read_text()
    apply_config_lines(cfg, text.splitlines(), site_table)
    return cfg
