"""Streaming grouping of input records into tracklets, exactly as the
reference CLI does it: *consecutive* same-designation records form one
tracklet, in file order (``digest2.c: readMPC80`` for MPC80,
``d2ades.c: parse_nodes`` for ADES XML -- both walk their input once,
starting a new tracklet only when the designation changes).

This deliberately differs from :mod:`pydigest2.observations`'s
``parse_*_file`` functions, which group by a whole-file dict (more
convenient for library use, and the MPC's own Python wrapper does the
same) -- see that module's docstring. OPERATION.md asks that input be
pre-sorted by designation then time, so for well-formed input the two
groupings agree; this module exists so the CLI matches the reference
program exactly even when that assumption doesn't hold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Tuple, Union

from .observations import Observation, apply_mpc80_second_line, parse_mpc80


@dataclass
class ValidTracklet:
    desig: str
    observations: List[Observation] = field(default_factory=list)


@dataclass
class InvalidRun:
    n_lines: int


TrackletOrInvalid = Union[ValidTracklet, InvalidRun]

# One record per input line/element: ("valid", desig, obs), ("invalid",), or
# ("skip",) -- e.g. a satellite/roving second line successfully applied to
# the previous observation, which doesn't start or extend a run either way.
Record = Tuple


def group_consecutive(records: Iterable[Record]) -> Iterator[TrackletOrInvalid]:
    current: Optional[TrackletOrInvalid] = None
    for record in records:
        kind = record[0]
        if kind == "skip":
            continue
        if kind == "valid":
            _, desig, obs = record
            if current is None or isinstance(current, InvalidRun) or current.desig != desig:
                if current is not None:
                    yield current
                current = ValidTracklet(desig=desig, observations=[obs])
            else:
                current.observations.append(obs)
        else:  # "invalid"
            if isinstance(current, InvalidRun):
                current.n_lines += 1
            else:
                if current is not None:
                    yield current
                current = InvalidRun(n_lines=1)
    if current is not None:
        yield current


def _mpc80_records(lines: Iterable[str]) -> Iterator[Record]:
    last_valid_obs: Optional[Observation] = None
    for raw in lines:
        line = raw.rstrip("\n")
        if len(line) >= 15 and line[14] in ("s", "v"):
            if last_valid_obs is not None and apply_mpc80_second_line(line, line[14], last_valid_obs):
                yield ("skip",)
            else:
                yield ("invalid",)
            continue
        parsed = parse_mpc80(line)
        if parsed is None:
            last_valid_obs = None
            yield ("invalid",)
        else:
            desig, obs = parsed
            last_valid_obs = obs
            yield ("valid", desig, obs)


def iter_mpc80_tracklets(lines: Iterable[str]) -> Iterator[TrackletOrInvalid]:
    return group_consecutive(_mpc80_records(lines))


def _ades_xml_records(path) -> Iterator[Record]:
    import xml.etree.ElementTree as ET

    from .observations import _parse_ades_optical

    root = ET.parse(path).getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""

    def opticals():
        found_nested = False
        for obs_block in root.iter(ns + "obsBlock"):
            for obs_data in obs_block.iter(ns + "obsData"):
                for optical in obs_data.iter(ns + "optical"):
                    found_nested = True
                    yield optical
        if not found_nested:
            yield from root.iter(ns + "optical")

    for optical_el in opticals():
        desig = None
        for tag in ("trkSub", "provID", "permID"):
            el = optical_el.find(ns + tag)
            if el is not None and el.text and el.text.strip():
                desig = el.text.strip()
                break
        obs = _parse_ades_optical(optical_el, ns)
        if obs is None or desig is None:
            yield ("invalid",)
        else:
            yield ("valid", desig, obs)


def iter_ades_xml_tracklets(path) -> Iterator[TrackletOrInvalid]:
    return group_consecutive(_ades_xml_records(path))
