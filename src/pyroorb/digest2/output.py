"""Score-table formatting, matching digest2.c's ``setup()`` (headings)
and ``fmtScores()`` (per-tracklet lines) column-for-column.
"""
from __future__ import annotations

from typing import List, Optional

from .classes import CLASS_ABBR, N_CLASSES
from .config import Config
from .ranging import ScoreResult


def format_mpc_desig(desig12: str) -> str:
    """The 7-character output field for a raw MPC80 12-char designation
    (digest2.c: eval()'s in-place desig shift, before fmtScores prints
    ``desig+5``).

    A non-blank permanent-number field (columns 1-5) is shown in the
    provisional-designation position instead (2 spaces + the number);
    otherwise columns 6-12 (discovery flag/note1 + provisional
    designation) are shown as-is.
    """
    desig12 = desig12.ljust(12)
    if desig12[0:5].strip():
        return "  " + desig12[0:5]
    return desig12[5:12]


def strtok_first_token(desig12: str) -> str:
    """First whitespace-delimited token, for the skip/error messages
    (digest2.c uses ``strtok(tk->desig, " ")``, which skips leading
    spaces then returns the run of non-space characters after them)."""
    parts = desig12.split()
    return parts[0] if parts else ""


def _fmt52(v: float) -> str:
    s = f"{v:5.2f}"
    return s if len(s) == 5 else "**.**"


def format_header_lines(cfg: Config) -> List[str]:
    lines: List[str] = []
    if not cfg.headings:
        return lines

    if cfg.raw and cfg.noid and cfg.class_columns:
        line1 = "-------"
        if cfg.rms:
            line1 += "  ----"
        for c in cfg.class_columns:
            line1 += f"   {CLASS_ABBR[c]:>3.3}  "
        line1 += " ---------------" if cfg.class_possible else ""
        lines.append(line1)

    line2 = "Desig. "
    if cfg.rms:
        line2 += "   RMS"
    if cfg.rms_prime:
        line2 += "  RMS'"
    for c in cfg.class_columns:
        line2 += " Raw NID" if (cfg.raw and cfg.noid) else f" {CLASS_ABBR[c]:>3.3}"
    if cfg.class_possible:
        line2 += " Other Possibilities" if cfg.class_columns else " Possibilities"
    lines.append(line2)
    return lines


def format_score_line(display_desig: str, result: ScoreResult, cfg: Config) -> Optional[str]:
    """Returns the formatted line, or None if suppressed by --limit."""
    if cfg.limit_class is not None:
        score = (result.raw_scores if cfg.limit_raw else result.noid_scores)[cfg.limit_class]
        if int(score + 0.5) < cfg.limit_value:
            return None

    buf = display_desig
    if cfg.rms:
        buf += f" {_fmt52(result.rms)}"
    if cfg.rms_prime:
        buf += f" {_fmt52(result.rms_prime)}"

    if cfg.class_possible:
        for c in cfg.class_columns:
            if cfg.raw:
                buf += f" {result.raw_scores[c]:3.0f}"
            if cfg.noid:
                buf += f" {result.noid_scores[c]:3.0f}"
        shown = set(cfg.class_columns)
        for c in range(N_CLASSES):
            if c in shown:
                continue
            p_score = result.noid_scores.get(c, 0.0) if cfg.noid else result.raw_scores.get(c, 0.0)
            if p_score > 0.5:
                buf += f" ({CLASS_ABBR[c]} {p_score:.0f})"
            elif p_score > 0:
                buf += f" ({CLASS_ABBR[c]} <1)"
    else:
        for c in cfg.class_compute:
            if cfg.raw:
                buf += f" {result.raw_scores[c]:3.0f}"
            if cfg.noid:
                buf += f" {result.noid_scores[c]:3.0f}"
    return buf
