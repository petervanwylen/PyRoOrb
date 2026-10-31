"""Public-bound code reads as production code.

The engine, itf/, web/ and their tests are published; their comments, docstrings and strings
state each rule and its measured reason, never the development record behind it: no item
numbers of the private to-do list, no who-decided-when attributions, no pointers to the
private research/validation folders or the private docs, no agent/session jargon. The
history of a rule lives in the private notes. This test keeps the published files that way.

PRIVATE lists the paths that stay in the private repository;
they are not scanned. A module leaves the list when it is released.
"""
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

PRIVATE = (
    # private modules, at the path each will have when it is released
    "src/pyroorb/neocpmon/", "src/pyroorb/idqueue/", "src/pyroorb/hyperlinc/",
    "src/pyroorb/visweeks/", "src/pyroorb/mislink/", "src/pyroorb/itfprio/",
    "src/pyroorb/photcal.py",
    "src/pyroorb/itf/heliolinc.py", "src/pyroorb/itf/exitf.py", "src/pyroorb/itf/batch.py",
    "src/pyroorb/itf/collect.py", "src/pyroorb/itf/sampling.py", "src/pyroorb/itf/backup/",
    "src/pyroorb/itf/clustering-idea.md", "src/pyroorb/itf/itf_maint.md",
    "src/pyroorb/itf/ladder-backtracking.md", "src/pyroorb/itf/thoroughness-pricing.md",
    "src/pyroorb/itf/training_data/itf_match_v2", "src/pyroorb/itf/training_data/itf_match_v3",
    "src/pyroorb/itf/training_data/itf_train_v2",
    # their tests
    "tests/private/",
    # never-released material, and the deployment
    "lab/", "web_addons.txt", "Procfile", "railway.json", "requirements.txt",
)

_SUFFIXES = (".py", ".js", ".html", ".css")

_FORBIDDEN = [
    ("to-do list item", r"\bgaps(\.md)?\s*#\s*\d|\bgaps\.md\b"),
    ("decision attribution",
     r"\bowner(?:'s)?\s+(?:rule|ruling|rulings|said|decided|decision|asked|wants|wanted|set|"
     r"named|chose|nominated|adjudicated|spotted|pointed|zapped)\b"
     r"|\(owner\b|\bowner[- ](?:nominated|adjudicated)\b|\bowner,?\s+\d{4}-\d\d"),
    ("private folder", r"\bresearch(?:_scratch)?/|(?<![\w/.-])val(?:_impactors)?/|\bscanners/"),
    ("private doc", r"DESIGN-NOTES|TYPICAL_AGENT|VISION\.md|NAMING\.md|scan-and-fix|"
                    r"STEP-BY-STEP|FITTING\.md|\bspeed\.md|OVERVIEW\.md|ITF-PRIO|IMPACT_RESEARCH|"
                    r"\bchunks\.md|itf-next-steps|itf-speedup|CLAUDE\.md"),
    ("agent jargon", r"read in this session|auto-memory|\bworktree"),
    ("private module name", r"\b(neocpmon|itfprio|hyperlinc|visweeks|photcal|heliolinc|itfitf)\b"),
]
# These name the private modules on purpose: they are the guards that keep them out.
_NAMES_PRIVATE_ON_PURPOSE = ("tests/test_public_boundary.py",)
_PATTERNS = [(what, re.compile(rx, re.IGNORECASE)) for what, rx in _FORBIDDEN]


def _public_files():
    out = subprocess.run(["git", "ls-files", "src", "tests", "conftest.py"], cwd=REPO,
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    me = pathlib.Path(__file__).resolve().relative_to(REPO).as_posix()
    return [p for p in out.stdout.split()
            if p.endswith(_SUFFIXES) and not p.startswith(PRIVATE) and p != me]


def test_public_code_carries_no_development_record():
    hits = []
    for rel in _public_files():
        path = REPO / rel
        if not path.is_file():
            continue
        for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for what, rx in _PATTERNS:
                if what == "private module name" and rel in _NAMES_PRIVATE_ON_PURPOSE:
                    continue
                if rx.search(line):
                    hits.append(f"{rel}:{n}: {what}: {line.strip()[:120]}")
    assert not hits, ("public-bound code must state the rule and its measured reason, not the "
                      "development record (see the module docstring):\n" + "\n".join(hits))


def test_scan_sees_the_public_tree():
    """A path-list typo must not silently empty the scan."""
    files = _public_files()
    assert "src/pyroorb/fit_object.py" in files and "src/pyroorb/itf/search.py" in files
    assert "src/pyroorb/web/engine_api.py" in files
    assert not any(f.startswith("src/pyroorb/neocpmon/") for f in files)
