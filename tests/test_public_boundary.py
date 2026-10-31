"""Public code never imports private code.

The published package must import and run with every private path absent. The private
paths are the ``PRIVATE`` list of ``test_public_code_hygiene`` (the same list the push
guard uses), so releasing a module is one edit to one list. Every import statement in a
public Python file is checked, including the ones inside functions: a lazy import of a
private module is still a failure the moment that function runs in a public install.

``PENDING`` lists couplings that exist and are being removed; it is empty and may only
shrink: a new coupling fails, and so does an entry that no longer exists, so the list cannot
outlive the code it excuses. Private web features reach the server as add-ons
(web/addons.py), never through an import in public code.
"""
import ast
import pathlib
import subprocess

import pytest

from test_public_code_hygiene import PRIVATE, REPO

PENDING = set()


def _module_of(path):
    """The import name of a PRIVATE path under src/, or None (tests, lab, data, docs)."""
    p = path.rstrip("/")
    if not p.startswith("src/"):
        return None
    name = p[len("src/"):]
    if name.endswith(".py"):
        name = name[:-3]
    elif "." in pathlib.PurePath(name).name:
        return None                       # a data or doc file
    return name.replace("/", ".")


PRIVATE_MODULES = sorted({m for m in map(_module_of, PRIVATE) if m})


def _private_root(name):
    for m in PRIVATE_MODULES:
        if name == m or name.startswith(m + "."):
            return m
    return None


def _public_python_files():
    out = subprocess.run(["git", "ls-files", "src", "tests", "conftest.py"], cwd=REPO,
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    return [rel for rel in out.stdout.split()
            if rel.endswith(".py") and not rel.startswith(PRIVATE)]


def _package_of(rel):
    """The package a file under src/ belongs to (for resolving relative imports)."""
    if not rel.startswith("src/"):
        return ""
    return ".".join(pathlib.PurePath(rel[len("src/"):]).parent.parts)


def _imported_names(rel, tree):
    pkg = _package_of(rel)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node.lineno, a.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = pkg.split(".") if pkg else []
                parts = parts[:len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join(parts + ([base] if base else []))
            yield node.lineno, base
            for a in node.names:
                yield node.lineno, f"{base}.{a.name}" if base else a.name


def _couplings():
    found = {}
    for rel in _public_python_files():
        path = REPO / rel
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(), filename=rel)
        for line, name in _imported_names(rel, tree):
            root = _private_root(name)
            if root:
                found.setdefault((rel, root), line)
    return found


def test_private_module_list_is_derived():
    """A path-list change must not silently empty the module list."""
    for m in ("pyroorb.neocpmon", "pyroorb.idqueue", "pyroorb.itf.batch", "pyroorb.itf.heliolinc",
              "pyroorb.itfprio", "pyroorb.photcal", "pyroorb.mislink"):
        assert m in PRIVATE_MODULES, PRIVATE_MODULES
    assert "pyroorb.itf.search" not in PRIVATE_MODULES
    assert "pyroorb.itf.training" not in PRIVATE_MODULES


def test_public_code_imports_nothing_private():
    found = _couplings()
    new = sorted(f"{rel}:{line}: imports {mod}" for (rel, mod), line in found.items()
                 if (rel, mod) not in PENDING)
    assert not new, ("public code must not import a private module (give the public code "
                     "a generic hook instead):\n" + "\n".join(new))


def test_pending_couplings_still_exist():
    gone = sorted(f"{rel} -> {mod}" for rel, mod in PENDING if (rel, mod) not in _couplings())
    assert not gone, "remove these from PENDING, they no longer exist:\n" + "\n".join(gone)


def test_the_scan_catches_a_lazy_private_import():
    tree = ast.parse("def f():\n    from pyroorb.neocpmon import config\n"
                     "    import pyroorb.itf.batch\n")
    names = [n for _, n in _imported_names("src/pyroorb/web/x.py", tree)]
    assert {_private_root(n) for n in names} >= {"pyroorb.neocpmon", "pyroorb.itf.batch"}
    rel = ast.parse("from .batch import main\nfrom ..neocpmon import config\n")
    got = [n for _, n in _imported_names("src/pyroorb/itf/x.py", rel)]
    assert "pyroorb.itf.batch" in got and "pyroorb.neocpmon" in got
