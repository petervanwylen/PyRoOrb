"""PyRoOrb is one package, imported one way.

Every engine module lives under ``pyroorb``. If anything put a package directory on
``sys.path``, the same file could load a second time under its bare name
(``fit_object`` beside ``pyroorb.fit_object``): two copies of every module-level table,
cache and lock, which diverge silently. These checks keep that from coming back."""
import pathlib
import subprocess
import sys

import pyroorb

PKG = pathlib.Path(pyroorb.__file__).resolve().parent
_BARE = sorted({p.stem for p in PKG.glob("*.py") if p.stem != "__init__"}
               | {p.stem for p in (PKG / "web").glob("*.py") if p.stem != "__init__"}
               | {"itf", "web", "pydigest2", "digest2"})


def test_importing_the_package_loads_no_engine_module_under_a_bare_name():
    code = ("import sys, pyroorb, pyroorb.fit_object, pyroorb.web.server, pyroorb.itf.search, "
            "pyroorb.digest2; print('\\n'.join(sorted(sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, env={"PYTHONPATH": str(PKG.parent),
                                          "PATH": "/usr/bin:/bin"}).stdout.split()
    leaked = sorted(set(out) & set(_BARE))
    assert not leaked, f"engine modules imported under bare names: {leaked}"


def test_this_test_process_has_no_bare_engine_modules_either():
    leaked = sorted(set(sys.modules) & set(_BARE))
    assert not leaked, f"engine modules imported under bare names: {leaked}"


def test_no_package_directory_is_on_sys_path():
    bad = [p for p in sys.path if p and pathlib.Path(p).resolve() in
           {PKG, PKG / "web", PKG / "itf"}]
    assert not bad, bad


def test_import_pyroorb_is_light():
    """`import pyroorb` must not load the engine: a caller sets its environment first."""
    code = "import sys, pyroorb; print(int('pyroorb.fit_object' in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, env={"PYTHONPATH": str(PKG.parent),
                                          "PATH": "/usr/bin:/bin"}).stdout.strip()
    assert out == "0"


def test_no_api_name_is_also_a_module_name():
    """Importing a submodule binds it on the package, so a module named like an API function
    (pyroorb/ephemeris.py beside pyroorb.ephemeris()) would silently replace the function."""
    clashes = [n for n in pyroorb.__all__ if (PKG / f"{n}.py").exists() or (PKG / n).is_dir()]
    assert not clashes, clashes


def test_the_api_survives_importing_the_engine():
    import pyroorb.ephemerides, pyroorb.egress, pyroorb.obs_sources, pyroorb.fit_object  # noqa
    for name in ("read_observations", "fetch_observations", "fit_orbit", "ephemeris", "export",
                 "obscodes"):
        attr = getattr(pyroorb, name)
        assert callable(attr) and not isinstance(attr, type(pyroorb)), name
