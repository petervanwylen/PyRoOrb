"""PyRoOrb -- orbit determination from asteroid astrometry.

    >>> import pyroorb
    >>> obs = pyroorb.fetch_observations("2023 QJ3")          # or read_observations(path)
    >>> fit = pyroorb.fit_orbit(obs, designation="2023 QJ3")
    >>> fit.is_determined, fit.rms_residual()
    >>> fit.elements(), fit.element_sigmas(), fit.covariance_matrix()
    >>> eph = pyroorb.ephemeris(fit, "2026-11-01", "2026-12-01", step=1.0, site="568")
    >>> print(pyroorb.export(fit, "mpcorb"))

The functions here are the supported surface; everything else is importable but internal
and may change in any release. The station table is implicit: every call uses the MPC
table that ships with PyRoOrb (refreshed when an observation names a station it lacks);
``obscodes=`` overrides it.
"""
import os
import warnings

__version__ = "0.9.0"

# The engine loads on first use, not at `import pyroorb`: importing the package (or any
# light submodule such as pyroorb.datadirs) must not read every module's environment-
# dependent settings before the caller has set them.


def __getattr__(name):
    if name == "FittedOrbit":
        from pyroorb.fit_object import FittedOrbit
        return FittedOrbit
    if name == "load_obscodes":
        from pyroorb.obs_sources import load_obscodes
        return load_obscodes
    raise AttributeError(f"module 'pyroorb' has no attribute {name!r}")


class UndeterminedOrbitWarning(UserWarning):
    """The observations leave the orbit undetermined; its elements are nominal only."""


class Observations(list):
    """A list of observation records (pass it to :func:`fit_orbit`; the records' keys are not
    part of the supported API). ``meta`` carries what the source said about them -- the
    designations, the format, any radar astrometry -- which :func:`fit_orbit` uses."""

    def __init__(self, records=(), meta=None):
        super().__init__(records)
        self.meta = dict(meta or {})


def obscodes():
    """The MPC observatory-code table as the engine uses it: ``{code: (lon, cos, sin)}``."""
    from pyroorb.obs_sources import load_obscodes
    return load_obscodes()


def _table(oc):
    return oc if oc is not None else obscodes()


def read_observations(source, *, obscodes=None, format=None):
    """Observations from an astrometry file: a path, or the file's text. OBS80 and ADES
    (PSV or XML) are recognised automatically; ``format`` ("obs80" / "ades") forces one."""
    text = source
    if isinstance(source, os.PathLike) or (
            isinstance(source, str) and "\n" not in source and os.path.isfile(source)):
        with open(source, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    from pyroorb import obs_sources as _obs_sources
    radar = []
    records, label = _obs_sources.read_obs_file(text, _table(obscodes), in_format=format,
                                                radar_out=radar)
    desigs = sorted({o.get("designation") for o in records if o.get("designation")})
    return Observations(records, {"format": label, "designations": desigs, "radar": radar})


def fetch_observations(designations, *, obscodes=None):
    """Observations of one designation or several (a list is fetched and combined as ONE
    object -- a proposed identification), from the MPC, with the SBN replica as fallback."""
    from pyroorb import obs_sources as _obs_sources
    desigs = [designations] if isinstance(designations, str) else list(designations)
    records, meta = _obs_sources.fetch_observations(desigs, _table(obscodes))
    failure = _obs_sources.fetch_failure_message(meta)
    if failure:
        raise RuntimeError(failure)
    return Observations(records, meta)


def fit_orbit(observations, designation=None, *, obscodes=None, force=False):
    """Fit an orbit to ``observations`` (from :func:`read_observations` or
    :func:`fetch_observations`, or a list of records). Returns a :class:`FittedOrbit`.

    When the observations do not determine the orbit (``fit.is_determined`` is False) the
    nominal solution is still returned and an :class:`UndeterminedOrbitWarning` is issued:
    the caller decides what to do with it (``fit.determinacy()`` says what is known).
    ``force`` is an experimental brute-force linker; see the engine's documentation."""
    from pyroorb import fit_object as _fit_object
    meta = getattr(observations, "meta", None) or None
    if designation is None:
        names = (meta or {}).get("designations") or sorted(
            {o.get("designation") for o in observations if o.get("designation")})
        designation = names[0] if names else "object"
    fit = _fit_object.fit_orbit(list(observations), _table(obscodes), desig=designation,
                                meta=meta, force=force)
    if not fit.is_determined:
        warnings.warn(f"{designation}: the observations leave the orbit undetermined; its "
                      f"elements are a nominal solution only (fit.determinacy())",
                      UndeterminedOrbitWarning, stacklevel=2)
    return fit


def _to_jd(t):
    """A time as a UTC Julian date: a number (already JD), an ISO string, or astropy Time."""
    if isinstance(t, (int, float)):
        return float(t)
    from astropy.time import Time
    return float((t if isinstance(t, Time) else Time(t, scale="utc")).utc.jd)


def ephemeris(fit, start, stop, step=1.0, site="500", *, obscodes=None, full=True):
    """The ephemeris of a fitted orbit from ``start`` to ``stop`` every ``step`` days, seen
    from MPC observatory ``site`` (default 500, the geocentre). Times may be ISO strings,
    astropy ``Time`` or UTC Julian dates. Returns one dict per epoch."""
    import pyroorb.ephemerides as _eph
    return _eph.compute_ephemeris(fit, _table(obscodes), _to_jd(start), _to_jd(stop),
                                  float(step), site=str(site), full=full)


_EXPORTS = {"mpcorb": "to_mpcorb", "sof": "to_sof", "eightline": "to_eightline",
            "covar": "to_covar_json"}


def export(fit, format="mpcorb"):
    """A fitted orbit in an exchange format, as text: "mpcorb" (one MPCORB.DAT line),
    "sof" (the MPC's Standard Orbit Format), "eightline" (find_orb's eight-line elements)
    or "covar" (find_orb's covar.json)."""
    from pyroorb import egress
    name = _EXPORTS.get(str(format).lower())
    if name is None:
        raise ValueError(f"unknown export format {format!r}; choose from {sorted(_EXPORTS)}")
    return getattr(egress, name)(fit)


# Every console entry point loads the `.env` settings file BEFORE importing the module it runs:
# several settings (the ephemeris and debias folders) are read once at import time.

def _cli():
    """Console entry point (`pyroorb ...`) — the fit_object CLI; reads sys.argv."""
    from pyroorb.envfile import load_env_file
    load_env_file()
    from pyroorb import cli
    cli.main()


def _web():
    """Console entry point (`pyroorb-web ...`) — launch the PyRoOrb web UI.
    Honours $PORT / $HOST (set by Railway and friends)."""
    from pyroorb.envfile import load_env_file
    load_env_file()
    from pyroorb.web import server
    server.main()


def _itf():
    """Console entry point (`pyroorb-itf search|recovery-suite|maintain ...`) — the ITF CLI
    (also `python -m pyroorb.itf`)."""
    from pyroorb.envfile import load_env_file
    load_env_file()
    from pyroorb.itf.__main__ import main
    return main()


def _fetch_data():
    """Console entry point (`pyroorb-fetch-data [--asteroids]`) — download the reference
    tables a fit needs (DE440, the debias tables; optionally the asteroid ephemeris)."""
    from pyroorb.envfile import load_env_file
    load_env_file()
    from pyroorb.fetch_data import main
    return main()


__all__ = ["read_observations", "fetch_observations", "fit_orbit", "ephemeris", "export",
           "FittedOrbit", "Observations", "UndeterminedOrbitWarning", "obscodes",
           "load_obscodes", "__version__"]
