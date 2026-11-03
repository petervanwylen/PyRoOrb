"""PyRoOrb -- orbit determination from asteroid astrometry.

    >>> import pyroorb
    >>> pyroorb.load_settings()                               # the .env file, if you keep one
    >>> obs = pyroorb.fetch_observations("2023 QJ3")          # or read_observations(path)
    >>> fit = pyroorb.fit_orbit(obs, designation="2023 QJ3")
    >>> fit.is_determined, fit.rms_residual()
    >>> fit.elements(), fit.element_sigmas(), fit.covariance_matrix()
    >>> eph = pyroorb.ephemeris(fit, "2026-11-01", "2026-12-01", step=1.0, site="568")
    >>> print(pyroorb.export(fit, "mpcorb"))

The names in ``__all__`` and the :class:`FittedOrbit` members listed in the API reference
(``docs/api.rst``) are the supported surface; everything else is importable but internal and
may change in any release. The station table is implicit: every call uses the MPC table that
ships with PyRoOrb (refreshed when an observation names a station it lacks); ``obscodes=``
overrides it. Units, frames and time scales are stated once, in ``docs/conventions.rst``.
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
    if name == "OrbitEnsemble":
        from pyroorb.ensemble import OrbitEnsemble
        return OrbitEnsemble
    if name == "load_obscodes":
        from pyroorb.obs_sources import load_obscodes
        return load_obscodes
    raise AttributeError(f"module 'pyroorb' has no attribute {name!r}")


class UndeterminedOrbitWarning(UserWarning):
    """The observations leave the orbit undetermined; its elements are nominal only."""


class Observations(list):
    """A list of observation records, plus what their source said about them.

    Pass it to :func:`fit_orbit`. Each record is a dict; these keys are supported (any other
    key is internal):

    ===================  =======================================================
    ``jd_utc``           time of the observation, Julian date, UTC
    ``ra_deg``           right ascension as reported, degrees (ICRF/J2000)
    ``dec_deg``          declination as reported, degrees (ICRF/J2000)
    ``obscode``          MPC observatory code
    ``designation``      the designation the observation was reported under
    ``mag``, ``band``    reported magnitude and band (None when not reported)
    ``sigma_ra``         the right-ascension uncertainty the fit starts from, arcsec
    ``sigma_dec``        the declination uncertainty the fit starts from, arcsec
    ``trksub``           the submitter's tracklet id, when known
    ``discovery``        True on the observation the MPC credits with the discovery
    ``ref``              the publication reference (MPEC, MPC batch), when known
    ===================  =======================================================

    Attributes
    ----------
    meta : dict
        What the source said: ``"designations"``, ``"format"``, and ``"radar"`` (the radar
        astrometry, which :func:`fit_orbit` fits jointly with the optical observations).
    """

    def __init__(self, records=(), meta=None):
        super().__init__(records)
        self.meta = dict(meta or {})

    def select(self, start=None, stop=None, *, stations=None, exclude_stations=None,
               radar=True):
        """A subset, as a new :class:`Observations` carrying the same ``meta``.

        Parameters
        ----------
        start, stop : str, float or astropy.time.Time, optional
            Keep observations with ``start <= time < stop``: ISO strings (UTC), astropy
            ``Time`` or UTC Julian dates.
        stations : iterable of str, optional
            Keep only these MPC observatory codes.
        exclude_stations : iterable of str, optional
            Drop these MPC observatory codes.
        radar : bool, default True
            Keep the radar astrometry (``meta["radar"]``), cut to the same time window.
            ``False`` drops it, giving an optical-only set.

        Returns
        -------
        Observations
        """
        lo = -float("inf") if start is None else _to_jd(start)
        hi = float("inf") if stop is None else _to_jd(stop)
        keep = None if stations is None else {str(c) for c in stations}
        drop = set() if exclude_stations is None else {str(c) for c in exclude_stations}
        rows = [o for o in self
                if lo <= o["jd_utc"] < hi
                and (keep is None or str(o.get("obscode")) in keep)
                and str(o.get("obscode")) not in drop]
        meta = dict(self.meta)
        if not radar:
            meta["radar"] = []
        elif meta.get("radar"):
            meta["radar"] = [r for r in meta["radar"]
                             if r.get("jd_utc") is None or lo <= r["jd_utc"] < hi]
        return Observations(rows, meta)


def load_settings(directory=None):
    """Load PyRoOrb's settings file (``.env``) into the environment.

    The first ``.env`` found searching upward from ``directory`` (default: the current
    directory) is read; a variable already set in the environment wins. Call it before the
    first fit: the ephemeris and debias folders (``PYROORB_EPHEM_DIR``,
    ``PYROORB_DEBIAS_DIR``) are read when the engine first loads. ``import pyroorb`` itself
    reads no settings. Loads once per process.

    Parameters
    ----------
    directory : str or os.PathLike, optional
        Where the upward search starts.

    Returns
    -------
    pathlib.Path or None
        The file loaded, or None when there is none.
    """
    from pyroorb.envfile import load_env_file
    return load_env_file(cwd=directory)


def obscodes():
    """The MPC observatory-code table as the engine uses it.

    Returns
    -------
    dict
        ``{code: (longitude_deg, rho_cos_phi, rho_sin_phi)}``, the MPC's parallax constants
        (geocentric distance in Earth radii).
    """
    from pyroorb.obs_sources import load_obscodes
    return load_obscodes()


def _table(oc):
    return oc if oc is not None else obscodes()


def read_observations(source, *, obscodes=None, format=None):
    """Observations from an astrometry file.

    Parameters
    ----------
    source : str or os.PathLike
        A path, or the file's text. MPC 80-column (OBS80, including its radar lines) and ADES
        (PSV or XML) are recognised automatically.
    obscodes : dict, optional
        An observatory table to use instead of the bundled MPC table.
    format : {"obs80", "ades"}, optional
        Force a format instead of recognising it.

    Returns
    -------
    Observations
        The optical observations; any radar astrometry is in ``meta["radar"]``.
    """
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
    """Observations of an object, fetched from the Minor Planet Center.

    Parameters
    ----------
    designations : str or list of str
        One designation, or several to be fetched and combined as ONE object (a proposed
        identification). Numbers, provisional designations, comet names and NEOCP
        temporary designations are accepted.
    obscodes : dict, optional
        An observatory table to use instead of the bundled MPC table.

    Returns
    -------
    Observations
        Optical observations, de-duplicated (one copy per detection), with the JPL radar
        astrometry in ``meta["radar"]``. Fetches are cached for a few hours (minutes for a
        NEOCP object).

    Raises
    ------
    RuntimeError
        When nothing could be fetched; the message says why.
    """
    from pyroorb import obs_sources as _obs_sources
    desigs = [designations] if isinstance(designations, str) else list(designations)
    records, meta = _obs_sources.fetch_observations(desigs, _table(obscodes))
    failure = _obs_sources.fetch_failure_message(meta)
    if failure:
        raise RuntimeError(failure)
    return Observations(records, meta)


# The public spelling of fit_object.fit_orbit's tri-state `auto_nongrav` (None = the engine's
# automatic discovery, which also honours PYROORB_AUTO_NONGRAV / PYROORB_AUTO_COMET_NONGRAV;
# False = gravity only).
_NONGRAV = {"auto": None, "none": False}
# nongrav.MODELS minus "none", spelled out so `import pyroorb` stays light.
_FORCED_NONGRAV = ("comet1", "comet2", "comet3", "yarkovsky", "srp", "srp2", "srp3")


def fit_orbit(observations, designation=None, *, obscodes=None, nongrav="auto",
              center=None, detect_satellite=True, mpc_seed=False):
    """Fit an orbit to observations.

    Parameters
    ----------
    observations : Observations or list of dict
        From :func:`read_observations` or :func:`fetch_observations` (whose ``meta``, e.g.
        the radar astrometry, is used), or a plain list of records.
    designation : str, optional
        The object's name in the results. Default: the first designation in the data.
    obscodes : dict, optional
        An observatory table to use instead of the bundled MPC table.
    nongrav : str, default "auto"
        ``"auto"``: the fit decides whether a non-gravitational model is warranted -- comet
        outgassing (A1/A2/A3), or Yarkovsky (A2) on an asteroid with q < 1.4 au -- by an
        information criterion, and keeps it only when it earns its parameters. ``"none"``:
        gravity only, the comparison fit for asking what the force explains. A model name
        forces that model whether or not it earns its keep: ``"comet1"`` / ``"comet2"`` /
        ``"comet3"`` (Marsden A1; A1, A2; A1, A2, A3), ``"yarkovsky"`` (A2, 1/r^2),
        ``"srp"`` / ``"srp2"`` / ``"srp3"`` (radiation pressure, 1/r^2). A forced model is
        fitted on the gravity fit's observation set and then re-joined with any radar.
    center : str, optional
        Fit a planetary satellite around this planet (e.g. ``"Jupiter"``) rather than the
        Sun. Implies ``nongrav="none"`` unless told otherwise.
    detect_satellite : bool, default True
        When ``center`` is not given, test whether the object is a satellite of a giant planet
        and refit it planetocentrically if so (``fit.center`` then names the planet).
    mpc_seed : bool, default False
        Start a long-arc object (more than 500 observations or five oppositions) from the
        MPC's published orbit, fetched over the network, instead of an initial orbit
        determination. The ``pyroorb`` command and the web UI do this. It is faster, and on
        some long arcs it changes which observations end up rejected; without it the fit uses
        nothing but the observations given.

    Returns
    -------
    FittedOrbit

    Warns
    -----
    UndeterminedOrbitWarning
        When the observations do not determine the orbit (``fit.is_determined`` is False).
        The nominal solution is still returned; :meth:`FittedOrbit.determinacy` says what the
        observations do constrain.

    Raises
    ------
    ValueError
        Fewer than three observations, or an unknown ``nongrav`` or ``center``.
    """
    from pyroorb import fit_object as _fit_object
    key = "auto" if nongrav is None else str(nongrav).lower()
    forced = key in _FORCED_NONGRAV
    if not (key in _NONGRAV or forced):
        raise ValueError(f"nongrav must be one of {sorted(_NONGRAV) + list(_FORCED_NONGRAV)}, "
                         f"not {nongrav!r}")
    if center is not None:
        if forced:
            raise ValueError("a planetary satellite is fitted gravity-only; "
                             "nongrav cannot be forced with center")
        from pyroorb import planetocentric as _pc
        center = _pc.resolve_planet(center)
    meta = getattr(observations, "meta", None) or None
    if designation is None:
        names = (meta or {}).get("designations") or sorted(
            {o.get("designation") for o in observations if o.get("designation")})
        designation = names[0] if names else "object"
    records = list(observations)
    seed = None
    if mpc_seed and center is None:
        # fetch_mpc_seed reads meta["designation"] (what the CLI's fetch puts there)
        seed = _fit_object.fetch_mpc_seed(records, {"designation": designation, **(meta or {})})
    table = _table(obscodes)
    fit = _fit_object.fit_orbit(records, table, desig=designation, meta=meta,
                                force=False, auto_nongrav=False if forced else _NONGRAV[key],
                                center=center,
                                auto_center=None if detect_satellite else False,
                                mpc_seed=seed)
    if forced:
        # What notebooks 05 and 06 did by hand: the model on the gravity fit's observation
        # set, kept whether or not it improves the fit, then the radar folded back in (the
        # joint state + A + radar solve).
        fit = _fit_object.refine_nongrav(fit, model=key, revert_if_worse=False,
                                         asteroids=bool(getattr(fit, "asteroids", False)))
        radar_rows = (meta or {}).get("radar")
        if radar_rows:
            from pyroorb import radar as _radar
            fit = _radar.refine_radar(fit, radar_rows, table)
    if not fit.is_determined:
        warnings.warn(f"{designation}: the observations leave the orbit undetermined; its "
                      f"elements are a nominal solution only (fit.determinacy())",
                      UndeterminedOrbitWarning, stacklevel=2)
    return fit


# Not `digest2`: that name is the vendored subpackage (pyroorb/digest2/), and importing it
# would rebind the package attribute over the function.
def digest2_scores(observations, *, per_tracklet=False):
    """digest2 class scores: what kind of object this astrometry looks like.

    digest2 (Keys et al. 2019, the MPC's NEOCP classifier; vendored) needs no orbit fit. It
    scores every bound orbit the observations admit against a model of the Solar System
    population. Each score runs 0-100; classes overlap (NEO contains N22 and N18) and do not
    sum to 100. digest2 samples orbits at random, so a score can differ by a fraction of a
    point between calls.

    The scores rest on a population model, so they describe the TYPICAL object with this
    astrometry, and are no evidence that an object is not unusual: an NEO seen at low
    elongation near the ecliptic is exactly the case digest2 files under the main belt.

    Parameters
    ----------
    observations : Observations or list of dict
        Usually one short arc: a tracklet, or a few nights.
    per_tracklet : bool, default False
        Score each tracklet separately (grouped by station and ``trksub``, or by station
        within four hours of its first observation) instead of the whole set.

    Returns
    -------
    dict or list of dict or None
        ``{"raw": {class: score}, "noid": {class: score}, "classes": {class: heading},
        "rms_arcsec": float, "n_obs": int}``. ``raw`` scores against the whole modelled
        population; ``noid`` against the part not yet identifiable, the flavour the MPC posts
        on the NEOCP (NEO >= 65). None when digest2 declines (fewer than two observations, no
        motion). With ``per_tracklet``, a list of such dicts, each with ``"indices"``
        (positions in ``observations``) and ``"station"``; tracklets digest2 declines are
        left out.
    """
    from pyroorb import d2score as _d2
    from pyroorb.digest2.classes import CLASS_ABBR, CLASS_HEADING
    heading = dict(zip(CLASS_ABBR, CLASS_HEADING))

    def tidy(sc):
        return {"raw": dict(sc["scores"]), "noid": dict(sc["noid_scores"]),
                "classes": {k: heading.get(k, k) for k in sc["scores"]},
                "rms_arcsec": float(sc["rms"]), "n_obs": int(sc["n_obs"])}

    records = list(observations)
    if not per_tracklet:
        sc = _d2.scores(records)
        return None if sc is None else tidy(sc)
    return [{**tidy(t), "indices": list(t["indices"]), "station": t["station"]}
            for t in _d2.tracklet_scores(records)]


def _to_jd(t):
    """A time as a UTC Julian date: a number (already JD), an ISO string, or astropy Time."""
    if isinstance(t, (int, float)):
        return float(t)
    from astropy.time import Time
    return float((t if isinstance(t, Time) else Time(t, scale="utc")).utc.jd)


def ephemeris(fit, start, stop, step=1.0, site="500", *, obscodes=None, full=True,
              observer=None):
    """The ephemeris of a fitted orbit, as seen from an observatory.

    Parameters
    ----------
    fit : FittedOrbit
    start, stop : str, float or astropy.time.Time
        The span, ``stop`` included. ISO strings are read as UTC; a bare number is a UTC
        Julian date. (The
        orbit's own epoch, ``fit.epoch_jd()``, is TDB: pass it as
        ``Time(fit.epoch_jd(), format="jd", scale="tdb")``, not as a number.)
    step : float, default 1.0
        Spacing in days.
    site : str, default "500"
        MPC observatory code; 500 is the geocentre.
    obscodes : dict, optional
        An observatory table to use instead of the bundled MPC table.
    full : bool, default True
        Include the observing-conditions columns (altitude, Moon, galactic latitude...).
        ``False`` is faster and gives positions, distances, brightness and uncertainty.
    observer : array_like, optional
        The observer's geocentric position in place of the station's: ``[x, y, z]`` in au,
        equatorial (ICRF axes), one vector for the whole span or one per epoch ``(n, 3)``.
        This is how a space telescope is placed; it is the vector an observation record
        carries as ``sat_pos_au`` (the MPC's satellite line). ``site`` then only labels the
        rows, and the altitude and airmass columns are left empty.

    Returns
    -------
    list of dict
        One row per epoch. Always: ``jd_utc``, ``date_iso`` (UTC), ``ra_deg``, ``dec_deg``
        and sexagesimal ``ra_hms`` / ``dec_dms`` (astrometric, ICRF, light-time corrected:
        the quantity the observations report and the fit's residuals compare), ``delta_au``
        (from the site), ``r_au`` (from the Sun), ``elong_deg``, ``phase_deg``, ``v_mag``,
        ``motion_arcsec_min``, ``motion_pa_deg``, and the 1-sigma sky-plane uncertainty
        ellipse from the linear covariance: ``unc_smaj_arcsec``, ``unc_smin_arcsec``,
        ``unc_pa_deg``, and ``unc_along_arcsec`` / ``unc_cross_arcsec`` along and across the
        motion. With ``full`` also ``alt_deg``, ``az_deg``, ``airmass``, ``sun_alt_deg``,
        ``moon_alt_deg``, ``moon_illum``, ``moon_sd_deg``, ``lunar_elong_deg``,
        ``gal_l_deg``, ``gal_b_deg``, ``ecl_lon_deg``, ``ecl_lat_deg`` and
        ``constellation``.
    """
    import pyroorb.ephemerides as _eph
    return _eph.compute_ephemeris(fit, _table(obscodes), _to_jd(start), _to_jd(stop),
                                  float(step), site=str(site), full=full, observer=observer)


def propagate(fit, times, *, covariance=False, nongrav=True):
    """The fitted orbit's state at other times.

    Integrated in the force model the orbit was fitted in (DE440 planets, GR, the asteroid
    perturbers if the fit used them, and its non-gravitational acceleration), whatever the
    process-wide settings are at call time.

    Parameters
    ----------
    fit : FittedOrbit
    times : str, float, astropy.time.Time, or a sequence of them
        ISO strings and bare numbers are UTC (Julian dates); an astropy ``Time`` carries its
        own scale, so ``Time(jd, format="jd", scale="tdb")`` for TDB.
    covariance : bool, default False
        Also return the linearly propagated 6 x 6 state covariance at each time. For a
        non-gravitational fit it carries the uncertainty of the fitted A coefficients too
        (and their correlation with the state), as every linear sigma here does.
    nongrav : bool, default True
        ``False`` integrates the same initial state without the non-gravitational force: what
        the orbit would do without it (the covariance is then the state's alone).

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, numpy.ndarray)
        States ``(n, 6)``, heliocentric equatorial (ICRF axes), au and au/day, one row per
        time in the order given; with ``covariance``, also ``(n, 6, 6)``.
    """
    from pyroorb import products as _p
    return _p.propagate(fit, times, covariance=covariance, nongrav=nongrav)


def osculating_elements(state, epoch, *, center="sun"):
    """Osculating two-body elements of any heliocentric equatorial state.

    Parameters
    ----------
    state : array_like
        ``[x, y, z, vx, vy, vz]``, heliocentric equatorial, au and au/day (e.g. a row of
        :func:`propagate`).
    epoch : str, float or astropy.time.Time
        The state's time (a bare number is a UTC Julian date, as everywhere).
    center : str, default "sun"
        ``"sun"``, or a planet (``"jupiter"`` ...) for planetocentric elements with that
        planet's GM.

    Returns
    -------
    dict
        The keys and units of :meth:`FittedOrbit.elements` (ecliptic J2000; ``Tp`` in TDB).
    """
    from pyroorb import products as _p
    return _p.osculating_elements(state, epoch, center=center)


def close_approaches(fit, start, stop, *, body="earth", max_distance=0.05, step=None,
                     nongrav=True, uncertainty=True):
    """Close approaches of the fitted orbit to a planet or the Moon.

    The nominal orbit is scanned on a coarse grid; every local minimum of the distance below
    ``max_distance`` is refined on one-minute samples, and the time of closest approach
    interpolated from the relative motion.

    Parameters
    ----------
    fit : FittedOrbit
    start, stop : str, float or astropy.time.Time
        The window (UTC for strings and numbers).
    body : str, default "earth"
        Any body of :func:`body_states` but the Sun: a planet, the Moon, Pluto, or one of the 16
        asteroid perturbers (``"ceres"``, ``"vesta"``, ...; needs their kernel).
    max_distance : float, default 0.05
        au.
    step : float, optional
        Coarse scan step, days. Default 0.5 for Earth and the Moon, 1 for the other inner
        planets, 2 beyond.
    nongrav : bool, default True
        ``False``: the same orbit without its non-gravitational force.
    uncertainty : bool, default True
        Linear 1-sigma of the distance and the time from the fit's covariance.

    Returns
    -------
    list of dict
        One per approach, in time order: ``body``, ``jd_utc``, ``jd_tdb``, ``date_iso`` (UTC),
        ``distance_au`` (from the body's centre), ``v_rel_km_s``, ``sigma_distance_au``,
        ``sigma_time_days``, ``at_window_edge`` (the minimum is the window's end, not an
        encounter inside it).

    Notes
    -----
    The sigmas are LINEAR: honest for a well-determined orbit, an under-statement when the
    uncertainty region is large or the encounter deflects it strongly (then the encounter's
    outcome is not a single Gaussian). They are not an impact assessment.
    """
    from pyroorb import products as _p
    return _p.close_approaches(fit, start, stop, body=body, max_distance=max_distance,
                               step=step, nongrav=nongrav, uncertainty=uncertainty)


def body_states(body, times):
    """Where a planet, the Moon or a massive asteroid is: its heliocentric state at given times.

    From the same ephemerides PyRoOrb integrates in, so a distance between a fitted orbit
    (:func:`propagate`) and a body is consistent to the integration's precision.

    Parameters
    ----------
    body : str
        ``"sun"``, ``"mercury"``, ``"venus"``, ``"earth"``, ``"moon"``, ``"mars"``,
        ``"jupiter"``, ``"saturn"``, ``"uranus"``, ``"neptune"``, ``"pluto"`` (DE440; the outer
        planets are their system barycentres), or one of the 16 asteroid perturbers --
        ``"ceres"``, ``"pallas"``, ``"juno"``, ``"vesta"``, ``"iris"``, ``"hygiea"``,
        ``"eunomia"``, ``"psyche"``, ``"euphrosyne"``, ``"europa"``, ``"cybele"``,
        ``"sylvia"``, ``"thisbe"``, ``"camilla"``, ``"davida"``, ``"interamnia"`` (JPL's
        sb441-n16 kernel, fetched by ``pyroorb-fetch-data --asteroids``). Case is ignored.
    times : str, float, astropy.time.Time, or a sequence of them
        Strings and numbers UTC, as everywhere.

    Returns
    -------
    numpy.ndarray
        ``(n, 6)``: heliocentric equatorial position and velocity, au and au/day.

    Raises
    ------
    ValueError
        An unknown body.
    FileNotFoundError
        An asteroid without its kernel installed.
    """
    from pyroorb import products as _p
    return _p.public_body_states(body, times)


def radar_residuals(fit, radar, *, obscodes=None, nongrav=True):
    """Radar delay and Doppler residuals of a fitted orbit.

    Works for any orbit and any radar list -- in particular for an orbit fitted WITHOUT the
    radar, to see how far the optical solution misses it. (A fit that included radar reports
    its joint residuals in ``fit.radar["rows"]``.)

    Parameters
    ----------
    fit : FittedOrbit
    radar : list of dict
        Radar astrometry, as :func:`fetch_observations` / :func:`read_observations` put it
        in ``meta["radar"]``.
    obscodes : dict, optional
        An observatory table to use instead of the bundled MPC table.
    nongrav : bool, default True

    Returns
    -------
    list of dict
        Per measurement: ``jd_utc`` (receive time), ``kind`` ("delay" or "doppler"), ``tx``,
        ``rx``, ``observed``, ``computed``, ``o_c`` (s round trip, or Hz), ``sigma``, ``z``
        (O-C in sigmas), and ``o_c_km`` (delay, as one-way range) or ``o_c_mm_s`` (Doppler,
        as range rate).
    """
    from pyroorb import products as _p
    return _p.radar_residuals(fit, radar, _table(obscodes), nongrav=nongrav)


def radar_prediction(fit, times, station, *, receiver=None, freq_hz=8560e6, obscodes=None,
                     nongrav=True, uncertainty=True):
    """Predicted radar round-trip delay and Doppler shift, for planning observations.

    Parameters
    ----------
    fit : FittedOrbit
    times : str, float, astropy.time.Time, or a sequence of them
        Receive times (UTC for strings and numbers).
    station : str
        MPC code of the transmitter (e.g. ``"253"``, Goldstone DSS-14; ``"252"``, DSS-13;
        ``"254"``, Haystack).
    receiver : str, optional
        MPC code of the receiver for a bistatic observation (e.g. ``"256"``, Green Bank).
        Default: monostatic.
    freq_hz : float, default 8.56e9
        Transmitter frequency (Goldstone X band).
    obscodes : dict, optional
    nongrav : bool, default True
    uncertainty : bool, default True
        Linear 1-sigma of the delay and Doppler from the fit's covariance.

    Returns
    -------
    list of dict
        Per time: ``jd_utc``, ``date_iso``, ``delay_s`` (round trip, including the Shapiro
        delay), ``doppler_hz``, ``range_au`` (half the round-trip light time),
        ``sigma_delay_s``, ``sigma_doppler_hz``.

    Notes
    -----
    Geometry only: whether the object is above the horizon or bright enough in radar is not
    checked (``ephemeris(..., site=station)`` gives the altitude).
    """
    from pyroorb import products as _p
    return _p.radar_prediction(fit, times, str(station), _table(obscodes), receiver=receiver,
                               freq_hz=freq_hz, nongrav=nongrav, uncertainty=uncertainty)


def sample_orbits(fit, n=500, method="covariance", *, rng_seed=0, n_sigma=3.0):
    """An ensemble of orbits the observations allow ("clones", "variants").

    Parameters
    ----------
    fit : FittedOrbit
    n : int, default 500
        How many orbits. For ``"lov"`` the number of steps along the line; for ``"sr"`` the
        number of trial orbits drawn, of which those that fit are kept.
    method : {"covariance", "lov", "sr"}, default "covariance"
        ``"covariance"``: random draws from the fit's full parameter covariance, the state and
        any non-gravitational coefficients together (each member carries its own A1/A2/A3).
        Linear in where the orbits are drawn, non-linear in everything computed from them.
        ``"lov"``: orbits along the line of variations, each re-fitted to the observations
        with only its position along the line held. It follows a curved valley of solutions
        far better than the covariance; gravity-only fits; ``n_sigma`` sets the span.
        ``"sr"``: statistical ranging, a SINGLE-NIGHT tool -- trial orbits through the first
        and last lines of sight at sampled distance and radial velocity, kept when they fit
        the astrometry. It sees the range ambiguity a local covariance hides, and its two-body
        trials make the spread an upper bound. Families are often small (measured on single
        survey tracklets: 0 to 130 of 400 trials kept), so check ``len(ensemble)``; on two or
        more nights the fitting orbits are too thin a sliver to sample and it usually finds
        none -- use ``"lov"`` or ``"covariance"`` there.
    rng_seed : int, default 0
        For ``"covariance"``.
    n_sigma : float, default 3.0
        For ``"lov"``: the members span +-n_sigma along the line, at equal-probability steps.

    Returns
    -------
    OrbitEnsemble

    Raises
    ------
    ValueError
        When the method cannot characterise this fit (no LOV for a non-gravitational fit, no
        SR family for a long arc, ...); the message says why.

    See Also
    --------
    range_posterior : the distance distribution of a short arc, with the population prior.
    """
    from pyroorb import ensemble as _e
    return _e.sample_orbits(fit, n, method, rng_seed=rng_seed, n_sigma=n_sigma)


def range_posterior(fit, prior="population"):
    """How far away is a short-arc object? The range profile, as a distribution.

    The range profile fixes the line-of-sight distance from the observer at the first
    included observation, re-fits the other five degrees of freedom, and records chi^2. On a
    tracklet it is usually a long flat valley: every distance from a few lunar distances to
    the main belt fits, and the least-squares nominal sits wherever the fit happened to stop.

    The POPULATION PRIOR adds what the astrometry does not know: how many real objects there
    are on each of those orbits, from digest2's model of the Solar System population (the
    Pan-STARRS Synthetic Solar System Model, Grav et al. 2011), including how common a body of
    the implied absolute magnitude is at each distance. A V = 21 slow mover then lands in
    the main belt rather than at 0.02 au.

    The prior describes the TYPICAL object with this astrometry. It is the wrong question
    for anything that asks whether an object is unusual -- hazard, impact, follow-up priority
    -- and PyRoOrb's own fit, hazard quantities and determinacy never use it: an NEO seen at
    low elongation near the ecliptic is exactly the object this prior files in the main belt.

    Parameters
    ----------
    fit : FittedOrbit
        Usually a fit to one tracklet or a few nights.
    prior : {"population", None}, default "population"
        ``None``: the likelihood alone, flat in distance.

    Returns
    -------
    dict
        ``rho_au`` (the trial distances), ``chi2``, ``likelihood``, ``prior`` (None without
        one), ``weight`` (posterior mass per node, summing to 1), ``bound`` (per node),
        ``rho_percentiles`` (``{5: , 50: , 95: }``, au), ``rho_nominal_au`` (the least-squares
        fit's distance), ``vmag`` (the arc's mean V, or None) and ``epoch_tdb``.

    Raises
    ------
    ValueError
        No range profile for this arc, or the prior cannot be formed.
    """
    from pyroorb import ensemble as _e
    post = _e.range_posterior(fit, prior=prior)
    return {k: v for k, v in post.items() if not k.startswith("_")}


def most_probable_orbit(fit):
    """The orbit to show for an arc that does not determine one: the population prior's choice.

    The range-profile orbit at the population posterior's MEDIAN distance
    (:func:`range_posterior`), its radial velocity chosen by the prior too, re-fitted to the
    observations. It fits the astrometry as well as the least-squares nominal does; what
    differs is which of the many equally good orbits it is: the most typical one, rather than
    wherever the fit happened to stop. This is the orbit the web UI draws on "show anyway".

    Every caveat of :func:`range_posterior` applies: it is a statement about the population,
    not evidence about this object, and must never feed a hazard or follow-up decision.

    Parameters
    ----------
    fit : FittedOrbit
        A fit to a single tracklet (where the range profile decides determinacy).

    Returns
    -------
    FittedOrbit or None
        A fitted orbit (all methods work) carrying ``most_probable`` =
        ``{"rho_med": au, "rho_pct": {"5": , "50": , "95": }, "vmag": }``; None off the
        single-tracklet regime, when the arc fits no orbit at any distance, or when no
        posterior can be formed.
    """
    from pyroorb import ensemble as _e
    return _e.most_probable_orbit(fit)


class asteroid_perturbers:
    """Fit with the 16 most massive asteroids as perturbers (JPL's sb441-n16 kernel).

    Works as a call (on from here on) or as a block that restores the previous setting::

        pyroorb.asteroid_perturbers()                 # every later fit
        with pyroorb.asteroid_perturbers():           # only inside the block
            fit = pyroorb.fit_orbit(obs)
        with pyroorb.asteroid_perturbers(False):      # planets only
            ...

    The setting is process-wide and not thread-safe. A fit records the force model it used
    (``fit.force_model``), and every prediction from it -- :func:`ephemeris`,
    :func:`propagate`, :func:`close_approaches`, the radar functions, the ensembles -- uses
    that model afterwards, whatever this setting is by then. An object that is one
    of the 16 is always fitted with itself left out. The kernel (~600 MB) is fetched by
    ``pyroorb-fetch-data --asteroids``.

    Parameters
    ----------
    on : bool, default True

    Raises
    ------
    FileNotFoundError
        ``on`` and the kernel is not installed.
    """

    def __init__(self, on=True):
        import pyroorb.force_assist as _fa
        self._fa = _fa
        self._previous = _fa.ASTEROIDS_PATH
        path = _fa.asteroid_kernel_path() if on else None
        if on and path is None:
            raise FileNotFoundError("the asteroid perturber kernel is not installed; "
                                    "run pyroorb-fetch-data --asteroids")
        _fa.ASTEROIDS_PATH = path

    def __repr__(self):
        path = self._fa.ASTEROIDS_PATH
        return ("asteroid perturbers on (" + os.path.basename(path) + ")") if path else \
            "asteroid perturbers off"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._fa.ASTEROIDS_PATH = self._previous
        return False


_EXPORTS = {"mpcorb": "to_mpcorb", "sof": "to_sof", "eightline": "to_eightline",
            "covar": "to_covar_json"}


def export(fit, format="mpcorb"):
    """A fitted orbit in an exchange format.

    Parameters
    ----------
    fit : FittedOrbit
    format : {"mpcorb", "sof", "eightline", "covar"}, default "mpcorb"
        ``"mpcorb"``: one MPCORB.DAT line. ``"sof"``: the MPC's Standard Orbit Format.
        ``"eightline"``: find_orb's eight-line elements. ``"covar"``: find_orb's
        ``covar.json`` (state, full covariance, nongrav parameters).

    Returns
    -------
    str
        The nominal orbit -- also for an orbit the observations do not determine, so check
        ``fit.is_determined`` first. (The command line and the web UI refuse those exports.)

    Raises
    ------
    ValueError
        An unknown format.
    """
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
    from pyroorb.console import utf8_stdio
    utf8_stdio()
    from pyroorb import cli
    cli.main()


def _web():
    """Console entry point (`pyroorb-web ...`) — launch the PyRoOrb web UI.
    Honours $PORT / $HOST (set by Railway and friends)."""
    from pyroorb.envfile import load_env_file
    load_env_file()
    from pyroorb.console import utf8_stdio
    utf8_stdio()
    from pyroorb.web import server
    server.main()


def _itf():
    """Console entry point (`pyroorb-itf search|recovery-suite|maintain ...`) — the ITF CLI
    (also `python -m pyroorb.itf`)."""
    from pyroorb.envfile import load_env_file
    load_env_file()
    from pyroorb.console import utf8_stdio
    utf8_stdio()
    from pyroorb.itf.__main__ import main
    return main()


def _fetch_data():
    """Console entry point (`pyroorb-fetch-data [--asteroids]`) — download the reference
    tables a fit needs (DE440, the debias tables; optionally the asteroid ephemeris)."""
    from pyroorb.envfile import load_env_file
    load_env_file()
    from pyroorb.console import utf8_stdio
    utf8_stdio()
    from pyroorb.fetch_data import main
    return main()


__all__ = ["load_settings", "read_observations", "fetch_observations", "fit_orbit",
           "asteroid_perturbers", "ephemeris", "export", "digest2_scores", "propagate",
           "osculating_elements", "close_approaches", "body_states", "radar_residuals",
           "radar_prediction",
           "sample_orbits", "range_posterior", "most_probable_orbit", "OrbitEnsemble",
           "FittedOrbit", "Observations", "UndeterminedOrbitWarning", "obscodes",
           "load_obscodes", "__version__"]
