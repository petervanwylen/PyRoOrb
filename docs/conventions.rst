Conventions
===========

One place for the units, frames and time scales every function uses.

Time
----

==========================  ==================================================================
Input times                 ISO strings are UTC; a bare number is a **UTC** Julian date;
                            an astropy ``Time`` carries its own scale.
Observation times           ``jd_utc``: Julian date, UTC.
The orbit's epoch           ``fit.epoch_jd()`` / ``fit.epoch_tdb``: Julian date, **TDB**.
Time of perihelion          ``elements()["Tp"]``: Julian date, TDB.
==========================  ==================================================================

The two scales differ by about 69 s. To use the epoch as an input time, wrap it:
``Time(fit.epoch_jd(), format="jd", scale="tdb")``.

Frames
------

* **Elements** (:meth:`~pyroorb.FittedOrbit.elements`): osculating, heliocentric, two-body
  about the Sun, in the ecliptic and equinox of J2000 (mean obliquity 23.43928111 deg).
* **State** (:meth:`~pyroorb.FittedOrbit.state`) and the ``"state"`` covariance: heliocentric,
  **equatorial** (ICRF axes), au and au/day. A non-gravitational fit appends its A
  coefficients after the six state components.
* **Sky positions** (observations, ephemerides): RA/Dec in degrees, ICRF. Ephemeris positions
  are astrometric (light-time corrected), the quantity the observations report.

Units
-----

=====================  ====================================================================
Distances              au (``a``, ``q``, ``Q``, ``delta_au``, ``r_au``, MOID)
Angles                 degrees (``i``, ``asc_node``, ``arg_per``, ``M``, RA, Dec)
Mean motion, period    deg/day, years
Residuals, sigmas      arcsec; the RA residual is multiplied by cos(dec)
Non-grav coefficients  au/day^2 (A1 radial, A2 transverse, A3 normal)
=====================  ====================================================================

Force model
-----------

Every fit integrates in DE440 (planets, the Moon, Pluto), general relativity, and the
non-gravitational acceleration the fit keeps. The 16 most massive asteroids join the whole fit
inside :class:`~pyroorb.asteroid_perturbers` (the same as setting ``PYROORB_ASTEROIDS_PATH``);
without it a library fit uses the planets only. (The command line's ``--asteroids`` and the web
UI's toggle instead re-polish a finished fit with them.)
``fit.force_model`` names the field (``+16ast`` when the asteroids were used). Every prediction
from a fit -- :func:`~pyroorb.ephemeris`, :func:`~pyroorb.propagate`,
:func:`~pyroorb.close_approaches`, the radar functions and the ensembles -- integrates in that
same field, whatever the global setting is when it is called.

Residuals and RMS
-----------------

* Residuals are O-C, **observed minus computed**.
* :meth:`~pyroorb.FittedOrbit.rms_residual` is the **per-axis** RMS (RA cos(dec) and Dec
  residuals pooled), the find_orb, MPC and JPL convention. It is a factor sqrt(2) smaller than
  the RMS of the total angular residual per observation.
* :meth:`~pyroorb.FittedOrbit.rms_normalized` is the RMS of residual/sigma; near 1 means the
  residuals match the weights. The PyRoOrb weights are deliberately conservative, so values
  well below 1 are normal.

Uncertainty
-----------

* The linear covariance is scaled by the residuals but never shrunk by them: a fit whose
  residuals are smaller than its weights reports the weights' uncertainty, not less.
* Element sigmas come from that covariance through the element Jacobian (linear propagation);
  :func:`~pyroorb.propagate`, :func:`~pyroorb.close_approaches` and the radar predictions
  carry the same covariance forward through the state-transition matrix. For an orbit whose
  uncertainty region is large or curved, sample it instead (:func:`~pyroorb.sample_orbits`).
* An orbit the observations do not determine is still returned, as a nominal solution, with
  ``fit.is_determined`` False and an :class:`~pyroorb.UndeterminedOrbitWarning`. The library
  never refuses to give you that solution; it is your call what to do with it.

Population priors
-----------------

:func:`~pyroorb.digest2_scores`, :func:`~pyroorb.range_posterior` (with its default prior) and
:func:`~pyroorb.most_probable_orbit` rest on a model of the Solar System population (the
Pan-STARRS Synthetic Solar System Model, Grav et al. 2011). They describe the TYPICAL object
with the astrometry given, which is the right question for "what is this likely to be" and the
wrong one for "is this object unusual": the fit, its hazard quantities, its uncertainty and
every ensemble never use it.
