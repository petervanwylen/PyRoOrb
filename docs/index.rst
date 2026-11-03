PyRoOrb
=======

**Raw astrometry in, a robust orbit out.** PyRoOrb fits asteroid and comet orbits from ADES
or MPC 80-column astrometry, with realistic uncertainties, in Python, on the REBOUND + ASSIST
dynamics stack. It ports the algorithms of Bill Gray's find_orb.

.. code-block:: python

   import pyroorb

   pyroorb.load_settings()                                # your .env, if you keep one
   obs = pyroorb.fetch_observations("2023 QJ3")           # or read_observations("my.psv")
   fit = pyroorb.fit_orbit(obs, designation="2023 QJ3")

   fit.is_determined, fit.rms_residual()                  # True, 0.24 (arcsec, per axis)
   fit.elements(), fit.element_sigmas()                   # a, e, q, Q, i, node, peri, M, Tp, n, P
   fit.moid_earth(), fit.orbit_class(), fit.uncertainty_parameter()

   eph = pyroorb.ephemeris(fit, "2026-11-01", "2026-12-01", step=1.0, site="568")
   print(pyroorb.export(fit, "mpcorb"))

   # gravity only, for comparison with the automatic non-gravitational choice
   grav = pyroorb.fit_orbit(obs, designation="2023 QJ3", nongrav="none")

   # what kind of object did the first night look like? (no fit needed)
   first_night = obs.select(stop=min(o["jd_utc"] for o in obs) + 0.5)
   pyroorb.digest2_scores(first_night)["raw"]["NEO"]

   # where will it be, and how sure are we?
   state, cov = pyroorb.propagate(fit, "2030-01-01", covariance=True)
   pyroorb.close_approaches(fit, "2026-01-01", "2036-01-01", body="mars", max_distance=0.2)
   clones = pyroorb.sample_orbits(fit, 500)               # or method="lov" / "sr"
   clones.sky_positions("2027-01-01", site="568")         # the non-linear sky uncertainty

   # one night: how far away is it likely to be? (population prior, display only)
   night = pyroorb.fit_orbit(first_night)
   pyroorb.range_posterior(night)["rho_percentiles"]

Installation, the command line and the web UI are described in the project README; worked
science examples are the notebooks in ``notebooks/``.

.. toctree::
   :maxdepth: 2

   api
   conventions
