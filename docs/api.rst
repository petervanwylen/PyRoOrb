API reference
=============

.. currentmodule:: pyroorb

Everything on this page is importable from the top-level package (``import pyroorb``).
Anything not listed here -- other ``FittedOrbit`` methods, the record keys not named under
:class:`Observations`, every submodule -- is internal and may change in any release, even
when a notebook uses it (those cells are marked *internal API, may change*).

Stability
---------

**Stable** names keep their meaning and signature through 1.x; a change goes through a
deprecation warning lasting at least one minor release. **Provisional** names are documented
and safe to use, but their return shape may still change before 1.0: they are the pieces
being unified into one ensemble/uncertainty interface. Experimental arguments say so in their
description.

Settings
--------

.. autofunction:: load_settings

Observations
------------

.. autofunction:: read_observations

.. autofunction:: fetch_observations

.. autoclass:: Observations
   :members: select

.. autofunction:: obscodes

Fitting
-------

.. autofunction:: fit_orbit

.. autoclass:: asteroid_perturbers

.. autoclass:: UndeterminedOrbitWarning

.. autoclass:: FetchError

Products
--------

.. autofunction:: ephemeris

.. autofunction:: export

Dynamics
--------

Each of these integrates in the force model the orbit was fitted in, and gives LINEAR
uncertainties from the fit's covariance; :func:`sample_orbits` is the non-linear tool.

.. autofunction:: propagate

.. autofunction:: osculating_elements

.. autofunction:: close_approaches

.. autofunction:: body_states

Radar
-----

.. autofunction:: radar_residuals

.. autofunction:: radar_prediction

Ensembles of orbits
-------------------

.. autofunction:: sample_orbits

.. autoclass:: OrbitEnsemble
   :members: propagate, elements, element_percentiles, bound_fraction, sky_positions,
             close_approaches

Short arcs and the population prior
-----------------------------------

.. autofunction:: digest2_scores

.. autofunction:: range_posterior

.. autofunction:: most_probable_orbit

The fitted orbit
----------------

.. autoclass:: FittedOrbit

Is it determined?
~~~~~~~~~~~~~~~~~

.. autoattribute:: FittedOrbit.is_determined

.. automethod:: FittedOrbit.determinacy

The solution
~~~~~~~~~~~~

.. automethod:: FittedOrbit.elements
.. automethod:: FittedOrbit.element_sigmas
.. automethod:: FittedOrbit.epoch_jd
.. automethod:: FittedOrbit.epoch_iso
.. automethod:: FittedOrbit.semimajor_axis
.. automethod:: FittedOrbit.eccentricity
.. automethod:: FittedOrbit.state
.. automethod:: FittedOrbit.state_vector
.. automethod:: FittedOrbit.covariance_matrix
.. automethod:: FittedOrbit.correlation_matrix
.. automethod:: FittedOrbit.parameter_covariance
.. automethod:: FittedOrbit.uncertainty_parameter

How the observations fit
~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: FittedOrbit.rms_residual
.. automethod:: FittedOrbit.rms_normalized
.. automethod:: FittedOrbit.residuals
.. automethod:: FittedOrbit.excluded_observations
.. automethod:: FittedOrbit.included_observations
.. automethod:: FittedOrbit.n_observations
.. automethod:: FittedOrbit.n_total_observations
.. automethod:: FittedOrbit.n_excluded
.. automethod:: FittedOrbit.degrees_of_freedom
.. automethod:: FittedOrbit.n_free_params
.. automethod:: FittedOrbit.arc_length_days
.. automethod:: FittedOrbit.n_oppositions
.. automethod:: FittedOrbit.observatory_codes
.. automethod:: FittedOrbit.is_combined
.. automethod:: FittedOrbit.linked_designations

Physical and hazard quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: FittedOrbit.absolute_magnitude
.. automethod:: FittedOrbit.slope_parameter
.. automethod:: FittedOrbit.n_magnitudes
.. automethod:: FittedOrbit.magnitude_residuals
.. automethod:: FittedOrbit.geocentric_range
.. automethod:: FittedOrbit.moid_earth
.. automethod:: FittedOrbit.orbit_class
.. automethod:: FittedOrbit.is_pha
.. automethod:: FittedOrbit.tisserand_jupiter

Provisional: non-linear uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The per-method summaries the web UI shows. :func:`sample_orbits` returns the same families as
orbits you can propagate, project onto the sky and scan for close approaches, and is the
interface to build on; these may be removed or reshaped before 1.0.

.. automethod:: FittedOrbit.monte_carlo_uncertainty
.. automethod:: FittedOrbit.element_ranges
.. automethod:: FittedOrbit.bound_fraction
.. automethod:: FittedOrbit.lov_uncertainty
.. automethod:: FittedOrbit.sr_uncertainty
.. automethod:: FittedOrbit.range_profile
.. automethod:: FittedOrbit.range_verdict

Provisional: non-gravitational forces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fit.nongrav_model`` and ``fit.nongrav_params`` (see :class:`FittedOrbit`) name the model
and its coefficients. The honest uncertainty of a coefficient is its profile-likelihood
interval, which can be asymmetric or open:

.. automethod:: FittedOrbit.rational_nongrav_intervals
.. autoattribute:: FittedOrbit.nongrav_unconstrained

Provisional: everything at once
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: FittedOrbit.as_dict
