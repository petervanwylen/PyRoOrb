"""ADES data for a line-based search: the identity registry.

The ITF search is OBS80-line-based by necessity -- the tracklet index stores raw
80-column records, observation identity is a column slice
(:func:`itf.obs80_fields.obs_unique_id`), night/opposition counting reads dates out of
columns, and the MPC identification document we submit is a pile of lines.  None
of that wants to change.

What DOES want to change is the data the ENGINE fits.  Round-tripping an
observation through 80 columns throws away everything ADES carries and OBS80
cannot: the submitter's reported rmsRA/rmsDec (so the fit falls back to the
Veres-2017 station model even when the real per-obs sigma is known), rmsMag,
logSNR, obsID, mode.  For the published seed that loss is total and pointless --
``obs_sources.fetch_observations`` fetched ADES in the first place.

So: **the line stays the identity, ADES becomes the data.**  A registry maps
``obs_unique_id(line) -> the ADES-only fields of that observation``, and
:meth:`AdesRegistry.enrich` overlays them onto the obs dicts parsed from the
lines.  Structure (position, designation, packed id, astCat, satellite/roving
vectors, the debiasing that has already been applied) is left exactly as the
OBS80 parse produced it; what is added is the fields OBS80 could never have
carried -- plus ``band``, the one field it carries LOSSILY (80 columns hold a
single character where ADES names the instrument's actual filter), which is why
it is overlaid rather than trusted from the parse.  See ``_CARRIED``.

Two deliberate non-goals:

* **Positions are never overlaid.**  For the ~74% of ITF rows submitted in the
  1992 format the database's ``ra``/``dec`` columns are just the 80-column
  values re-expressed, so there is nothing to gain; for the natively-ADES rest
  the gain is ~0.01 arcsec, far under the 0.1 arcsec SIGMA_FLOOR, and taking it
  would mean re-deriving the EFCC18 debias correction that the parse already
  applied.  Not worth the risk for a tenth of a sigma.
* **astCat is never overlaid.**  ``debias`` keys on the catalog name spelled the
  way the OBS80 column-72 code maps to it; the database spells the same catalog
  differently ("Gaia3E"), and overwriting it would silently disable debiasing.
"""

from .obs80_fields import obs_unique_id

# The ADES-only keys carried across.  Deliberately NOT ra_deg/dec_deg/jd_utc/
# astcat/designation/packed -- see the module docstring.
_CARRIED = ("sigma_ra", "sigma_dec", "uncert_src", "ades_rms_ra",
            "ades_rms_dec", "sigma_time_sec", "mag_sigma", "log_snr",
            "obs_id", "trksub", "mode", "band")

# `band` is here for the same reason as the sigmas: OBS80 column 71 is ONE
# character, and since photometry.BAND_SHIFT is keyed on the full ADES band that
# collapse now loses real information. An LSST 'Lu' renders as 'u' and would come
# back charged the MPC's generic Sloan u' correction (-2.50) instead of LSST's
# own (-1.77) -- a 0.7 mag error in H and in every displayed photometric O-C.
# Overlaying the band does NOT change the line: the same collapse still runs on
# the way out, so column 71 (and observation identity, which is column-aware)
# is byte-identical. Structure stays OBS80's, data becomes ADES's, as above.


def _extras_from_obs(obs):
    """The ADES-only fields of an already-parsed engine obs dict (the seed path:
    these come from ``ades.records_to_obs`` via the normal ADES fetch)."""
    return {k: obs[k] for k in _CARRIED if obs.get(k) is not None}


def _extras_from_record(rec):
    """The ADES-only fields of a raw ADES record -- an ``obs_sbn`` row, whose
    column names ARE the ADES field names (the MPC's ADES_DF API serves this
    same table, which is why :func:`ades.records_to_obs` reads these very keys).

    The sigma is computed with the SAME :func:`weights.sigma_for` the ADES fetch
    path uses, so a reported rms reaches the fit through one code path only --
    including its floor/ceiling clamp.  When the submitter reported no rms (the
    common case in the ITF: only ~7.5% of rows carry one), no sigma is emitted
    at all and the engine's Veres-2017 station fallback applies unchanged.
    """
    from pyroorb.weights import ades_rms_usable, sigma_for

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    rms_ra, rms_dec = rec.get("rmsra"), rec.get("rmsdec")
    extras = {}
    if ades_rms_usable(rms_ra, rms_dec):
        sra, sdec = sigma_for(rec.get("stn"), rec.get("astcat"), rms_ra, rms_dec)
        extras["sigma_ra"] = float(sra)
        extras["sigma_dec"] = float(sdec)
        extras["uncert_src"] = "ADES"
        extras["ades_rms_ra"] = _f(rms_ra)
        extras["ades_rms_dec"] = _f(rms_dec)
    for key, col in (("sigma_time_sec", "rmstime"), ("mag_sigma", "rmsmag"),
                     ("log_snr", "logsnr")):
        v = _f(rec.get(col))
        if v is not None:
            extras[key] = v
    for key, col in (("obs_id", "obsid"), ("trksub", "trksub"),
                     ("mode", "mode"), ("band", "band")):
        v = rec.get(col)
        if v is not None and str(v).strip():
            extras[key] = str(v).strip()
    return extras


class AdesRegistry:
    """``obs_unique_id`` -> ADES-only fields, for every observation the search
    has seen in a richer form than its 80-column rendering.

    Empty is a perfectly good state: every consumer degrades to exactly the
    previous OBS80-only behaviour, so a search that never gets ADES data (an
    offline harness, a pasted OBS80 blob) is unaffected.
    """

    def __init__(self):
        self._by_id = {}
        self.n_ades_sigma = 0        # how many entries carry a reported sigma

    def __len__(self):
        return len(self._by_id)

    def _put(self, line, extras):
        if not line or not extras:
            return
        key = obs_unique_id(line)
        if not key.strip():
            return
        if extras.get("uncert_src") == "ADES" and key not in self._by_id:
            self.n_ades_sigma += 1
        self._by_id[key] = extras

    def add_obs(self, obs_iter):
        """Register parsed engine obs dicts (the ADES seed fetch).  Each obs is
        keyed by the id of the 80-column line it renders to -- the SAME
        rendering the seed text is built from, so the ids match by
        construction."""
        from pyroorb.obs80 import obs80_base_line
        for o in obs_iter or ():
            extras = _extras_from_obs(o)
            if not extras:
                continue
            line = o.get("obs80_line")
            if not line:
                try:
                    line = obs80_base_line(o).splitlines()[0]
                except Exception:
                    continue
            self._put(line, extras)

    def add_records(self, rows):
        """Register raw ADES records (``obs_sbn`` candidate rows).  Each row
        must carry its own ``obs80`` rendering -- that is the identity."""
        for r in rows or ():
            line = r.get("obs80") or ""
            if len(line) == 160:            # two-line record: identity is line 1
                line = line[:80]
            self._put(line, _extras_from_record(r))

    def trksubs(self, ids):
        """``{obs_unique_id: trksub}`` for those of ``ids`` registered with one
        (the database column, for candidate rows -- see linkage.db_trksubs)."""
        out = {}
        for k in ids:
            trk = (self._by_id.get(k) or {}).get("trksub")
            if trk:
                out[k] = trk
        return out

    def enrich(self, obs_list):
        """Overlay the registered ADES fields onto obs dicts parsed from lines.
        Mutates in place and returns how many observations were enriched."""
        if not self._by_id:
            return 0
        n = 0
        for o in obs_list:
            line = o.get("obs80_line")
            if not line:
                continue
            extras = self._by_id.get(obs_unique_id(line))
            if not extras:
                continue
            o.update(extras)
            n += 1
        return n
