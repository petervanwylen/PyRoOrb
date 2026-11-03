"""Star-catalog astrometric debiasing — Eggl/Farnocchia/Chesley/Chamberlin (EFCC18).

Different reference star catalogs carry systematic position offsets (and proper-
motion-induced drifts) relative to Gaia-DR2. These biases are often LARGER than the
random per-observation error, so find_orb/JPL/MPC subtract them before fitting. We
implement the same scheme using the authoritative JPL/Eggl bias table (`bias.dat`,
NSIDE=64 HEALPix, 26 catalogs) and the HEALPix pixel lookup in healpix.py.

Data (download once; see DEBIAS_DIR): JPL `debias_2018.tgz` ->
  bias.dat   : 49152 tiles x 26 catalogs x [dRA, dDEC, pmRA, pmDEC]
               dRA/dDEC arcsec at J2000 (dRA already includes cosδ); pm in mas/yr.
  tiles.dat  : tile centers (used only to confirm RING ordering — verified 100%).

Per the bias.dat README, the correction to SUBTRACT from an observed (RA, Dec) is
  ΔRA  = (dRA  + (t-2000)*pmRA /1000) / cos(Dec)     [arcsec, in RA]
  ΔDec =  dDEC + (t-2000)*pmDEC/1000                  [arcsec]
with t the observation epoch in years. Biases are w.r.t. Gaia-DR2, so observations
already on Gaia-DR2 (or DR3) get ~zero correction.

PROVENANCE: data + formula from Eggl et al. (2020, Icarus) via JPL SSD
(https://ssd.jpl.nasa.gov/ftp/ssd/debias/); same scheme find_orb uses (bias.cpp).
No find_orb code copied. numpy.
"""
import os
from pathlib import Path
import numpy as np

from pyroorb import healpix

from pyroorb import datadirs

DEBIAS_DIR = Path(os.environ.get("PYROORB_DEBIAS_DIR") or datadirs.table_dir("debias"))
# JPL/Eggl EFCC18 bias table (~10 MB tgz -> bias.dat + tiles.dat). Auto-fetched into
# DEBIAS_DIR on first use if absent (see ensure_debias_tables); set PYROORB_DEBIAS_AUTOFETCH=0
# to opt out (then debiasing is simply skipped when the table is missing, as before).
DEBIAS_URL = "https://ssd.jpl.nasa.gov/ftp/ssd/debias/debias_2018.tgz"
NSIDE = 64
J2000 = 2451545.0
_FETCH_TRIED = False    # at-most-once auto-fetch guard (so a failed download isn't retried per batch)

# 26 catalog single-char MPC codes, in bias.dat column order.
CATALOG_CODES = list("abcdegijlmnopqrtuvwLNQRSUY")
assert len(CATALOG_CODES) == 26

# ADES astCat value -> MPC catalog code. Catalogs not in the bias table (notably
# Gaia-DR2 itself, the reference, and Gaia-DR3) map to None => zero correction.
ADES_CAT_TO_CODE = {
    "USNOA1": "a", "USNOSA1": "b", "USNOA2": "c", "USNOSA2": "d", "UCAC1": "e",
    "Tyc2": "g", "Tycho2": "g", "GSC11": "i", "GSC12": "j", "ACT": "l",
    "GSCACT": "m", "SDSS8": "n", "USNOB1": "o", "PPM": "p", "UCAC4": "q",
    "UCAC2": "r", "PPMXL": "t", "UCAC3": "u", "NOMAD": "v", "CMC14": "w",
    "2MASS": "L", "SDSS7": "N", "CMC15": "Q", "SSTRC4": "R", "URAT1": "S",
    "Gaia1": "U", "UCAC5": "Y",
    # reference / too-new (no debias):
    "Gaia2": None, "Gaia3": None, "Gaia3E": None,
}

_BIAS = None        # cached (49152, 26, 4) array
_CODE_IDX = {c: i for i, c in enumerate(CATALOG_CODES)}


def available():
    return (DEBIAS_DIR / "bias.dat").exists() or (DEBIAS_DIR / "bias.npy").exists()


def ensure_debias_tables(verbose=False):
    """Download + extract the EFCC18 bias table into DEBIAS_DIR on first use if absent,
    so a fresh box / container self-provisions (the same way the ephemeris kernels do).
    Idempotent and at-most-once per process. Graceful: a failed fetch (offline, firewalled,
    JPL unreachable) leaves debiasing OFF -- bias_for() then returns (0, 0) and fits proceed
    undebiased, exactly as before this auto-fetch existed -- it never raises. Returns
    available(). Honours PYROORB_DEBIAS_AUTOFETCH=0 to skip the download entirely."""
    global _FETCH_TRIED
    if available():
        return True
    if _FETCH_TRIED or os.environ.get("PYROORB_DEBIAS_AUTOFETCH", "1") == "0":
        return False
    _FETCH_TRIED = True
    import urllib.request, tarfile, tempfile
    tmp_path = None
    try:
        DEBIAS_DIR.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"  downloading EFCC18 debias tables (~10 MB) from {DEBIAS_URL} ...")
        fd, tmp_path = tempfile.mkstemp(suffix=".tgz")
        os.close(fd)
        urllib.request.urlretrieve(DEBIAS_URL, tmp_path)
        with tarfile.open(tmp_path) as tf:
            for m in tf.getmembers():
                base = Path(m.name).name
                if base in ("bias.dat", "tiles.dat"):
                    m.name = base                       # flatten any leading path
                    tf.extract(m, DEBIAS_DIR, filter="data")
    except Exception as ex:                             # offline / firewall / partial -> no debias
        if verbose:
            print(f"  debias download failed ({ex}); continuing without debiasing")
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return available()


def _load():
    """Load bias.dat (cached to bias.npy on first use). Shape (NPIX, 26, 4)."""
    global _BIAS
    if _BIAS is not None:
        return _BIAS
    npy = DEBIAS_DIR / "bias.npy"
    if npy.exists():
        _BIAS = np.load(npy)
    else:
        # bias.dat is whitespace-delimited but negative values can touch the
        # preceding one ("..1.04-0.49.."), so split with a number regex, not
        # str.split / np.loadtxt. One tile per line, 104 values each.
        import re
        rx = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
        vals = []
        with open(DEBIAS_DIR / "bias.dat", encoding="utf-8") as f:
            for line in f:
                if line.startswith("!") or not line.strip():
                    continue
                vals.extend(rx.findall(line))
        arr = np.array(vals, dtype=np.float32)
        assert arr.size == 49152 * 26 * 4, arr.size
        _BIAS = arr.reshape(49152, 26, 4)
        np.save(npy, _BIAS)
    return _BIAS


def cat_code(astcat):
    """Star catalog -> MPC catalog code (or None if unknown / reference / absent).

    Accepts BOTH forms the two ingest paths carry: an ADES `astCat` NAME ("UCAC4",
    "Gaia2") and the raw single-letter MPC code that lives in OBS80 col 72 ("q", "V").
    They can't collide -- every ADES name is >= 3 chars -- and a raw code is used only
    when it actually indexes the bias table, so Gaia-DR2/DR3 ('V'/'W'/'X', the reference
    frame) and the catalogs EFCC18 doesn't tabulate return None => zero correction, the
    same answer the ADES map gives them."""
    if not astcat:
        return None
    s = str(astcat).strip()
    if len(s) == 1:
        return s if s in _CODE_IDX else None      # OBS80 col-72 raw MPC catalogue code
    return ADES_CAT_TO_CODE.get(s)


def bias_for(ra_deg, dec_deg, astcat, jd_utc):
    """Correction (ΔRA_deg, ΔDec_deg) to SUBTRACT from the observed position to
    debias to the Gaia-DR2 frame. Returns (0, 0) if the catalog has no table entry
    (unknown, or already Gaia-DR2/DR3). `ra/dec` may be arrays or scalars."""
    code = cat_code(astcat)
    # Once the table is loaded its presence is settled: do not stat the file again per
    # observation -- on a slow (network / virtiofs) mount that stat is ~0.5 s, and a
    # 7000-obs comet spent over an hour in it before fitting anything.
    if code is None or (_BIAS is None and not available()):
        return 0.0, 0.0
    bias = _load()
    pix = healpix.ang2pix(NSIDE, ra_deg, dec_deg)
    # widen the float32 table entries to float64 up front: the RA term is promoted anyway
    # (by the /cos(dec) below), so leaving them narrow gave the two axes DIFFERENT precision
    # for no reason -- the Dec correction carried ~7 digits where RA carried ~16.
    vals = bias[pix, _CODE_IDX[code]].astype(float)
    dRA, dDEC, pmRA, pmDEC = vals.T if np.ndim(pix) else vals
    yr = (jd_utc - J2000) / 365.25                       # years since 2000.0
    cosd = np.cos(np.radians(dec_deg))
    dra_arcsec = (dRA + yr * pmRA / 1000.0) / cosd        # arcsec, in RA
    ddec_arcsec = dDEC + yr * pmDEC / 1000.0              # arcsec
    return dra_arcsec / 3600.0, ddec_arcsec / 3600.0


def debias_obs(obs):
    """Debias a list of obs dicts IN PLACE (uses o['astcat'], o['ra_deg'],
    o['dec_deg'], o['jd_utc']). Records the raw position and applied shift; returns
    the number of observations actually corrected (non-zero).

    IDEMPOTENT: an obs already carrying a `debiased` verdict is skipped, so running this
    over a list twice -- or over a mix of fresh and already-corrected records (a frozen
    fixture, or ADES obs grafted onto OBS80 ones) -- can never apply the shift twice.
    Each ingest path calls it once on its own fresh records; this guard is what keeps
    that true now that there are two of them."""
    ensure_debias_tables()          # self-provision on first use; no-op if present / opted out
    if not available():
        return 0
    n = 0
    for o in obs:
        if "debiased" in o:
            continue                # already corrected (or already judged uncorrectable)
        dra, ddec = bias_for(o["ra_deg"], o["dec_deg"], o.get("astcat"), o["jd_utc"])
        if dra or ddec:
            o["ra_raw"], o["dec_raw"] = o["ra_deg"], o["dec_deg"]
            o["ra_deg"] = o["ra_deg"] - dra
            o["dec_deg"] = o["dec_deg"] - ddec
            o["debias_arcsec"] = (dra * 3600.0 * np.cos(np.radians(o["dec_deg"])),
                                  ddec * 3600.0)
            o["debiased"] = True
            n += 1
        else:
            o["debiased"] = False
    return n
