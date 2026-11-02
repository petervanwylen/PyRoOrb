"""Local MPCORB cache + random-asteroid sampling (with optional element filters).

The Minor Planet Center's MPCORB.DAT(.gz) is the catalogue of all ~1.5 M known
asteroid orbits. We cache it locally (in the cache directory, datadirs.cache_dir()) and build a small
parsed index (designation + a, e, i, H) so repeated random sampling is fast.

    from mpcorb import sample_asteroids
    desigs = sample_asteroids(50, seed=7)                       # 50 random asteroids
    neos   = sample_asteroids(20, seed=7, q=(None, 1.3))        # perihelion < 1.3 au
    hi_e   = sample_asteroids(10, e=(0.5, None), a=(2.0, 3.5))  # eccentric main-belt

Anything missing/corrupt is (re)fetched. The download goes through the sandbox's
chunk-corrupting proxy, so it's done in byte-range chunks with a scan-and-patch pass
(re-fetch any block that comes back zero-filled) -- the same fix used for the DE440
ephemeris. Validated by a clean gzip decompression.
"""
import os, gzip, math
from pathlib import Path
import numpy as np

from pyroorb import datadirs

ROOT = Path(__file__).resolve().parent
DATA_DIR = datadirs.cache_dir()  # re-derivable download cache (see datadirs)
MPCORB_URL = "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"
MPCORB_GZ = DATA_DIR / "MPCORB.DAT.gz"
MPCORB_NPZ = DATA_DIR / "MPCORB_index.npz"

# fixed-width MPCORB columns (0-indexed slices), see https://minorplanetcenter.net/iau/info/MPOrbitFormat.html
# (The MPC also publishes a named-field mpcorb_extended.json.gz / query API with these same
#  quantities as keys -- cleaner than fixed-width, but a fresh large download through the
#  sandbox's chunk-corrupting proxy; we parse the already-cached flat file here.)
_COL = {"packed": (0, 7), "H": (8, 13), "incl": (59, 68),
        "ecc": (70, 79), "a": (92, 103), "nopp": (123, 126), "arc": (127, 136)}


def _parse_arc_years(field):
    """MPCORB's observed-arc field: 'YYYY-YYYY' (first-last year) for a MULTI-opposition
    orbit, or 'NNN days' for a single-opposition one. Return the arc length in years
    (the multi-opp year span is what flags VM6-like multi-revolution two-opp objects)."""
    field = field.strip()
    if not field:
        return math.nan
    if field.endswith("days"):
        try:
            return int(field.split()[0]) / 365.25
        except ValueError:
            return math.nan
    parts = field.split("-")                    # 'YYYY-YYYY'
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return float(int(parts[1]) - int(parts[0]))
    return math.nan


# --------------------------------------------------------------------------- download
def _max_zero_run(b):
    isz = (b == 0).astype(np.int8)
    d = np.diff(np.concatenate(([0], isz, [0])))
    s = np.flatnonzero(d == 1); e = np.flatnonzero(d == -1)
    return int((e - s).max()) if s.size else 0


def robust_download(url, dst, thresh=65536, chunk=4 * 1024 * 1024, verbose=True):
    """Download `url` to `dst` defeating the proxy's chunk-corruption: fetch in byte
    ranges, then re-fetch (scan-and-patch) any block that arrives with a >= `thresh`
    zero-run. Returns the path. Requires the server to honour Range requests."""
    import requests
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    size = int(sess.head(url, timeout=60).headers["Content-Length"])

    def fetch(a, b):
        for _ in range(20):
            try:
                r = sess.get(url, headers={"Range": f"bytes={a}-{b-1}",
                                           "Cache-Control": "no-cache"}, timeout=240)
                c = r.content
            except Exception:
                continue
            if len(c) == (b - a) and _max_zero_run(np.frombuffer(c, np.uint8)) < thresh:
                return np.frombuffer(c, np.uint8).copy()
        return None

    d = np.empty(size, dtype=np.uint8); off = 0
    while off < size:
        ln = min(chunk, size - off)
        c = fetch(off, off + ln)
        if c is None:
            raise RuntimeError(f"download failed at byte {off}")
        d[off:off + ln] = c; off += ln
        if verbose and (off // chunk) % 8 == 0:
            print(f"  {off/1e6:.0f}/{size/1e6:.0f} MB", flush=True)
    # scan-and-patch any residual zero-runs (corruption that slipped a chunk boundary)
    for _ in range(40):
        isz = (d == 0).astype(np.int8); df = np.diff(np.concatenate(([0], isz, [0])))
        s = np.flatnonzero(df == 1); e = np.flatnonzero(df == -1)
        runs = [(int(s[k]), int(e[k])) for k in range(len(s)) if e[k] - s[k] >= thresh]
        if not runs:
            break
        for a, b in runs:
            p = fetch(max(0, a - 4096), min(size, b + 4096))
            if p is None:
                raise RuntimeError(f"patch failed at byte {a}")
            d[max(0, a - 4096):min(size, b + 4096)] = p
    d.tofile(dst)
    return dst


def ensure_mpcorb(verbose=True):
    """Return the path to a valid local MPCORB.DAT.gz, downloading it if the cache is
    absent or fails to decompress."""
    if MPCORB_GZ.exists():
        try:
            with gzip.open(MPCORB_GZ, "rb") as f:        # cheap integrity probe
                f.read(1 << 20)
            return MPCORB_GZ
        except Exception:
            if verbose:
                print("cached MPCORB is corrupt -- refetching", flush=True)
    if verbose:
        print(f"downloading MPCORB -> {MPCORB_GZ}", flush=True)
    robust_download(MPCORB_URL, MPCORB_GZ, verbose=verbose)
    with gzip.open(MPCORB_GZ, "rt", errors="replace") as f:   # validate
        sum(1 for _ in f)
    return MPCORB_GZ


# ----------------------------------------------------------------------------- index
def _parse_field(line, key):
    a, b = _COL[key]
    try:
        return float(line[a:b])
    except ValueError:
        return math.nan


def build_index(verbose=True):
    """Parse MPCORB.DAT.gz into a compact .npz (packed designation + a, e, i, H) for
    fast repeated sampling. Rebuilt when missing or older than the catalogue."""
    ensure_mpcorb(verbose=verbose)
    if (MPCORB_NPZ.exists()
            and MPCORB_NPZ.stat().st_mtime >= MPCORB_GZ.stat().st_mtime):
        try:                                           # reuse only if it has the current fields
            if {"nopp", "arc_years"} <= set(np.load(MPCORB_NPZ, allow_pickle=False).files):
                return MPCORB_NPZ
        except Exception:
            pass
    if verbose:
        print("building MPCORB index...", flush=True)
    packed, H, a, e, incl, nopp, arc = [], [], [], [], [], [], []
    with gzip.open(MPCORB_GZ, "rt", errors="replace") as f:
        started = False
        for ln in f:
            if not started:
                if ln.startswith("----------"):
                    started = True
                continue
            if len(ln) > 100 and ln[0:7].strip():
                packed.append(ln[0:7])
                H.append(_parse_field(ln, "H")); a.append(_parse_field(ln, "a"))
                e.append(_parse_field(ln, "ecc")); incl.append(_parse_field(ln, "incl"))
                try:
                    nopp.append(int(ln[_COL["nopp"][0]:_COL["nopp"][1]]))
                except ValueError:
                    nopp.append(0)
                arc.append(_parse_arc_years(ln[_COL["arc"][0]:_COL["arc"][1]]))
    import io
    buf = io.BytesIO()                                 # savez to memory, then write with
    np.savez(buf, packed=np.array(packed), H=np.array(H, float),   # flush+fsync+atomic
             a=np.array(a, float), e=np.array(e, float), incl=np.array(incl, float),
             nopp=np.array(nopp, int), arc_years=np.array(arc, float))
    blob = buf.getvalue()
    tmp = str(MPCORB_NPZ) + ".tmp"                      # rename -- the sandbox FS leaves
    for attempt in range(6):                            # null-byte holes / 512KB truncations
        with open(tmp, "wb") as fp:                     # on big direct writes, so verify the
            fp.write(blob); fp.flush(); os.fsync(fp.fileno())   # tmp loads before replacing
        try:
            np.load(tmp, allow_pickle=False).files
        except Exception:
            if attempt == 5:
                raise
            continue                                    # truncated write -> rewrite
        os.replace(tmp, MPCORB_NPZ)
        break
    if verbose:
        print(f"  indexed {len(packed):,} asteroids", flush=True)
    return MPCORB_NPZ


def _between(x, rng):
    """Boolean mask for x within (lo, hi); either bound may be None."""
    m = np.isfinite(x)
    if rng is None:
        return m
    lo, hi = rng
    if lo is not None:
        m &= x >= lo
    if hi is not None:
        m &= x <= hi
    return m


def sample_asteroids(n, seed=None, *, a=None, e=None, incl=None, q=None, H=None,
                     nopp=None, arc_years=None, verbose=False):
    """Return `n` random asteroid designations from MPCORB matching the optional
    element filters. Each filter is a (min, max) tuple; either bound may be None.

      a    semimajor axis (au)        incl inclination (deg)
      e    eccentricity               q    perihelion distance a*(1-e) (au)
      H    absolute magnitude         nopp number of oppositions observed
      arc_years  observed-arc length in years

    For VM6-like multi-revolution two-opposition objects (the hard linkage case), sample
    faint + few oppositions + long arc, e.g. H=(17, None), nopp=(2, 2), arc_years=(8, None).

    Designations are the human-readable form (e.g. '51661', '2021 LM1') ready to pass
    to obs_sources.fetch_observations. Deterministic for a given `seed`."""
    from pyroorb import obs_sources
    build_index(verbose=verbose)
    z = np.load(MPCORB_NPZ, allow_pickle=False)
    av, ev, iv, Hv = z["a"], z["e"], z["incl"], z["H"]
    npv, arcv = z["nopp"].astype(float), z["arc_years"]
    qv = av * (1.0 - ev)
    mask = (_between(av, a) & _between(ev, e) & _between(iv, incl)
            & _between(qv, q) & _between(Hv, H)
            & _between(npv, nopp) & _between(arcv, arc_years))
    idx = np.flatnonzero(mask)
    if len(idx) < n:
        raise ValueError(f"only {len(idx)} asteroids match the filters (< {n})")
    rng = np.random.RandomState(seed)
    pick = rng.choice(idx, size=n, replace=False)
    packed = z["packed"]
    return [obs_sources._unpack(packed[i].strip()) for i in pick]


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for d in sample_asteroids(n, seed=0, verbose=True):
        print(d)
