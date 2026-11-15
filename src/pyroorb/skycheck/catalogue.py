"""The sky check's catalogue: every orbit the MPC publishes, in one memory-mapped set of columns.

Sources, refreshed when more than a day old (a failed download keeps the last good copy):

* ``MPCORB.DAT.gz`` -- every asteroid with an orbit (``pyroorb.mpcorb`` downloads and caches it);
* ``CometEls.txt`` -- every comet, parabolic and hyperbolic ones included, osculating at a
  recent date.

Each orbit is stored in perihelion form (q, e, i, node, argument of perihelion, perihelion time
T) beside its osculation epoch, so one conic solver serves asteroids and comets
(:mod:`pyroorb.skycheck.conic`), and with the unit vectors P and Q of its orbit plane already
rotated to the equatorial frame (computing them costs as much as a whole propagation). Columns
are float64 ``.npy`` files read memory-mapped: a query touches every row's elements once and
only its candidates' other columns, and the operating system may drop pages under pressure.
The whole set is ~260 MB on disk for 1.6 M orbits.

Times are TT Julian dates (the MPC's element time scale).
"""
import gzip
import json
import math
import os
import shutil
import threading
import time
import urllib.request
from array import array
from pathlib import Path

import numpy as np

from pyroorb import datadirs, mpcorb
from pyroorb.skycheck import conic

COMETELS_URL = "https://www.minorplanetcenter.net/iau/MPCORB/CometEls.txt"
MAX_AGE_S = 86400.0              # sources older than a day are refreshed
KIND_ASTEROID, KIND_COMET = 0, 1
FLOAT_COLS = ("q", "e", "i", "node", "peri", "T", "epoch", "H", "G", "U", "nopp")
DESIG_LEN, NAME_LEN = 24, 48


def catalogue_dir():
    """Where the built columns live (the per-user cache: rebuilt from the sources at will)."""
    return datadirs.cache_dir() / "skycheck_catalogue"


def comets_path():
    return datadirs.cache_dir() / "CometEls.txt"


# --------------------------------------------------------------------------- parsing
def _f(s):
    try:
        return float(s)
    except ValueError:
        return math.nan


def _jd(y, mo, d):
    """Julian date of calendar date y-mo-d (d may carry a fraction), proleptic Gregorian."""
    a = (14 - mo) // 12
    yy, mm = y + 4800 - a, mo + 12 * a - 3
    return (int(d) + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400
            - 32045 - 0.5 + (d - int(d)))


def unpack_epoch(s):
    """Packed MPC date ('K26BE' = 2026 Nov 14) -> JD at 0h. None when unreadable."""
    s = (s or "").strip()
    cent = {"I": 1800, "J": 1900, "K": 2000}.get(s[:1])
    if len(s) != 5 or cent is None:
        return None

    def code(c):
        return int(c) if c.isdigit() else ord(c) - 55
    try:
        return _jd(cent + int(s[1:3]), code(s[3]), code(s[4]))
    except ValueError:
        return None


def readable(field, packed):
    """MPCORB's readable designation column -> (designation, display name):
    '(433) Eros' -> ('433', '(433) Eros'); '2021 CA16' -> ('2021 CA16', '2021 CA16')."""
    s = field.strip()
    if s.startswith("(") and ")" in s:
        return s[1:s.index(")")], s
    return (s or packed), (s or packed)


def parse_mpcorb_line(ln):
    """One MPCORB.DAT record -> (designation, name, columns) or None (not an orbit line).

    The MPCORB layout is the MPC's export format (columns per MPOrbitFormat.html): H 9-13,
    G 15-19, packed epoch 21-25, M 27-35, peri 38-46, node 49-57, i 60-68, e 71-79, a 93-103,
    U 106, oppositions 124-126, readable designation 167-194."""
    if len(ln) < 160 or not ln[0:7].strip():
        return None
    epoch = unpack_epoch(ln[20:25])
    a, e = _f(ln[92:103]), _f(ln[70:79])
    M = _f(ln[26:35])
    if epoch is None or not (a > 0) or not (0 <= e < 1) or not math.isfinite(M):
        return None
    desig, name = readable(ln[166:194], ln[0:7].strip())
    u = ln[105:106]
    cols = {"q": a * (1 - e), "e": e, "i": _f(ln[59:68]), "node": _f(ln[48:57]),
            "peri": _f(ln[37:46]),
            "T": float(conic.perihelion_time(a, e, M, epoch)), "epoch": epoch,
            "H": _f(ln[8:13]), "G": _f(ln[14:19]),
            "U": float(u) if u.isdigit() else math.nan, "nopp": _f(ln[123:126])}
    return desig, name, cols


def parse_comet_line(ln):
    """One CometEls.txt record -> (designation, name, columns) or None.

    Columns (the MPC's comet export format): perihelion time 15-29 (yyyy mm dd.dddd TT), q 31-39,
    e 42-49, peri 52-59, node 62-69, i 72-79, osculation epoch 82-89 (yyyymmdd), H 92-95,
    G 97-100 (the total-magnitude law's slope, m = H + 5 log D + 2.5 G log r), name 103-158."""
    if len(ln) < 100:
        return None
    q, e = _f(ln[30:39]), _f(ln[41:49])
    try:
        T = _jd(int(ln[14:18]), int(ln[19:21]), float(ln[22:29]))
    except ValueError:
        return None
    ep = ln[81:89].strip()
    epoch = _jd(int(ep[:4]), int(ep[4:6]), int(ep[6:8])) if ep.isdigit() and len(ep) == 8 else T
    if not (q > 0) or not (e >= 0):
        return None
    name = ln[102:158].strip()
    desig = name.split(" (")[0].strip() or ln[0:12].strip()
    cols = {"q": q, "e": e, "i": _f(ln[71:79]), "node": _f(ln[61:69]), "peri": _f(ln[51:59]),
            "T": T, "epoch": epoch, "H": _f(ln[91:95]), "G": _f(ln[96:100]),
            "U": math.nan, "nopp": math.nan}
    return desig, name or desig, cols


# --------------------------------------------------------------------------- sources
def _stale(path, max_age_s):
    return not path.exists() or time.time() - path.stat().st_mtime > max_age_s


def refresh_sources(max_age_s=MAX_AGE_S, verbose=False):
    """Download MPCORB and the comet elements when older than ``max_age_s``. A failed download
    keeps the previous copy (a day-old catalogue beats none); with no copy at all it raises."""
    gz = mpcorb.MPCORB_GZ
    if _stale(gz, max_age_s):
        try:
            mpcorb.download_gz_atomic(mpcorb.MPCORB_URL, gz, verbose=verbose)
        except Exception as ex:                                     # noqa: BLE001
            if not gz.exists():
                raise
            if verbose:
                print(f"MPCORB refresh failed ({ex}); keeping the copy of "
                      f"{time.ctime(gz.stat().st_mtime)}", flush=True)
    cp = comets_path()
    if _stale(cp, max_age_s):
        try:
            with urllib.request.urlopen(COMETELS_URL, timeout=120) as r:
                txt = r.read().decode("utf-8", "replace")
            if txt.count("\n") > 100:
                tmp = cp.with_name(f"{cp.name}.{os.getpid()}.{threading.get_ident()}.part")
                tmp.write_text(txt, encoding="utf-8")
                os.replace(tmp, cp)
        except Exception as ex:                                     # noqa: BLE001
            if verbose:
                print(f"comet elements refresh failed ({ex})", flush=True)


# --------------------------------------------------------------------------- build
REOSCULATE_DAYS = 30.0      # an asteroid epoch further than this from the main one is moved
_REOSCULATE_CHUNK = 1500


def reosculate(cols, P, Q, asteroid, verbose=False):
    """Carry the asteroids whose elements are osculating far from the catalogue's main epoch
    to that epoch, by numerical integration (ASSIST: DE440 planets, Moon, GR), in place.

    MPCORB gives almost every orbit at one standard epoch, but ~0.1% (objects not observed for
    years, mostly) keep an epoch decades old; two-body from there misplaces them by degrees,
    which no quick-pass margin could cover. Integrated once here, every asteroid shares one
    epoch and one margin rule. Returns (number moved, main epoch). Without the ephemeris the
    orbits stay as published and the count is 0."""
    ep = cols["epoch"]
    if not asteroid.any():
        return 0, None
    vals, counts = np.unique(ep[asteroid], return_counts=True)
    main = float(vals[np.argmax(counts)])
    move = np.flatnonzero(asteroid & (np.abs(ep - main) > REOSCULATE_DAYS))
    if not len(move):
        return 0, main
    try:
        from pyroorb import force_assist
        moved = 0
        for epv in np.unique(ep[move]):
            sel = move[ep[move] == epv]
            for s in range(0, len(sel), _REOSCULATE_CHUNK):
                k = sel[s:s + _REOSCULATE_CHUNK]
                pos, vel = conic.state_at(cols["q"][k], cols["e"][k], cols["T"][k],
                                          P[k], Q[k], epv)
                out = force_assist.helio_states_multi(float(epv), list(np.hstack([pos, vel])),
                                                      [main], asteroids=False)[0]
                q, e, T, P2, Q2 = conic.elliptic_elements(out[:, :3], out[:, 3:], main)
                ok = np.isfinite(q) & np.isfinite(T)
                k, P2, Q2 = k[ok], P2[ok], Q2[ok]
                cols["q"][k], cols["e"][k], cols["T"][k] = q[ok], e[ok], T[ok]
                cols["epoch"][k] = main
                P[k], Q[k] = P2, Q2
                cols["i"][k], cols["node"][k], cols["peri"][k] = conic.angles(P2, Q2)
                moved += int(ok.sum())
    except Exception as ex:                                         # noqa: BLE001
        if verbose:
            print(f"sky-check catalogue: old-epoch orbits left as published "
                  f"({type(ex).__name__}: {ex})", flush=True)
        return 0, main
    if verbose:
        print(f"sky-check catalogue: {moved:,} orbits carried to the main epoch", flush=True)
    return moved, main


# Objects that pass close to Earth: two-body is meaningless across an encounter (a 0.01 au
# pass bends the orbit by degrees), and the quick pass's margin is measured on objects with
# no such pass. So every orbit whose two-body path comes within NEAR_AU of the Earth in the
# window is integrated once at build time and its daily states stored; a query in the window
# interpolates those instead. Over the measured window ~8,300 of 1.58 M orbits qualify.
NEAR_AU = 0.2
NEAR_Q_MAX = 1.3                 # perihelion beyond this cannot come within NEAR_AU of Earth
NEAR_BEFORE, NEAR_AFTER = 400.0, 60.0   # window around the build date (days)


def near_earth_scan(cols, P, Q, days, verbose=False):
    """Indices of the orbits whose two-body path comes within NEAR_AU of the Earth on any of
    ``days`` (TT JDs, daily)."""
    from pyroorb import obs_sources
    cand = np.flatnonzero(cols["q"] < NEAR_Q_MAX)
    if not len(cand):
        return cand
    earth = obs_sources._geocentric_positions(np.asarray(days, float))
    q, e, T = cols["q"][cand], cols["e"][cand], cols["T"][cand]
    Pc, Qc = P[cand], Q[cand]
    dmin = np.full(len(cand), np.inf)
    for j, d in enumerate(days):
        p, _ = conic.state_at(q, e, T, Pc, Qc, d)
        dmin = np.minimum(dmin, np.linalg.norm(p - earth[j], axis=1))
    return cand[dmin < NEAR_AU]


def near_earth_states(cols, P, Q, idx, days, verbose=False):
    """ASSIST heliocentric equatorial states (len(idx), len(days), 6) of the orbits ``idx`` on
    ``days``, integrated from each orbit's epoch."""
    from pyroorb import force_assist
    out = np.full((len(idx), len(days), 6), np.nan)
    ep = cols["epoch"][idx]
    for epv in np.unique(ep):
        rows = np.flatnonzero(ep == epv)
        for s in range(0, len(rows), _REOSCULATE_CHUNK):
            r = rows[s:s + _REOSCULATE_CHUNK]
            k = idx[r]
            pos, vel = conic.state_at(cols["q"][k], cols["e"][k], cols["T"][k], P[k], Q[k], epv)
            st = force_assist.helio_states_multi(float(epv), list(np.hstack([pos, vel])),
                                                 np.asarray(days, float), asteroids=False)
            out[r] = np.transpose(st, (1, 0, 2))
    return out


def build(verbose=False, root=None):
    """Parse the sources into a NEW build directory under ``root`` and return its path.

    Every build is its own directory (``b<unix time>``) and a reader takes the newest complete
    one, so a rebuild never touches files another reader has memory-mapped (on Windows a
    mapped file cannot be renamed or deleted); older builds are removed once nothing holds
    them."""
    root = Path(root) if root is not None else catalogue_dir()
    desigs, names, kinds = bytearray(), bytearray(), bytearray()
    cols = {c: array("d") for c in FLOAT_COLS}

    def add(got, kind):
        d, n, row = got
        desigs.extend(d.encode("ascii", "replace")[:DESIG_LEN].ljust(DESIG_LEN, b"\0"))
        names.extend(n.encode("ascii", "replace")[:NAME_LEN].ljust(NAME_LEN, b"\0"))
        kinds.append(kind)
        for c in FLOAT_COLS:
            cols[c].append(row[c])

    gz, cp = mpcorb.MPCORB_GZ, comets_path()
    sources = {"mpcorb_mtime": gz.stat().st_mtime,
               "comets_mtime": cp.stat().st_mtime if cp.exists() else None}
    with gzip.open(gz, "rt", encoding="utf-8", errors="replace") as f:
        started = False
        for ln in f:
            if not started:
                started = ln.startswith("-----")
                continue
            got = parse_mpcorb_line(ln)
            if got:
                add(got, KIND_ASTEROID)
    if cp.exists():
        for ln in cp.read_text(encoding="utf-8", errors="replace").splitlines():
            got = parse_comet_line(ln)
            if got:
                add(got, KIND_COMET)
    n = len(kinds)
    stamp = time.time()
    out = root / f"b{int(stamp * 1000)}"
    tmp = root / f".{out.name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    arrays = {c: np.frombuffer(v, dtype=np.float64).copy() for c, v in cols.items()}
    P, Q = conic.orientation(arrays["i"], arrays["node"], arrays["peri"])
    kind_arr = np.frombuffer(bytes(kinds), dtype=np.uint8)
    n_moved, main_epoch = reosculate(arrays, P, Q, kind_arr == KIND_ASTEROID, verbose=verbose)
    for c, v in arrays.items():
        np.save(tmp / f"{c}.npy", v)
    np.save(tmp / "P.npy", P)
    np.save(tmp / "Q.npy", Q)
    np.save(tmp / "kind.npy", np.frombuffer(bytes(kinds), dtype=np.uint8))
    np.save(tmp / "desig.npy", np.frombuffer(bytes(desigs), dtype=f"S{DESIG_LEN}"))
    np.save(tmp / "name.npy", np.frombuffer(bytes(names), dtype=f"S{NAME_LEN}"))
    for p in tmp.glob("*.npy"):                                     # verify before publishing
        if len(np.load(p, mmap_mode="r")) != n:
            raise OSError(f"catalogue column {p.name} came out short")
    near_t0 = math.floor(stamp / 86400.0 + 2440587.5) - NEAR_BEFORE + 0.5   # 0h, TT ~ UTC
    days = near_t0 + np.arange(0.0, NEAR_BEFORE + NEAR_AFTER + 1.0)
    try:
        near = near_earth_scan(arrays, P, Q, days, verbose=verbose)
        states = near_earth_states(arrays, P, Q, near, days, verbose=verbose)
        good = np.isfinite(states).all(axis=(1, 2))
        near, states = near[good], states[good]
    except Exception as ex:                                         # noqa: BLE001
        if verbose:
            print(f"sky-check catalogue: no near-Earth ephemeris ({type(ex).__name__}: {ex})",
                  flush=True)
        near, states, near_t0 = np.zeros(0, np.int64), np.zeros((0, len(days), 6)), None
    np.save(tmp / "near_index.npy", near.astype(np.int64))
    np.save(tmp / "near_states.npy", states)
    if verbose:
        print(f"sky-check catalogue: {len(near):,} near-Earth orbits integrated over "
              f"{len(days)} days", flush=True)
    meta = {"rows": n, "built": stamp, "main_epoch": main_epoch, "reosculated": n_moved,
            "near_t0": near_t0, "near_days": int(len(days)), **sources}
    (tmp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")   # last: complete
    tmp.rename(out)
    if verbose:
        print(f"sky-check catalogue: {n:,} orbits -> {out}", flush=True)
    return out


# --------------------------------------------------------------------------- load
class Catalogue:
    """Column arrays (``cols[name]``), P and Q (n, 3), ``kind``, designations and names, and
    the near-Earth ephemeris: ``near_index`` (catalogue rows) with ``near_states`` (rows, days,
    6), daily from ``meta["near_t0"]``."""

    def __init__(self, cols, P, Q, kind, desig, name, meta=None, near_index=None,
                 near_states=None):
        self.cols, self.P, self.Q, self.kind = cols, P, Q, kind
        self.desig, self.name = desig, name
        self.meta = meta or {}
        self.near_index = np.zeros(0, np.int64) if near_index is None else near_index
        self.near_states = np.zeros((0, 0, 6)) if near_states is None else near_states

    def __len__(self):
        return len(self.kind)

    def designation(self, k):
        return self.desig[k].decode("ascii", "replace").rstrip("\0").strip()

    def display_name(self, k):
        return self.name[k].decode("ascii", "replace").rstrip("\0").strip()

    @classmethod
    def from_rows(cls, rows):
        """A small in-memory catalogue from ``[(designation, name, columns, kind)]`` (columns as
        :func:`parse_mpcorb_line` gives them). Used for NEOCP orbits and by tests."""
        cols = {c: np.array([r[2].get(c, math.nan) for r in rows], float) for c in FLOAT_COLS}
        P, Q = conic.orientation(cols["i"], cols["node"], cols["peri"])
        return cls(cols, P, Q, np.array([r[3] for r in rows], np.uint8),
                   np.array([r[0].encode("ascii", "replace")[:DESIG_LEN] for r in rows],
                            dtype=f"S{DESIG_LEN}"),
                   np.array([r[1].encode("ascii", "replace")[:NAME_LEN] for r in rows],
                            dtype=f"S{NAME_LEN}"))


_LOCK = threading.Lock()            # guards the cached Catalogue
_BUILD_LOCK = threading.Lock()      # one download / build at a time
_CACHE = {"cat": None, "path": None}
NEED = FLOAT_COLS + ("P", "Q", "kind", "desig", "name", "near_index", "near_states")


def _complete(d):
    return (d / "meta.json").exists() and all((d / f"{c}.npy").exists() for c in NEED)


def _meta(d):
    try:
        return json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def current_build(root=None):
    """The newest complete build directory, or None."""
    root = Path(root) if root is not None else catalogue_dir()
    if not root.is_dir():
        return None
    builds = sorted((d for d in root.iterdir() if d.is_dir() and d.name.startswith("b")
                     and _complete(d)), key=lambda d: d.name)
    return builds[-1] if builds else None


def _out_of_date(d):
    """The build predates the sources it was made from."""
    m = _meta(d)
    gz, cp = mpcorb.MPCORB_GZ, comets_path()
    return ((gz.exists() and gz.stat().st_mtime > (m.get("mpcorb_mtime") or 0) + 1)
            or (cp.exists() and cp.stat().st_mtime > (m.get("comets_mtime") or 0) + 1))


def _prune(root, keep):
    """Remove builds other than ``keep``; one still mapped somewhere stays until next time."""
    for d in root.iterdir():
        if d.is_dir() and d != keep and (d.name.startswith("b") or d.name.endswith(".tmp")):
            if d.name.endswith(".tmp") and time.time() - d.stat().st_mtime < 3600:
                continue                                            # a build in progress
            shutil.rmtree(d, ignore_errors=True)


def update(max_age_s=MAX_AGE_S, verbose=False):
    """Download stale sources and build when the newest build predates them. Returns the build
    in use afterwards. Safe to call from several threads (one does the work)."""
    root = catalogue_dir()
    with _BUILD_LOCK:
        refresh_sources(max_age_s=max_age_s, verbose=verbose)
        cur = current_build(root)
        if cur is None or _out_of_date(cur):
            cur = build(verbose=verbose, root=root)
        return cur


def load(refresh=True, max_age_s=MAX_AGE_S, verbose=False):
    """The catalogue, memory-mapped and cached per process.

    With ``refresh`` the sources are downloaded when older than ``max_age_s`` and a new build is
    made when the newest predates them (the first build downloads ~90 MB and parses for about a
    minute). Without it the newest existing build is used as it is, and one is made only if
    none exists -- the web server's queries, which never wait on a download."""
    root = catalogue_dir()
    if refresh:
        cur = update(max_age_s=max_age_s, verbose=verbose)
    else:
        cur = current_build(root)
        if cur is None:
            with _BUILD_LOCK:
                cur = current_build(root)
                if cur is None:
                    if not mpcorb.MPCORB_GZ.exists():
                        mpcorb.ensure_mpcorb(verbose=verbose)
                    cur = build(verbose=verbose, root=root)
    with _LOCK:
        if _CACHE["cat"] is None or _CACHE["path"] != cur:
            ld = lambda c: np.load(cur / f"{c}.npy", mmap_mode="r")
            _CACHE["cat"] = Catalogue({c: ld(c) for c in FLOAT_COLS}, ld("P"), ld("Q"),
                                      ld("kind"), ld("desig"), ld("name"), _meta(cur),
                                      ld("near_index"), ld("near_states"))
            _CACHE["path"] = cur
            _prune(root, cur)
        return _CACHE["cat"]


def refresh_in_background(max_age_s=MAX_AGE_S):
    """Bring the sources and the build up to date on a daemon thread (the web server's periodic
    refresh); queries keep using the loaded catalogue meanwhile."""
    th = threading.Thread(target=_quiet_update, args=(max_age_s,), name="skycheck-catalogue",
                          daemon=True)
    th.start()
    return th


def _quiet_update(max_age_s):
    try:
        update(max_age_s=max_age_s)
    except Exception as ex:                                         # noqa: BLE001
        print(f"sky-check catalogue refresh failed: {type(ex).__name__}: {ex}", flush=True)
