"""Asteroid perturbers beyond the 16: the field an orbit is integrated in, and the screen that
decides which extra bodies an orbit needs.

The 16 massive asteroids of JPL's sb441-n16 kernel leave out bodies that measurably deflect
individual orbits: (19) Fortuna passed 2016 RM33 = 2005 SA198 at 0.0011 au in 2011, and without
it the fit rejects the whole 2005 apparition (6.3" off); with it all 15 observations fit.

Field model. ``asteroids`` on a force backend is False (planets only), True (the 16), or a
:class:`Field` -- the 16 plus a set of extra bodies. A Field is truthy, so every existing
"perturbers on?" test keeps working, and it rides through ``forcemodel.PinnedForce`` into every
product of the fit.

Masses. The 16 and the weakly determined bodies use the DE440 header constants (MA####, which
ASSIST joins by number); 165 others use SiMDA's compiled mass (Kretlow 2020, EPSC2020-690;
data/perturber_masses.json, rank A-C) -- see mass_table() for the measurements behind that.

Kernel. The extra bodies come from ONE Sun-centred SPK built from JPL's sb441-n373 (DE441
small-body perturbers),
the 16 copied verbatim over 1800-2200, the other 327 re-fitted within 20 km over 1900-2100 (a
20 km position error moves a 0.001 au encounter's force by 1e-4; refits with 5, 20 and 50 km
kernels agreed to 1e-4" and 1.5 m in state). A field with extra bodies is bounded by 1900-2100;
a sweep outside it falls back to the 16 (sb441-n16, or the kernel's own 16 within 1800-2200),
else planets-only, and says so once (force_assist._fallback_ephem).

Screen. For each candidate body the linear first-order change it makes to the observations is
built from ONE state-transition integration and the fit's own weighted design matrix:
``dchi2_j = ||(I - P) W^1/2 s_j||^2`` -- the chi2 the observations would gain if body j were
added, after the orbit absorbs what it can. It is scored on ALL observations, rejected ones
included: outlier rejection is exactly what hides a deflection (on the included observations
alone Fortuna scores 15; on all of them 1159, and the refit gains 1282). A body joins the field
at dchi2 >= THRESHOLD; it adds no free parameter (its mass is fixed), so it cannot overfit.
"""
import atexit
import os
import threading
from pathlib import Path

import numpy as np

KERNEL_NAME = "pyroorb-perturbers-1900-2100-20km.bsp"
KERNEL_ENV = "PYROORB_PERTURBER_KERNEL"
SWITCH_ENV = "PYROORB_PERTURBERS"            # "off" / "0" / "no" disables the automatic screen
BIG16 = (1, 2, 3, 4, 7, 10, 15, 16, 31, 52, 65, 87, 88, 107, 511, 704)
# A body joins at dchi2 >= 1: it costs ~2% of a fit and cannot overfit; on 100 random
# main-belt orbits this flags ~1%, and every threshold from 1 to ~25 gave the same observation
# sets on everything measured, so the low end keeps the small genuine gains too.
THRESHOLD = 1.0
MAX_EXTRA = 8
AU_KM = 149597870.7
ARC = 206264.806
# Coarse stage: every body on a 4-day grid from a memory-mapped float32 position table.
STEP = 4.0
# Fine stage for bodies that pass close (a 10 km/s pass moves 0.023 au in 4 days and can hide
# between coarse nodes) or score above FINE_DCHI2: exact kernel positions on a 0.25-day grid.
FINE_DCHI2 = 0.01
FINE_DIST = 0.03
FINE_WIN = 8.0
FINE_STEP = 0.25
# A pass closer than CLOSE_B lasts hours (0.00006 au at 5 km/s: about an hour), shorter than
# the fine step; its kick is taken analytically (straight-line small-deflection impulse
# 2GM/(b v) toward the perturber) instead of sampled -- sampling under-predicted (310585)'s
# 0.00006 au Bamberga pass 300-fold.
CLOSE_B = 0.002


class Field(frozenset):
    """The 16 asteroid perturbers plus these extra bodies (MP numbers). Always truthy."""

    def __bool__(self):
        return True

    def __repr__(self):
        return f"Field({sorted(self)})"

    def label(self):
        """'+(19)+(29)' -- the extra bodies in numbered-asteroid style, for the force-model tag."""
        return "".join(f"+({n})" for n in sorted(self))


def as_field(value):
    """Normalise an ``asteroids`` value: None stays None (the caller's default), a Field or any
    set/list of extra bodies becomes a Field, anything else a bool."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None if value is None else bool(value)
    if isinstance(value, (set, frozenset, list, tuple)):
        # no extra bodies is just the 16
        return Field(int(n) for n in value) if value else True
    return bool(value)


def extras(value):
    """The extra bodies of an ``asteroids`` value (empty for False / True / None)."""
    return frozenset(value) if isinstance(value, Field) else frozenset()


_ALL = {}


def all_extras():
    """Every body of the extra-perturber kernel outside the 16 (empty without the kernel)."""
    kern = kernel_path()
    if kern is None:
        return frozenset()
    if kern not in _ALL:
        from jplephem.spk import SPK
        k = SPK.open(kern)
        try:
            _ALL[kern] = frozenset(s.target - 2000000 for s in k.segments) - frozenset(BIG16)
        finally:
            k.close()
    return _ALL[kern]


def all_field(exclude=None):
    """The whole kernel as a Field: the 16 plus every other body, without `exclude` (the
    fitted object's own number -- a body is never integrated inside itself); None without the
    kernel."""
    a = all_extras() - {exclude}
    return Field(a) if a else None


def is_all(value):
    """True for a Field holding every kernel body outside the 16 (the object itself aside)."""
    if not isinstance(value, Field):
        return False
    a = all_extras()
    return bool(a) and frozenset(value) <= a and len(a - frozenset(value)) <= 1


def enabled():
    """False when the automatic screen is switched off (PYROORB_PERTURBERS=off)."""
    return os.environ.get(SWITCH_ENV, "").strip().lower() not in ("0", "off", "no", "false")


# The kernel ships inside the package (~26 MB), so every install screens perturbers with no
# download or build step.
BUNDLED_KERNEL = Path(__file__).resolve().parent / "data" / KERNEL_NAME


def kernel_path():
    """The extra-perturber kernel: PYROORB_PERTURBER_KERNEL when set, else the copy bundled
    with the package, else <ephem dir>/KERNEL_NAME (a rebuild, ``perturber_kernel.build``);
    None when none exists."""
    want = os.environ.get(KERNEL_ENV, "").strip()
    if want:
        return want if Path(want).exists() else None
    if BUNDLED_KERNEL.exists():
        return str(BUNDLED_KERNEL)
    from pyroorb import force_assist as fa
    p = Path(fa.EPHEM_DIR) / KERNEL_NAME
    return str(p) if p.exists() else None


# ---------------------------------------------------------------------------------- ephemerides
# One assist.Ephem per body set, its SPK target table compacted to exactly those bodies (ASSIST's
# force loop runs over spk_asteroids->num, so a compacted table costs what its bodies cost).
# Each Ephem memory-maps the files: ~1 MB resident and ~1 ms to build, so a set per fit is cheap
# and concurrent fits never share a mutable table.
_LOCK = threading.Lock()
_EPHEMS = {}
_WINDOW = {}          # id(ephem) -> (jd_lo, jd_hi) the compacted table covers
# Released before interpreter teardown: an assist.Ephem freed after ASSIST's C library has been
# unloaded raises in its __del__.
atexit.register(_EPHEMS.clear)


def ephem_for(bodies):
    """The cached Ephem integrating exactly `bodies` (MP numbers present in the kernel), or None
    when the kernel is missing or ASSIST's table layout cannot be verified."""
    bodies = frozenset(int(b) for b in bodies)
    e = _EPHEMS.get(bodies)
    if e is not None:
        return e
    kern = kernel_path()
    if kern is None:
        return None
    from pyroorb import force_assist as fa
    if not fa.spk_layout_supported():
        return None
    import assist
    import ctypes
    with _LOCK:
        e = _EPHEMS.get(bodies)
        if e is not None:
            return e
        eph = assist.Ephem(fa.ensure_planets_kernel(), kern)
        spk = fa._SpkS.from_address(eph.spk_asteroids)
        n = spk.num
        size = ctypes.sizeof(fa._SpkTarget)
        raw = [bytes((ctypes.c_char * size).from_address(ctypes.addressof(spk.targets[i])))
               for i in range(n)]
        codes = [fa._SpkTarget.from_buffer_copy(b).code for b in raw]
        if not all(2000000 < c < 3000000 for c in codes):
            return None
        keep = [i for i in range(n) if codes[i] - 2000000 in bodies]
        order = keep + [i for i in range(n) if i not in set(keep)]
        for j, i in enumerate(order):
            ctypes.memmove(ctypes.addressof(spk.targets[j]), raw[i], size)
        spk.num = len(keep)
        tg = [spk.targets[j] for j in range(len(keep))]
        # the field's masses: SiMDA where mass_table() covers the body (see there), else DE440
        table = mass_table()["masses"]
        for t in tg:
            e = table.get(t.code - 2000000)
            if e and t.code - 2000000 not in BIG16:
                t.mass = float(e["gm_au3_day2"])
        _WINDOW[id(eph)] = ((max(t.beg for t in tg), min(t.end for t in tg)) if tg
                            else (-np.inf, np.inf))
        _EPHEMS[bodies] = eph
        return eph


def window(ephem):
    """(jd_lo, jd_hi) a compacted Ephem covers; None for any other Ephem."""
    return _WINDOW.get(id(ephem))


def field_bodies(value, self_number=None):
    """Every body a Field integrates: the 16 plus its extras, minus the object itself."""
    out = set(BIG16) | set(extras(value))
    out.discard(self_number)
    return frozenset(out)


# ---------------------------------------------------------------------------------- screen
class _Table:
    """Positions of every kernel body on a fixed 4-day grid, float32, memory-mapped."""

    def __init__(self, kernel):
        from jplephem.spk import SPK
        self.kern = SPK.open(kernel)
        self.seg = {s.target - 2000000: s for s in self.kern.segments}
        self.bodies = sorted(self.seg)
        self.row = {b: i for i, b in enumerate(self.bodies)}
        lo = max(s.start_jd for s in self.kern.segments)
        hi = min(s.end_jd for s in self.kern.segments)
        self.t0 = float(np.ceil(lo + 1.0))
        nt = int((hi - 1.0 - self.t0) // STEP)
        self.t_hi = self.t0 + STEP * (nt - 1)
        path = self._path(kernel)
        try:
            self.pos = np.load(path, mmap_mode="r")
            if self.pos.shape != (len(self.bodies), nt, 3):
                raise ValueError("stale table")
        except Exception:
            t = self.t0 + STEP * np.arange(nt)
            arr = np.empty((len(self.bodies), nt, 3), np.float32)
            for i, b in enumerate(self.bodies):
                arr[i] = (np.asarray(self.seg[b].compute(t)).T / AU_KM).astype(np.float32)
            # one temporary name per writer: processes that build the table at once (a pool's
            # first fits after a kernel change) each moved the other's file away, and the
            # loser's screen failed with FileNotFoundError. os.replace is atomic, so the last
            # complete table wins and every reader maps a whole one.
            tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npy")
            np.save(tmp, arr)
            os.replace(tmp, path)
            self.pos = np.load(path, mmap_mode="r")

    @staticmethod
    def _path(kernel):
        p = Path(kernel)
        st = p.stat()
        name = f"{p.stem}.table4d.{st.st_size}.npy"
        # beside the kernel when that folder is writable, never inside the package (the 75 MB
        # table would land in an editable checkout and in every wheel built from it)
        if p.resolve().parent != BUNDLED_KERNEL.parent and os.access(p.parent, os.W_OK):
            return p.parent / name
        from pyroorb import datadirs
        d = datadirs.cache_dir() / "perturbers"
        d.mkdir(parents=True, exist_ok=True)
        return d / name


_TABLES = {}


def _close_tables():
    for t in _TABLES.values():
        try:
            t.kern.close()
        except Exception:
            pass
    _TABLES.clear()


atexit.register(_close_tables)


def _table():
    kern = kernel_path()
    if kern is None:
        return None
    t = _TABLES.get(kern)
    if t is None:
        with _LOCK:
            t = _TABLES.get(kern)
            if t is None:
                t = _TABLES[kern] = _Table(kern)
    return t


_GM = {}
_MASSES = {}


def mass_table():
    """The bundled SiMDA mass table (data/perturber_masses.json): {"masses": {number: {"kg",
    "kg_err", "rank", "gm_au3_day2"}}, "source", "citation", "snapshot", "policy"}.

    Why: the DE440 header masses come from the planetary-ephemeris fit, which constrains most of
    the 343 only weakly. SiMDA (Kretlow 2020) combines the published determinations -- close
    encounters, binaries, spacecraft -- per body. Where an encounter constrains the mass strongly
    its mass fitted better in 4 of 5 deflections measured ((17) Thetis by (11) Parthenope: chi2
    -85; (310585) by (324) Bamberga: -20; (311771) by (39) Laetitia, whose DE440 mass made the fit
    worse than no Laetitia at all: -11; (987) by (29) Amphitrite: -11). Used for the bodies outside
    the 16 with SiMDA rank A-C; the 16 keep DE440, so a +16 fit is unchanged and agrees with
    sb441-n16, and ranks D-E (weakly determined) keep DE440 too."""
    if not _MASSES:
        import json
        p = Path(__file__).with_name("data") / "perturber_masses.json"
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
        except OSError:
            doc = {"masses": {}}
        doc["masses"] = {int(k): v for k, v in doc.get("masses", {}).items()}
        _MASSES.update(doc)
    return _MASSES


def mass_source(body):
    """'SiMDA (rank A)' or 'DE440' -- where the field's mass for this body comes from."""
    e = mass_table()["masses"].get(int(body))
    return f"SiMDA (rank {e['rank']})" if e and int(body) not in BIG16 else "DE440"


def gm(body):
    """GM (au^3/day^2) a kernel body perturbs with: SiMDA's for the bodies mass_table() covers,
    else the DE440 header value ASSIST joins."""
    e = mass_table()["masses"].get(int(body))
    if e and int(body) not in BIG16:
        return float(e["gm_au3_day2"])
    if not _GM:
        import assist
        from pyroorb import force_assist as fa
        eph = assist.Ephem(fa.ensure_planets_kernel(), kernel_path())
        spk = fa._SpkS.from_address(eph.spk_asteroids)
        with _LOCK:
            _GM.update({spk.targets[i].code - 2000000: spk.targets[i].mass
                        for i in range(spk.num)})
    return _GM.get(int(body), 0.0)


def de440_gm(body):
    """The DE440 header GM of a kernel body, whatever mass the field uses."""
    gm(1)                                                  # fills _GM
    return _GM.get(int(body), 0.0)


_NAMES = {}


def name(number):
    """The asteroid's name ('Fortuna'), from the bundled table of the kernel's bodies; '' if none."""
    if not _NAMES:
        import json
        p = Path(__file__).with_name("data") / "perturber_names.json"
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)              # {"_about": credit, "names": {number: name}}
            _NAMES.update({int(k): v for k, v in doc.get("names", {}).items()})
        except OSError:
            _NAMES[0] = ""
    return _NAMES.get(int(number), "")


def _date(jd_tdb):
    """Calendar date (UTC to the day) of a TDB Julian date: 'YYYY-MM-DD'."""
    z = int(jd_tdb + 0.5)
    a = z + 32044
    b = (4 * a + 3) // 146097
    c = a - 146097 * b // 4
    d = (4 * c + 3) // 1461
    e = c - 1461 * d // 4
    m = (5 * e + 2) // 153
    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = 100 * b + d - 4800 + m // 10
    return f"{year:04d}-{month:02d}-{day:02d}"


def mass_kg(body):
    """The mass (kg) the field uses for a body: GM / G, G as the mass table states it."""
    return gm(body) * AU_KM ** 3 * 1e9 / 86400.0 ** 2 / float(mass_table().get("G", 6.67430e-11))


def describe(number, score):
    """One extra body as the fit reports it: {"number", "name", "dchi2", "closest_au",
    "closest_date", "mass_kg", "mass_source"}, from a Screen score (dchi2, closest approach au,
    its JD TDB)."""
    dchi2, dist, jd = score
    return {"number": int(number), "name": name(number), "dchi2": round(float(dchi2), 1),
            "closest_au": round(float(dist), 6), "closest_date": _date(jd),
            "mass_kg": float(f"{mass_kg(number):.3g}"), "mass_source": mass_source(number)}


def summary(details):
    """'(19) Fortuna: dchi2 1178, closest 0.0011 au on 2011-04-13, mass 8.88e+18 kg (SiMDA
    (rank A)); ...' for a list of describe() entries -- the one line every surface prints."""
    out = []
    for e in details:
        label = f"({e['number']}) {e.get('name') or ''}".strip()
        au = e["closest_au"]
        dist = f"{au:.2g}" if au < 0.001 else f"{au:.4f}"
        mass = (f", mass {e['mass_kg']:.3g} kg ({e['mass_source']})"
                if e.get("mass_kg") else "")
        out.append(f"{label}: dchi2 {e['dchi2']:g}, closest {dist} au on {e['closest_date']}{mass}")
    return "; ".join(out)


def self_number(fit):
    """MP number when the fitted object is itself a numbered asteroid ('(29)', '29 Amphitrite')."""
    import re
    for name in [getattr(fit, "designation", None)] + list(getattr(fit, "designations", None) or []):
        s = str(name or "").strip()
        m = re.fullmatch(r"\((\d+)\).*", s)
        if m:
            return int(m.group(1))
        m = re.fullmatch(r"(\d+)(?:\s+(\S.*))?", s)
        # '2016 RM33' is a provisional designation (year + half-month code), not a number
        if m and not (m.group(2) and re.fullmatch(r"[A-Z]{2}\d*", m.group(2))):
            return int(m.group(1))
    return None


class Screen:
    """Result of :func:`screen`: per-body (dchi2, closest approach au, its JD TDB), the picks
    (dchi2 >= THRESHOLD, largest first, at most MAX_EXTRA), and CPU seconds."""

    def __init__(self, scores, seconds):
        self.scores = scores
        self.seconds = seconds
        ranked = sorted(((v[0], b) for b, v in scores.items()), reverse=True)
        self.picks = [b for d, b in ranked if d >= THRESHOLD][:MAX_EXTRA]

    def top(self, n=5):
        return sorted(((b,) + tuple(v) for b, v in self.scores.items()), key=lambda r: -r[1])[:n]


def _nights(times, sra, sde, Robs):
    """Collapse observations to one normal point per UTC date (the signature of a perturber is
    smooth within a night): mean epoch, mean observer position, combined sigma per axis."""
    key = np.floor(times - 0.5).astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    w_ra, w_de = 1.0 / sra ** 2, 1.0 / sde ** 2
    cnt = np.bincount(inv)
    t = np.bincount(inv, weights=times) / cnt
    R = np.stack([np.bincount(inv, weights=Robs[:, k]) / cnt for k in range(3)], 1)
    s_ra = 1.0 / np.sqrt(np.bincount(inv, weights=w_ra))
    s_de = 1.0 / np.sqrt(np.bincount(inv, weights=w_de))
    return t, s_ra, s_de, R


def screen(fit, candidates=None, base=None):
    """Score candidate extra bodies for `fit` (see the module docstring). `base` is the field the
    state-transition matrix is integrated in (default: the fit's own); `candidates` default to
    every kernel body not already in it, minus the object itself. Returns a :class:`Screen`,
    or None when the kernel is missing or the arc leaves the kernel window."""
    import time
    from pyroorb import force_assist as fa
    from pyroorb.products import fit_asteroids
    c0 = time.process_time()
    tab = _table()
    ctx = getattr(fit, "_ctx", None)
    if tab is None or ctx is None:
        return None
    if base is None:
        base = fit_asteroids(fit)
    me = self_number(fit)
    in_field = field_bodies(base) if base else frozenset()
    if candidates is None:
        candidates = [b for b in tab.bodies if b not in in_field and b != me]
    candidates = [b for b in candidates if b in tab.row and b != me]
    t0 = float(ctx["epoch_tdb"])
    tobs = np.asarray(ctx["times_tdb"], float)
    if tobs.min() - 30 < tab.t0 or tobs.max() + 1 > tab.t_hi or not candidates:
        return None
    tn, sra, sde, Robs = _nights(tobs, np.asarray(ctx["sigma_ra"], float),
                                 np.asarray(ctx["sigma_dec"], float),
                                 np.asarray(ctx["Robs"], float))
    i_lo = int(np.floor((tn.min() - 30 - tab.t0) / STEP))
    i_hi = int(np.ceil((tn.max() + 1 - tab.t0) / STEP)) + 1
    tgrid = tab.t0 + STEP * np.arange(i_lo, i_hi)
    pos0, vel0 = fit.state_vector()
    X0 = np.concatenate([pos0, vel0])
    h = np.array([1e-7] * 3 + [1e-9] * 3)
    states = [X0] + [X0 + np.eye(6)[k] * h[k] for k in range(6)]
    backend = ctx["backend"]
    run = getattr(backend, "helio_states_multi", None) or fa.helio_states_multi

    def stm(times):
        out = run(t0, states, times, asteroids=base)
        nom = out[:, 0, :]
        return nom[:, :3], np.stack([(out[:, k + 1, :] - nom) / h[k] for k in range(6)], axis=2)

    times = np.unique(np.concatenate([tgrid, tn, [t0]]))
    r, Phi = stm(times)
    Phiinv = np.linalg.inv(Phi)
    io = np.searchsorted(times, tn)
    ig = np.searchsorted(times, tgrid)
    rho = r[io] - Robs
    d = np.linalg.norm(rho, axis=1)
    u = rho / d[:, None]
    ra, dec = np.arctan2(u[:, 1], u[:, 0]), np.arcsin(u[:, 2])
    eRA = np.stack([-np.sin(ra), np.cos(ra), 0 * ra], 1)
    eDE = np.stack([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)], 1)
    kra = ARC / (d * sra)
    kde = ARC / (d * sde)
    Po = Phi[io][:, :3, :]
    H = np.vstack([np.einsum("ij,ijk->ik", eRA, Po) * kra[:, None],
                   np.einsum("ij,ijk->ik", eDE, Po) * kde[:, None]])
    Q, _ = np.linalg.qr(H)

    def dchi2(cum_obs):                                  # (nobs, B, 6) -> (B,)
        dX = np.einsum("nij,nbj->nbi", Po, cum_obs)
        s = np.concatenate([np.einsum("nj,nbj->nb", eRA, dX) * kra[:, None],
                            np.einsum("nj,nbj->nb", eDE, dX) * kde[:, None]])
        e = s - Q @ (Q.T @ s)
        return (e * e).sum(0)

    rg = r[ig]
    Pg = Phiinv[ig][:, :, 3:]
    dtg = np.diff(tgrid)
    k_obs = np.clip(np.searchsorted(tgrid, tn) - 1, 0, len(tgrid) - 1)
    k0 = int(np.clip(np.searchsorted(tgrid, t0) - 1, 0, len(tgrid) - 1))
    gms = {b: gm(b) for b in candidates}
    scores, fine = {}, []
    for c in range(0, len(candidates), 32):
        bs = candidates[c:c + 32]
        rj = np.asarray(tab.pos[[tab.row[b] for b in bs]][:, i_lo:i_hi, :], float).transpose(1, 0, 2)
        dr = rj - rg[:, None, :]
        dist = np.linalg.norm(dr, axis=2)
        g_ = np.array([gms[b] for b in bs])
        acc = g_[None, :, None] * (dr / dist[:, :, None] ** 3
                                   - rj / np.linalg.norm(rj, axis=2)[:, :, None] ** 3)
        g = np.einsum("nij,nbj->nbi", Pg, acc)
        cum = np.concatenate([np.zeros((1,) + g.shape[1:]),
                              np.cumsum(0.5 * (g[1:] + g[:-1]) * dtg[:, None, None], 0)])
        cum -= cum[k0]
        dch = dchi2(cum[k_obs])
        dmin = dist.min(0)
        for j, b in enumerate(bs):
            scores[b] = (float(dch[j]), float(dmin[j]), float(tgrid[int(dist[:, j].argmin())]))
            if dch[j] > FINE_DCHI2 or dmin[j] < FINE_DIST:
                fine.append(b)
    if fine:
        extra = np.unique(np.concatenate(
            [np.arange(scores[b][2] - FINE_WIN, scores[b][2] + FINE_WIN, FINE_STEP) for b in fine]))
        tf = np.unique(np.concatenate([times, extra[(extra > times[0]) & (extra < times[-1])]]))
        rf, Phif = stm(tf)
        Pfi = np.linalg.inv(Phif)[:, :, 3:]
        iof = np.searchsorted(tf, tn)
        i0f = int(np.searchsorted(tf, t0))
        dtf = np.diff(tf)
        for b in fine:
            p, v = tab.seg[b].compute_and_differentiate(tf)
            rj = np.asarray(p).T / AU_KM
            vj = np.asarray(v).T / AU_KM
            dr = rj - rf
            dist = np.linalg.norm(dr, axis=1)
            acc = gms[b] * (dr / dist[:, None] ** 3 - rj / np.linalg.norm(rj, axis=1)[:, None] ** 3)
            jm = int(dist.argmin())
            impulse = None
            if dist[jm] < CLOSE_B and 0 < jm < len(tf) - 1:
                # straight-line closest approach from the nearest sample
                # relative velocity of the perturber w.r.t. the object; dr = r_j - r
                vr = vj[jm] - np.gradient(rf, tf, axis=0)[jm]
                tca = tf[jm] - float(dr[jm] @ vr) / float(vr @ vr)
                bvec = dr[jm] + vr * (tca - tf[jm])
                bb = float(np.linalg.norm(bvec))
                vv = float(np.linalg.norm(vr))
                half = max(1.0, 20.0 * bb / vv)
                inwin = np.abs(tf - tca) < half
                acc[inwin] = 0.0
                T = half
                dv = (2 * gms[b] / (bb * vv)) * (vv * T / np.hypot(bb, vv * T)) * (bvec / bb)
                ic = int(np.clip(np.searchsorted(tf, tca), 0, len(tf) - 1))
                impulse = (tca, Pfi[ic] @ dv)
                dist_b = bb
            else:
                dist_b = float(dist[jm])
            g = np.einsum("nij,nj->ni", Pfi, acc)
            cum = np.concatenate([[np.zeros(6)], np.cumsum(0.5 * (g[1:] + g[:-1]) * dtf[:, None], 0)])
            cum -= cum[i0f]
            if impulse is not None:
                tca, J = impulse
                cum += np.outer((tf > tca).astype(float) - float(t0 > tca), J)
            scores[b] = (float(dchi2(cum[iof][:, None, :])[0]), dist_b, float(tf[jm]))
    return Screen(scores, time.process_time() - c0)
