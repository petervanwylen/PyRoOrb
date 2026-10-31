"""HEALPix pixel lookup and inclusive disc search, in numpy.

The engine needs two HEALPix operations: the pixel holding a sky position
(EFCC18 debias tiles, RING order; the ITF tracklet index, NESTED order) and the
pixels a disc touches (ITF corridor windows against that index). Both follow the
HEALPix C++ library (Gorski et al. 2005, ApJ 622, 759; ``healpix_base.cc``
``loc2pix``, ``pix2loc``, ``max_pixrad`` and the hierarchical NESTED
``query_disc_internal``) operation for operation, so pixel numbers -- including
points exactly on a pixel edge or corner -- and inclusive disc sets are the ones
healpy returns. That keeps the debias table lookup and an index built with
healpy valid, and it removes a compiled dependency that has no Windows build.
"""
import functools
import math

import numpy as np

_INV_HALFPI = 0.6366197723675813430755350534900574
_HALFPI = 1.570796326794896619231321691639751
_TWOTHIRD = 2.0 / 3.0
# face -> ring / longitude offsets of its southern corner (HEALPix jrll / jpll)
_JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int64)
_JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)


def _order(nside):
    order = int(nside).bit_length() - 1
    if nside < 1 or (1 << order) != nside:
        raise ValueError(f"nside must be a power of 2, got {nside}")
    return order


def _spread_bits(v, order):
    out = np.zeros_like(v)
    for b in range(order):
        out |= ((v >> b) & 1) << (2 * b)
    return out


# even bits of a byte gathered into a nibble: the inverse of bit interleaving
_COMPRESS8 = np.array([sum(((b >> (2 * i)) & 1) << i for i in range(4))
                       for b in range(256)], dtype=np.int64)


def _compress_bits(v, order):
    out = _COMPRESS8[v & 0xFF]
    for k in range(1, (2 * order + 7) // 8):
        out = out | (_COMPRESS8[(v >> (8 * k)) & 0xFF] << (4 * k))
    return out


def _fmodulo4(v):
    """``v mod 4`` in [0, 4), the HEALPix ``fmodulo`` (exact at the wrap)."""
    out = np.where(v < 4.0, v, np.fmod(v, 4.0))
    neg = v < 0
    if np.any(neg):
        t = np.fmod(v[neg], 4.0) + 4.0
        out[neg] = np.where(t == 4.0, 0.0, t)
    return out


def ang2pix(nside, ra_deg, dec_deg, nest=False):
    """Pixel index of each (RA, Dec) in degrees; scalar in, int out."""
    order = _order(nside)
    scalar = np.ndim(ra_deg) == 0 and np.ndim(dec_deg) == 0
    if scalar:
        return _ang2pix_one(nside, order, float(ra_deg), float(dec_deg), nest)
    lon = np.atleast_1d(np.asarray(ra_deg, dtype=float))
    lat = np.atleast_1d(np.asarray(dec_deg, dtype=float))
    lon, lat = np.broadcast_arrays(lon, lat)
    theta = np.pi / 2.0 - np.radians(lat)
    phi = np.radians(lon)
    z = np.cos(theta)
    sth = np.sin(theta)
    have_sth = (theta < 0.01) | (theta > 3.14159 - 0.01)
    za = np.abs(z)
    tt = _fmodulo4(phi * _INV_HALFPI)
    pix = np.empty(z.shape, dtype=np.int64)

    eq = za <= _TWOTHIRD
    if np.any(eq):
        t1 = nside * (0.5 + tt[eq])
        t2 = nside * (z[eq] * 0.75) if nest else nside * z[eq] * 0.75
        jp = (t1 - t2).astype(np.int64)
        jm = (t1 + t2).astype(np.int64)
        if nest:
            ifp, ifm = jp >> order, jm >> order
            face = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))
            ix = jm & (nside - 1)
            iy = nside - (jp & (nside - 1)) - 1
            pix[eq] = _xyf2nest(ix, iy, face, order)
        else:
            nl4 = 4 * nside
            ir = nside + 1 + jp - jm
            kshift = 1 - (ir & 1)
            t = jp + jm - nside + kshift + 1 + nl4 + nl4
            ip = (t >> 1) & (nl4 - 1) if order > 0 else (t >> 1) % nl4
            pix[eq] = 2 * nside * (nside - 1) + (ir - 1) * nl4 + ip

    cap = ~eq
    if np.any(cap):
        zc, zac, ttc = z[cap], za[cap], tt[cap]
        use_sth = (zac >= 0.99) & have_sth[cap]
        tmp = np.where(use_sth,
                       nside * sth[cap] / np.sqrt((1.0 + zac) / 3.0),
                       nside * np.sqrt(3.0 * np.clip(1.0 - zac, 0.0, None)))
        if nest:
            ntt = np.minimum(3, ttc.astype(np.int64))
            tp = ttc - ntt
            jp = np.minimum((tp * tmp).astype(np.int64), nside - 1)
            jm = np.minimum(((1.0 - tp) * tmp).astype(np.int64), nside - 1)
            north = zc >= 0
            pix[cap] = np.where(
                north,
                _xyf2nest(nside - jm - 1, nside - jp - 1, ntt, order),
                _xyf2nest(jp, jm, ntt + 8, order))
        else:
            tp = ttc - ttc.astype(np.int64)
            jp = (tp * tmp).astype(np.int64)
            jm = ((1.0 - tp) * tmp).astype(np.int64)
            ir = jp + jm + 1
            ip = (ttc * ir).astype(np.int64)
            npix = 12 * nside * nside
            pix[cap] = np.where(zc > 0, 2 * ir * (ir - 1) + ip,
                                npix - 2 * ir * (ir + 1) + ip)
    return pix.reshape(np.shape(lon))


def _spread_one(v, order):
    out = 0
    for b in range(order):
        out |= ((v >> b) & 1) << (2 * b)
    return out


def _ang2pix_one(nside, order, ra_deg, dec_deg, nest):
    """Scalar :func:`ang2pix` in plain floats: the debias lookup runs once per
    observation, where numpy's per-call overhead would dominate."""
    theta = math.pi / 2.0 - math.radians(dec_deg)
    phi = math.radians(ra_deg)
    z, sth = math.cos(theta), math.sin(theta)
    have_sth = theta < 0.01 or theta > 3.14159 - 0.01
    za = abs(z)
    tt = phi * _INV_HALFPI
    if tt >= 4.0:
        tt = math.fmod(tt, 4.0)
    elif tt < 0.0:
        tt = math.fmod(tt, 4.0) + 4.0
        if tt == 4.0:
            tt = 0.0
    if za <= _TWOTHIRD:
        t1 = nside * (0.5 + tt)
        t2 = nside * (z * 0.75) if nest else nside * z * 0.75
        jp, jm = int(t1 - t2), int(t1 + t2)
        if nest:
            ifp, ifm = jp >> order, jm >> order
            face = (ifp | 4) if ifp == ifm else (ifp if ifp < ifm else ifm + 8)
            return ((face << (2 * order)) + _spread_one(jm & (nside - 1), order)
                    + (_spread_one(nside - (jp & (nside - 1)) - 1, order) << 1))
        nl4 = 4 * nside
        ir = nside + 1 + jp - jm
        t = jp + jm - nside + (1 - (ir & 1)) + 1 + nl4 + nl4
        ip = (t >> 1) & (nl4 - 1) if order > 0 else (t >> 1) % nl4
        return 2 * nside * (nside - 1) + (ir - 1) * nl4 + ip
    if za >= 0.99 and have_sth:
        tmp = nside * sth / math.sqrt((1.0 + za) / 3.0)
    else:
        tmp = nside * math.sqrt(3.0 * max(1.0 - za, 0.0))
    if nest:
        ntt = min(3, int(tt))
        tp = tt - ntt
        jp = min(int(tp * tmp), nside - 1)
        jm = min(int((1.0 - tp) * tmp), nside - 1)
        if z >= 0:
            ix, iy, face = nside - jm - 1, nside - jp - 1, ntt
        else:
            ix, iy, face = jp, jm, ntt + 8
        return (face << (2 * order)) + _spread_one(ix, order) + (_spread_one(iy, order) << 1)
    tp = tt - int(tt)
    jp, jm = int(tp * tmp), int((1.0 - tp) * tmp)
    ir = jp + jm + 1
    ip = int(tt * ir)
    if z > 0:
        return 2 * ir * (ir - 1) + ip
    return 12 * nside * nside - 2 * ir * (ir + 1) + ip


def _xyf2nest(ix, iy, face, order):
    return (face << (2 * order)) + _spread_bits(ix, order) + (_spread_bits(iy, order) << 1)


def _nest_zphi(pix, order):
    """(z, phi) of NESTED pixel centres at ``order``."""
    nside = 1 << order
    npix = 12 * nside * nside
    fact2 = 4.0 / npix
    fact1 = (nside << 1) * fact2
    face, ix, iy = _nest2xyf(pix, order)
    return _xyf_zphi(face, ix, iy, order)


def _nest2xyf(pix, order):
    ipf = pix & ((1 << (2 * order)) - 1)
    return pix >> (2 * order), _compress_bits(ipf, order), _compress_bits(ipf >> 1, order)


def _xyf_zphi(face, ix, iy, order):
    """(z, phi) of the pixel centre at face coordinates (ix, iy) of ``face``."""
    nside = 1 << order
    npix = 12 * nside * nside
    fact2 = 4.0 / npix
    fact1 = (nside << 1) * fact2
    jr = (_JRLL[face] << order) - ix - iy - 1
    north, south = jr < nside, jr > 3 * nside
    nr = np.where(north, jr, np.where(south, 4 * nside - jr, nside))
    t = (nr * nr) * fact2
    z = np.where(north, 1.0 - t, np.where(south, t - 1.0, (2 * nside - jr) * fact1))
    k = _JPLL[face] * nr + ix - iy
    k = np.where(k < 0, k + 8 * nr, k)
    phi = np.where(nr == nside, 0.75 * _HALFPI * k * fact1, (0.5 * _HALFPI * k) / nr)
    return z, phi


@functools.lru_cache(maxsize=None)
def _max_pixrad(nside):
    """Largest centre-to-corner angle of any pixel (radians)."""
    za, pa = _TWOTHIRD, math.pi / (4 * nside)
    t1 = (1.0 - 1.0 / nside) ** 2
    zb = 1.0 - t1 / 3.0
    sa, sb = math.sqrt((1 - za) * (1 + za)), math.sqrt((1 - zb) * (1 + zb))
    va = (sa * math.cos(pa), sa * math.sin(pa), za)
    vb = (sb, 0.0, zb)
    cross = (va[1] * vb[2] - va[2] * vb[1], va[2] * vb[0] - va[0] * vb[2],
             va[0] * vb[1] - va[1] * vb[0])
    dot = va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]
    return math.atan2(math.sqrt(sum(c * c for c in cross)), dot)


_TABLE_MAX_ORDER = 6     # pixel-centre tables up to nside 64 (65 k pixels, ~1.5 MB)


@functools.lru_cache(maxsize=None)
def _level_table(o):
    """(z, (1-z)(1+z), phi) of every NESTED pixel centre at order ``o``."""
    z, phi = _nest_zphi(np.arange(12 << (2 * o), dtype=np.int64), o)
    return z, (1.0 - z) * (1.0 + z), phi


def _centres(pix, o):
    if o <= _TABLE_MAX_ORDER:
        z, w, phi = _level_table(o)
        return z[pix], w[pix], phi[pix]
    z, phi = _nest_zphi(pix, o)
    return z, (1.0 - z) * (1.0 + z), phi


_DX4 = np.array([0, 1, 0, 1], dtype=np.int64)   # child (dx, dy) in NESTED order
_DY4 = np.array([0, 0, 1, 1], dtype=np.int64)


def query_disc_nest(nside, vec, radius, fact=4):
    """NESTED pixels overlapping one disc: centre unit vector ``vec``,
    ``radius`` in radians (see :func:`query_discs_nest`)."""
    return query_discs_nest(nside, np.reshape(vec, (1, 3)), [radius], fact)[0]


def query_discs_nest(nside, vecs, radii, fact=4):
    """NESTED pixels overlapping each of several discs (centre unit vectors
    ``vecs`` (n, 3), ``radii`` in radians). Inclusive, with sub-pixel checks
    ``fact`` (a power of 2) times finer: may return a few pixels just outside,
    never misses one inside. Returns one sorted int64 array per disc.

    The HEALPix walk descends from the 12 base pixels, keeping a pixel while its
    centre lies within the radius plus the largest pixel radius at that level;
    a boundary pixel at the target level is kept if some sub-pixel reached that
    way has its centre in the disc (or, at the finest level, in the margin).
    The same tests are evaluated level by level for all discs at once, which
    yields the same sets and amortises the per-call cost over a whole sweep.
    Sub-trees wholly inside a disc are emitted as ranges."""
    order = _order(nside)
    oplus = int(fact).bit_length() - 1
    if fact < 1 or (1 << oplus) != fact:
        raise ValueError("fact must be a power of 2")
    vecs = np.asarray(vecs, dtype=float).reshape(-1, 3)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    n = len(radii)
    npix = 12 * nside * nside
    out = [None] * n
    whole = radii >= math.pi
    for i in np.flatnonzero(whole):
        out[i] = np.arange(npix, dtype=np.int64)
    x, y, zv = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    theta = np.arctan2(np.sqrt(x * x + y * y), zv)
    pphi = np.where((x != 0.0) | (y != 0.0), np.arctan2(y, x), 0.0)
    pphi = np.where(pphi < 0.0, pphi + 2 * math.pi, pphi)
    pz = np.cos(theta)
    psz = (1.0 - pz) * (1.0 + pz)
    cosrad = np.cos(radii)
    margin, core = [], []           # cos(radius +/- largest pixel radius), per level
    for o in range(order + oplus + 1):
        dr = _max_pixrad(1 << o)
        margin.append(np.where(radii + dr > math.pi, -1.0, np.cos(radii + dr)))
        core.append(np.where(radii - dr < 0.0, 1.0, np.cos(radii - dr)))

    def cosdist(d, z, w, phi):
        return pz[d] * z + np.cos(pphi[d] - phi) * np.sqrt(psz[d] * w)

    d = np.repeat(np.flatnonzero(~whole), 12)
    pix = np.tile(np.arange(12, dtype=np.int64), int((~whole).sum()))
    run_d, run_lo, run_n = [], [], []
    for o in range(order + 1):
        if not pix.size:
            break
        z, w, phi = _centres(pix, o)
        cang = cosdist(d, z, w, phi)
        live = cang > margin[o][d]
        d, pix, cang = d[live], pix[live], cang[live]
        if o == order:
            break
        full = cang > core[o][d]
        if full.any():
            sd = 2 * (order - o)
            run_d.append(d[full]); run_lo.append(pix[full] << sd)
            run_n.append(np.full(int(full.sum()), 1 << sd, dtype=np.int64))
            d, pix = d[~full], pix[~full]
        d = np.repeat(d, 4)
        pix = _children(pix)

    got_d, got_p = [], []
    if pix.size:
        inside = cang >= cosrad[d]
        got_d.append(d[inside]); got_p.append(pix[inside])
        ed, ep = d[~inside], pix[~inside]
        if oplus == 0:
            got_d.append(ed); got_p.append(ep)
        elif ep.size:
            # descend each edge pixel's sub-pixels while they stay in the
            # margin; the edge pixel is kept on the first sub-pixel whose centre
            # is in the disc, or at the finest level in its margin
            face, ix, iy = _nest2xyf(ep, order)
            keep = np.zeros(ep.size, bool)
            e = np.arange(ep.size)          # which edge pixel each sub-pixel is in
            sx, sy = ix, iy
            for j in range(1, oplus + 1):
                e = np.repeat(e, 4)
                sx = (np.repeat(sx, 4) << 1) + np.tile(_DX4, sx.size)
                sy = (np.repeat(sy, 4) << 1) + np.tile(_DY4, sy.size)
                sub = order + j
                zz, pp = _xyf_zphi(face[e], sx, sy, sub)
                de = ed[e]
                c = cosdist(de, zz, (1.0 - zz) * (1.0 + zz), pp)
                if j < oplus:
                    keep[e[c >= cosrad[de]]] = True
                    go = (c > margin[sub][de]) & ~keep[e]
                    e, sx, sy = e[go], sx[go], sy[go]
                else:
                    keep[e[c > margin[sub][de]]] = True
            got_d.append(ed[keep]); got_p.append(ep[keep])
    if run_d:
        rd, lo, rn = (np.concatenate(v) for v in (run_d, run_lo, run_n))
        got_d.append(np.repeat(rd, rn))
        got_p.append(np.repeat(lo - np.r_[0, np.cumsum(rn)[:-1]], rn)
                     + np.arange(int(rn.sum()), dtype=np.int64))
    allp = np.concatenate(got_p) if got_p else np.empty(0, dtype=np.int64)
    alld = np.concatenate(got_d) if got_d else np.empty(0, dtype=np.int64)
    key = np.sort(alld * npix + allp)
    cuts = np.searchsorted(key, np.arange(n + 1) * npix)
    for i in range(n):
        if out[i] is None:
            out[i] = key[cuts[i]:cuts[i + 1]] - i * npix
    return out


def _children(pix):
    return (4 * pix[:, None] + np.arange(4, dtype=np.int64)).ravel()
