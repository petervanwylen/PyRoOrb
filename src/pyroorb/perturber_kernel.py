"""Build the extra-perturber kernel (perturbers.KERNEL_NAME) from JPL's sb441-n373.

sb441-n373.bsp holds the 343 DE441 small-body perturbers plus 30 trans-Neptunian objects over
DE441's whole span -- 15 GB. Only the records inside the windows below are read, over HTTP range
requests (~175 MB), and written as ONE Sun-centred DAF/SPK of type 2 that ASSIST loads as its
asteroid file:

* the 16 bodies of sb441-n16 are JPL's records verbatim over 1800-2200 (BIG16_START_YEAR ..
  BIG16_STOP_YEAR), so a fit in the 16 integrates exactly as with sb441-n16 there. The wider span
  is for the default install, which has no sb441-n16 to fall back to: every numbered asteroid
  observed before 1900 (409 in MPCORB, from (1) Ceres in 1801) keeps the 16 instead of dropping
  to planets-only, at 7.6 MB for the extra two centuries;
* the other 327 asteroids with a DE440 mass are re-fitted per body to the cheapest record
  layout that stays within TOL_KM of JPL (eccentric Mars-crossers such as (433) Eros need short
  records near perihelion; a single 128-day layout left them 1000-2000 km off), over 1900-2100
  (START_YEAR .. STOP_YEAR). A field with any of them is therefore bounded by 1900-2100, the
  16 alone by 1800-2200 (perturbers.window: each compacted Ephem covers the intersection of
  its bodies' segments).

ASSIST's reader fixes the layout: a target's segment is found from the span of its FIRST
segment, so each target gets exactly one segment; record size, interval and coefficient count
are read from each segment's trailer (fewer than 32 coefficients).
"""
import concurrent.futures as cf
import struct
import urllib.request

import numpy as np

N373_URL = "https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n373.bsp"
START_YEAR, STOP_YEAR = 1900, 2100
BIG16_START_YEAR, BIG16_STOP_YEAR = 1800, 2200
TOL_KM = 20.0
BIG16 = (1, 2, 3, 4, 7, 10, 15, 16, 31, 52, 65, 87, 88, 107, 511, 704)
DAY = 86400.0
REC = 1024
ND, NI = 2, 6
SS = ND + (NI + 1) // 2
NSUM_MAX = (128 - 3) // SS
# (record days, coefficients), cheapest bytes-per-year first
CHOICES = sorted(((d, n) for d in (32, 64, 128, 256, 512) for n in range(6, 17)),
                 key=lambda c: ((2 + 3 * c[1]) / c[0], -c[0]))


def _range(url, a, b, tries=4):
    """Bytes [a, b) of `url`. A build makes ~700 range requests, 8 at a time, so a transient
    gateway or connection error is retried (2, 4, 8 s) rather than ending the build."""
    import time
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            break
        except OSError:                       # URLError, HTTPError and timeouts are OSErrors
            if k == tries - 1:
                raise
            time.sleep(2.0 * 2 ** k)
    if len(data) != b - a:
        raise IOError(f"range {a}-{b} returned {len(data)} bytes")
    return data


def _summaries(read):
    fr = read(0, REC)
    if fr[:8] != b"DAF/SPK " or fr[88:96] != b"LTL-IEEE":
        raise ValueError("not a little-endian DAF/SPK file")
    rec, = struct.unpack("<i", fr[76:80])
    segs = []
    while rec:
        b = read((rec - 1) * REC, rec * REC)
        nxt, _, nsum = struct.unpack("<3d", b[:24])
        for k in range(int(nsum)):
            o = 24 + k * SS * 8
            et0, et1 = struct.unpack("<2d", b[o:o + 16])
            segs.append((et0, et1) + struct.unpack(f"<{NI}i", b[o + 16:o + 16 + 4 * NI]))
        rec = int(nxt)
    return fr, segs


def de440_masses(planets_path):
    """{MP number: GM} from the DE440 header constants MA0001..MA7999 (MA8xxx are TNOs)."""
    with open(planets_path, "rb") as f:
        b = f.read(65536)
    names = [b[252 + 6 * i:258 + 6 * i] for i in range(400)]
    o = 252 + 2400 + 24
    ncon, = struct.unpack("<i", b[o:o + 4])
    o += 4 + 16
    ipt = list(struct.unpack("<36i", b[o:o + 144])); o += 144 + 4
    ipt += list(struct.unpack("<3i", b[o:o + 12])); o += 12
    names += [b[o + 6 * i:o + 6 * i + 6] for i in range(ncon - 400)]
    ncoeff = max(ipt[3 * i] + ipt[3 * i + 1] * ipt[3 * i + 2] * (2 if i == 11 else 3)
                 for i in range(13)) - 1
    vals = struct.unpack(f"<{ncon}d", b[ncoeff * 8:ncoeff * 8 + 8 * ncon])
    out = {}
    for n, v in zip(names, vals):
        n = n.decode("ascii").strip()
        if n.startswith("MA") and n[2:].isdigit() and int(n[2:]) < 8000:
            out[int(n[2:])] = v
    return out


def _cheb_eval(records, init, intlen, et):
    nc = (records.shape[1] - 2) // 3
    i = np.clip(((et - init) // intlen).astype(int), 0, len(records) - 1)
    r = records[i]
    z = (et - r[:, 0]) / r[:, 1]
    T = np.empty((len(et), nc))
    T[:, 0] = 1.0
    if nc > 1:
        T[:, 1] = z
    for p in range(2, nc):
        T[:, p] = 2 * z * T[:, p - 1] - T[:, p - 2]
    return np.stack([np.einsum("ij,ij->i", T, r[:, 2 + k * nc:2 + (k + 1) * nc]) for k in range(3)], 1)


def _refit(records, init, intlen, t0, t1, days, ncoef):
    L = days * DAY
    nrec = int(np.ceil((t1 - t0) / L))
    m = 3 * ncoef
    x = np.cos(np.pi * (np.arange(m) + 0.5) / m)
    mids = t0 + (np.arange(nrec) + 0.5) * L
    pos = _cheb_eval(records, init, intlen, (mids[:, None] + x[None, :] * L / 2).ravel()).reshape(nrec, m, 3)
    pinv = np.linalg.pinv(np.polynomial.chebyshev.chebvander(x, ncoef - 1))
    out = np.empty((nrec, 2 + 3 * ncoef))
    for k in range(3):
        out[:, 2 + k * ncoef:2 + (k + 1) * ncoef] = pos[:, :, k] @ pinv.T
    out[:, 0] = mids
    out[:, 1] = L / 2
    return out, t0, L


def _write_spk(path, segments, comment, template_fr):
    raw = comment.replace("\n", "\0").encode("ascii") + b"\x04"
    ncom = -(-len(raw) // 1000)
    com = b"".join(raw[i * 1000:(i + 1) * 1000].ljust(REC, b"\0") for i in range(ncom))
    blocks = []
    rec_no = 2 + ncom
    for c in range(0, len(segments), NSUM_MAX):
        chunk = segments[c:c + NSUM_MAX]
        addr = (rec_no + 1) * 128 + 1
        summ, names, data = [], [], []
        for tgt, init, intlen, recs, eb, ee in chunk:
            arr = np.concatenate([recs.ravel(), [init, intlen, recs.shape[1], len(recs)]])
            summ.append(struct.pack("<2d6i", eb, ee, tgt, 10, 1, 2, addr, addr + len(arr) - 1))
            names.append(f"{tgt}".ljust(SS * 8).encode("ascii"))
            data.append(arr.astype("<f8").tobytes())
            addr += len(arr)
        nrec_data = -(-sum(len(d) for d in data) // REC)
        nxt = rec_no + 2 + nrec_data if c + NSUM_MAX < len(segments) else 0
        prv = blocks[-1][3] if blocks else 0
        srec = (struct.pack("<3d", nxt, prv, len(chunk)) + b"".join(summ)).ljust(REC, b"\0")
        blocks.append((srec, b"".join(names).ljust(REC, b" "),
                       b"".join(data).ljust(nrec_data * REC, b"\0"), rec_no, addr))
        rec_no += 2 + nrec_data
    fr = bytearray(template_fr)
    fr[16:76] = b"pyroorb perturber kernel".ljust(60)
    fr[76:88] = struct.pack("<3i", 2 + ncom, blocks[-1][3], blocks[-1][4])
    tmp = str(path) + ".part"
    with open(tmp, "wb") as f:
        f.write(bytes(fr))
        f.write(com)
        for srec, nrec, data, _, _ in blocks:
            f.write(srec); f.write(nrec); f.write(data)
    import os
    os.replace(tmp, path)


# The SHA-256 of the kernel bundled with the package (perturbers.BUNDLED_KERNEL), the build
# this module made from sb441-n373 at TOL_KM; a test checks the shipped file against it.
KERNEL_SHA256 = "e3c9361cf2096d47845ccd174247edc63ab2e37c6608d84cc58af8c4bf1ba4db"


def comment_text(n_refit, tol_km, worst_km):
    """The kernel's comment area: who produced this modified kernel, from what, how it was
    validated, and the source's terms. NAIF's rules for a modified kernel ask that the file be
    renamed, annotated and validated, and that the attribution name whoever made the last
    modification; JPL's data-use policy asks for the source URL and the credit line, and that
    no endorsement by Caltech, JPL or NASA be claimed or implied."""
    return (
        "PyRoOrb asteroid-perturber kernel (an ASSIST asteroid file, SPK type 2, Sun-centred)\n"
        "\n"
        "PRODUCER: the PyRoOrb project (Peter VanWylen), https://github.com/petervanwylen/PyRoOrb\n"
        "This is a MODIFIED kernel made by PyRoOrb from JPL data. It is NOT a JPL product, and no\n"
        "endorsement by Caltech, JPL or NASA is claimed or implied.\n"
        "\n"
        "SOURCE: JPL Solar System Dynamics sb441-n373.bsp (the DE441 small-body perturbers),\n"
        f"{N373_URL}\n"
        "Park, R. S., Folkner, W. M., Williams, J. G., Boggs, D. H., 2021, AJ 161, 105.\n"
        "Courtesy NASA/JPL-Caltech.\n"
        "\n"
        "MODIFICATIONS: one segment per body; the 343 asteroids with a DE440 header mass (the\n"
        "trans-Neptunian objects left out). The 16 bodies of sb441-n16 are JPL's records verbatim,\n"
        f"cut to {BIG16_START_YEAR}-{BIG16_STOP_YEAR}; the other {n_refit} are cut to "
        f"{START_YEAR}-{STOP_YEAR} and re-fitted per\n"
        f"body to the cheapest Chebyshev record layout within {tol_km:g} km of JPL "
        f"(worst {worst_km:.1f} km).\n"
        "VALIDATION: read back with jplephem and ASSIST; the 16 bit-identical to sb441-n16.\n"
        "Masses are not in this file (ASSIST joins the DE440 header GMs; PyRoOrb may override).\n"
        "\n"
        "TERMS: distributed under JPL's data-use policy (use by any person for any purpose; Caltech\n"
        "makes no warranty as to quality and is not responsible for any loss from its use) and\n"
        "NAIF's rules for redistributing modified kernels.\n")


def build(out_path, planets_path, url=N373_URL, tol_km=TOL_KM, verbose=True, workers=8):
    """Read n373's records (the 16 over 1800-2200, the rest over 1900-2100) and write the
    kernel to `out_path`; returns the path."""
    read = lambda a, b: _range(url, a, b)
    fr, segs = _summaries(read)
    gm = de440_masses(planets_path)
    t0 = (START_YEAR - 2000) * 365.25 * DAY
    t1 = (STOP_YEAR - 2000) * 365.25 * DAY

    def span(n):
        if n in BIG16:
            return ((BIG16_START_YEAR - 2000) * 365.25 * DAY,
                    (BIG16_STOP_YEAR - 2000) * 365.25 * DAY)
        return t0, t1

    jobs = [s for s in segs if 0 < s[2] - 2000000 < 8000 and s[2] - 2000000 in gm
            and s[0] <= span(s[2] - 2000000)[0] and s[1] >= span(s[2] - 2000000)[1]]
    if {s[2] - 2000000 for s in jobs} != set(gm):
        raise ValueError("n373 does not cover the window with one segment per DE440-mass asteroid")

    def one(s):
        init, intlen, rsize, _ = struct.unpack("<4d", read((s[7] - 4) * 8, s[7] * 8))
        rsize = int(rsize)
        a0, a1 = span(s[2] - 2000000)
        i0 = int((a0 - init) // intlen)
        i1 = int(np.ceil((a1 - init) / intlen))
        a = (s[6] - 1) * 8 + i0 * rsize * 8
        data = np.frombuffer(read(a, a + (i1 - i0) * rsize * 8), "<f8").reshape(i1 - i0, rsize)
        return s[2] - 2000000, data, init + i0 * intlen, intlen

    cut = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        for k, (n, data, init, intlen) in enumerate(ex.map(one, jobs)):
            cut[n] = (data, init, intlen)
            if verbose and k % 100 == 0:
                print(f"  read {k + 1}/{len(jobs)} bodies", flush=True)
    test_et = np.random.default_rng(7).uniform(t0, t1, 20000)
    segments, worst = [], 0.0
    for n in sorted(cut, key=lambda n: (n not in BIG16, n)):
        recs, init, intlen = cut[n]
        if n in BIG16:
            segments.append((2000000 + n, init, intlen, recs, init, init + len(recs) * intlen))
            continue
        ref = _cheb_eval(recs, init, intlen, test_et)
        for d, nc in CHOICES:
            new, i2, L = _refit(recs, init, intlen, t0, t1, d, nc)
            err = float(np.abs(_cheb_eval(new, i2, L, test_et) - ref).max())
            if err <= tol_km:
                break
        else:
            raise ValueError(f"({n}): no record layout within {tol_km} km")
        worst = max(worst, err)
        segments.append((2000000 + n, i2, L, new, i2, i2 + len(new) * L))
    comment = comment_text(len(segments) - 16, tol_km, worst)
    _write_spk(out_path, segments, comment, fr)
    if verbose:
        print(f"  wrote {out_path}: {len(segments)} bodies, worst re-fit {worst:.1f} km")
    return out_path
