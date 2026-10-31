"""The distance/angle search kernel -- the innermost ~95% of the runtime.

This module holds exactly one function, :func:`search_all`, which fuses
what ``d2math.c`` spells as ``dRange``, ``searchDistance``,
``offsetMotionVector``, ``setupDistanceDependentVectors``,
``solveAngleRange``, ``searchAngles``, ``aRange``, ``tagAngle``,
``tkRand``, ``qeiToBin``/``hToBin``, and all fifteen orbit-class
predicates.

Two things about how it is written, both deliberate:

**It is one long function, not fifteen small ones.** This runs ~350k-500k
inner iterations per tracklet, and in CPython the per-call and
attribute-lookup overhead of the decomposed version was ~60% of total
runtime. The readable, one-concept-per-function form of every piece
still exists and is still authoritative -- :mod:`pydigest2.classes` for
the class predicates and :mod:`pydigest2.binning` for the bin lookups,
both directly unit-tested against the C definitions. Each block below
names the C function it mirrors; change the readable module first, then
mirror it here.

**It takes pre-allocated buffers and uses only indexing, arithmetic and
``math``.** That is what lets the *same source* serve as both the pure
-Python implementation and, when numba is installed, an ``njit``-compiled
one (see :mod:`pydigest2.ranging`). Writing it once means the compiled
and interpreted paths cannot silently diverge; ``tests/test_kernel.py``
asserts they agree exactly. The buffers are plain lists on the pure
-Python path and NumPy arrays under numba -- both support the ``buf[i]``
/ ``buf[i] = v`` used here -- except the population-model arrays, which
are always NumPy (they are read only on genuinely-new tags, so the
scalar-indexing cost there is immaterial).

Numeric fidelity: every arithmetic expression is written in the same
association order as the C source, because the bin-tagging decisions
downstream are threshold comparisons on these values. Integer
arithmetic for the LCG is done in ``uint64`` so it wraps exactly as the
C ``uint64_t`` does before the 59-bit mask is applied.
"""
from __future__ import annotations

import math

# Model extents, duplicated as plain ints so the kernel body stays free of
# imported globals (numba resolves these as compile-time constants).
_QX = 29
_EX = 8
_IX = 11
_HX = 18

_LCGA = 302875106592253          # 13**13, per NAG
_LCGM_MASK = (1 << 59) - 1
_INV_LCGM = 1.0 / (1 << 59)

_MIN_DISTANCE = 0.05
_MAX_DISTANCE = 100.0
_MIN_DISTANCE_STEP = 0.2
_MIN_ANGLE_STEP = 0.1
_AGE_LIMIT = 1
_U = 0.01720209895 * 0.01720209895        # K*K
_INV_K = 1.0 / 0.01720209895

_PI = math.pi


def search_all(
    pair_ra, pair_dec, pair_rms_ra, pair_rms_dec, obs_err,
    soe, coe, is_ades, no_threshold, no_obs_err,
    sun_obs, invdt, invdtsq, vmag, class_bits,
    all_ss, unk_ss, all_class, unk_class,
    qpart, epart, ipart, hpart,
    rand_state,
    d_in, d_out, g_in, g_out, d_keys,
    st_a1, st_a2, st_age, dst_lo, dst_hi, dst_age,
    sum_all_in, sum_unk_in, sum_all_out, sum_unk_out,
):
    """Run the whole distance/angle search for one tracklet.

    Accumulates population sums into ``sum_*`` (length-15 buffers) and
    returns the final LCG state. All other buffers are scratch.
    """
    scale = invdt * _INV_K
    so0x = sun_obs[0]
    so0y = sun_obs[1]
    so0z = sun_obs[2]
    so1x = sun_obs[3]
    so1y = sun_obs[4]
    so1z = sun_obs[5]

    st_cap = len(st_a1)
    dst_cap = len(dst_lo)

    # ---- dRange (d2math.c), as an explicit stack ----------------------
    # The traversal order is part of the algorithm: searchDistance draws
    # from the LCG, so a node's random numbers must be consumed before
    # its children's. Pushing the right half first makes the left half
    # pop first, reproducing the C code's pre-order descent.
    # The two seed distances are searched first, exactly as score() does.
    # score() searches the two limit distances first, then bisects between
    # them; age < 0 marks an entry as "search this exact distance, don't
    # recurse". Entries are laid out in reverse of the order they run,
    # since this is a LIFO.
    dst_lo[0] = _MIN_DISTANCE     # runs last: dRange(MIN, MAX, 0)
    dst_hi[0] = _MAX_DISTANCE
    dst_age[0] = 0
    dst_lo[1] = _MAX_DISTANCE     # runs second: searchDistance(MAX)
    dst_hi[1] = _MAX_DISTANCE
    dst_age[1] = -1
    dst_lo[2] = _MIN_DISTANCE     # runs first: searchDistance(MIN)
    dst_hi[2] = _MIN_DISTANCE
    dst_age[2] = -1
    dsp = 3

    while dsp > 0:
        dsp -= 1
        lo = dst_lo[dsp]
        hi = dst_hi[dsp]
        age_d = dst_age[dsp]
        if age_d < 0:
            d = lo                 # seed distance, searched directly
        else:
            d = (lo + hi) * 0.5

        # ============ searchDistance (d2math.c) ========================
        n_keys = 0                 # d-tags are cleared per distance
        d_any_tag = False
        new_tag = False

        ri = -1
        while ri <= 1:
            di = -1
            while di <= 1:
                # ---- offsetMotionVector (d2math.c) -------------------
                rx = ri
                dx = di
                u0x = 0.0
                u0y = 0.0
                u0z = 0.0
                u1x = 0.0
                u1y = 0.0
                u1z = 0.0
                for i in range(2):
                    efc = obs_err[i]
                    if is_ades:
                        # ---- updateRMSValues (d2math.c) -------------
                        rr = pair_rms_ra[i]
                        rd = pair_rms_dec[i]
                        e2 = efc if efc != 0.0 else 1.0
                        if rr == 0.0 and rd == 0.0:
                            rr = e2
                            rd = e2
                        else:
                            if rr == 0.0:
                                rr = rd
                            if rd == 0.0:
                                rd = rr
                            if no_threshold:
                                if e2 > rr:
                                    rr = e2
                                if e2 > rd:
                                    rd = e2
                            else:
                                mn = 0.7 * e2
                                mx = 5.0 * e2
                                if rr < mn:
                                    rr = mn
                                if rd < mn:
                                    rd = mn
                                if rr > mx:
                                    rr = mx
                                if rd > mx:
                                    rd = mx
                                if e2 > rr:
                                    rr = e2
                                if e2 > rd:
                                    rd = e2
                        dec = pair_dec[i] + dx * rd * 0.5
                        cosdec = math.cos(dec)
                        ra = pair_ra[i] + rx * rr * 0.5 * cosdec
                    else:
                        dec = pair_dec[i] + dx * efc * 0.5
                        cosdec = math.cos(dec)
                        ra = pair_ra[i] + rx * efc * 0.5 * cosdec
                    vx = math.cos(ra) * cosdec
                    vy = math.sin(ra) * cosdec
                    vz = math.sin(dec)
                    # ecRotate to ecliptic coordinates
                    ey = vz * soe + vy * coe
                    vz = vz * coe - vy * soe
                    vy = ey
                    if i == 0:
                        u0x = vx
                        u0y = vy
                        u0z = vz
                    else:
                        u1x = vx
                        u1y = vy
                        u1z = vz
                    rx = -rx
                    dx = -dx

                # ---- setupDistanceDependentVectors (d2math.c) --------
                oo0x = u0x * d
                oo0y = u0y * d
                oo0z = u0z * d
                s0x = so0x + oo0x
                s0y = so0y + oo0y
                s0z = so0z + oo0z
                s0_magsq = s0x * s0x + s0y * s0y + s0z * s0z
                s0_mag = math.sqrt(s0_magsq)
                o1x = s0x - so1x
                o1y = s0y - so1y
                o1z = s0z - so1z
                o1_magsq = o1x * o1x + o1y * o1y + o1z * o1z
                o1_mag = math.sqrt(o1_magsq)

                # H magnitude. The C code approximates x**.63 / x**1.22
                # with a 1024-point interpolated table purely for speed;
                # this is outside the inner loop, so the exact power is
                # used -- strictly more accurate, and the table's
                # interpolation error is well inside this port's tolerance.
                rdelta = d * s0_mag
                cospsi = (oo0x * s0x + oo0y * s0y + oo0z * s0z) / rdelta
                if cospsi > -0.9999:
                    tanhalf = math.sqrt(1.0 - cospsi * cospsi) / (1.0 + cospsi)
                    phi1 = math.exp(-3.33 * tanhalf ** 0.63)
                    phi2 = math.exp(-1.87 * tanhalf ** 1.22)
                    hmag = (vmag - 5.0 * math.log10(rdelta)
                            + 2.5 * math.log10(0.85 * phi1 + 0.15 * phi2))
                else:
                    # straight into the sun; valid but meaningless H
                    hmag = 30.0

                # hToBin (d2model.c)
                ih = 0
                while ih < _HX - 1 and hmag >= hpart[ih]:
                    ih += 1
                lo_h22 = hmag < 22.5
                lo_h18 = hmag < 18.5

                # ---- solveAngleRange (d2math.c) ----------------------
                th = (o1x * u1x + o1y * u1y + o1z * u1z) / o1_mag
                tz = math.acos(th)
                aa = invdtsq
                bb = (-2.0 * o1_mag * th) * aa
                cc = o1_magsq * aa - 2.0 * _U / s0_mag
                dsc = bb * bb - 4 * aa * cc
                # "not >" so inf/nan fail too, as in the C source
                if not (dsc > 0.0):
                    if no_obs_err:
                        di = 2       # break out of both offset loops
                        ri = 2
                    di += 1
                    continue

                sd = math.sqrt(dsc)
                sd1 = -sd
                inv2aa = 0.5 / aa
                ang1 = 0.0
                ang2 = 0.0
                while True:
                    dd2 = (-bb + sd1) * inv2aa
                    d2s = dd2 * dd2
                    nns = d2s + o1_magsq - 2.0 * dd2 * o1_mag * th
                    nn = math.sqrt(nns)
                    ca = (nns + o1_magsq - d2s) / (2.0 * nn * o1_mag)
                    sa = dd2 * math.sin(tz) / nn
                    ang2 = 2.0 * math.atan2(sa, 1.0 + ca)
                    if sd1 == sd:
                        break
                    ang1 = ang2
                    sd1 = sd

                # ======== aRange + tagAngle (d2math.c) ================
                sp = 0
                st_a1[0] = ang1
                st_a2[0] = ang2
                st_age[0] = 0
                sp = 1
                any_tag = False

                while sp > 0:
                    sp -= 1
                    a1 = st_a1[sp]
                    a2 = st_a2[sp]
                    age = st_age[sp]
                    d3 = (a2 - a1) / 3.0

                    # ---- tkRand (d2math.c) ----
                    rand_state = (rand_state * _LCGA) & _LCGM_MASK
                    mid = a1 + d3 + d3 * (rand_state * _INV_LCGM)

                    tagged = False
                    dd = o1_mag * math.sin(mid) / math.sin(_PI - mid - tz)

                    # velocity, scaled by the gravitational constant
                    vx = (dd * u1x - o1x) * scale
                    vy = (dd * u1y - o1y) * scale
                    vz = (dd * u1z - o1z) * scale

                    # momentum vector (cross3) and its magnitude
                    hvx = s0y * vz - s0z * vy
                    hvy = s0z * vx - s0x * vz
                    hvz = s0x * vy - s0y * vx
                    hsq = hvx * hvx + hvy * hvy + hvz * hvz
                    vsq = vx * vx + vy * vy + vz * vz
                    temp = 2.0 - s0_mag * vsq

                    # stability: require a < 100, then e < .99
                    if s0_mag <= temp * 100.0:
                        e = math.sqrt(1.0 - hsq * (temp / s0_mag))
                        if e <= 0.99:
                            hm = math.sqrt(hsq)
                            # reliable i=0 check; handles precision loss in h
                            if hvz >= hm:
                                orb_i = 0.0
                            else:
                                orb_i = math.degrees(math.acos(hvz / hm))
                            q = (s0_mag / temp) * (1.0 - e)

                            # ---- qeiToBin (d2model.c) ----
                            iq = 0
                            while iq < _QX and q >= qpart[iq]:
                                iq += 1
                            if iq < _QX:
                                ie = 0
                                while ie < _EX and e >= epart[ie]:
                                    ie += 1
                                if ie < _EX:
                                    ii = 0
                                    while ii < _IX and orb_i >= ipart[ii]:
                                        ii += 1
                                    if ii < _IX:
                                        # ==== orbit-class predicates ====
                                        # (d2model.c; see pydigest2.classes)
                                        # One bit per class.
                                        if q < 1.3:
                                            # Int and NEO implied by q < 1.3
                                            m = 3
                                            if lo_h22:
                                                m = 7
                                                if lo_h18:
                                                    m = 15
                                        else:
                                            m = 0
                                            # aphelion term, shared by Int and MC
                                            qq = q * (1.0 + e) / (1.0 - e)
                                            if e >= 0.5 or orb_i >= 40.0 or qq > 10.0:
                                                m = 1                     # Int
                                            if q < 1.67 and qq > 1.58:
                                                m |= 16                   # MC
                                            t = (5.2 * (1.0 - e) / q
                                                 + 2.0 * math.sqrt(q * (1.0 + e) / 5.2)
                                                 * math.cos(math.radians(orb_i)))
                                            if 2.0 < t and t < 3.0:
                                                m |= 16384                # JFC

                                        # The rest are bounded windows in
                                        # semi-major axis, so branch on a.
                                        a = q / (1.0 - e)
                                        if a < 2.5:
                                            if (a > 2.1 and q >= 1.67
                                                    and orb_i < ((a - 2.1) / .4) * 10 + 7):
                                                m |= 128                  # MB1
                                            if a < 2.45:
                                                if (a > 2.2 and q >= 1.5
                                                        and 20 <= orb_i and orb_i <= 27):
                                                    m |= 64               # Pho
                                                if (1.78 < a and a < 2.0 and e <= .18
                                                        and 16 <= orb_i and orb_i <= 34):
                                                    m |= 32               # Hun
                                        elif a < 2.8:
                                            if a > 2.5:
                                                if e <= .35 and 24 <= orb_i and orb_i <= 37:
                                                    m |= 256              # Pal
                                                if e <= .45 and orb_i <= 20:
                                                    m |= 1024             # MB2
                                                if (2.55 < a and a < 2.72 and e <= .25
                                                        and 20 <= orb_i and orb_i <= 23.5):
                                                    m |= 512              # Han
                                        elif a < 3.25:
                                            if (a > 2.8 and e <= .4
                                                    and orb_i < ((a - 2.8) / .45) * 16 + 20):
                                                m |= 2048                 # MB3
                                        elif a < 4.02:
                                            if a > 3.9 and orb_i <= 18 and e <= .4:
                                                m |= 4096                 # Hil
                                        elif a < 5.35:
                                            if a > 5.05 and e <= .22 and orb_i <= 38:
                                                m |= 8192                 # JTr

                                        # ==== bin tagging ====
                                        key = ((iq * _EX + ie) * _IX + ii) * _HX + ih
                                        mi = m & class_bits
                                        mo = class_bits & ~m
                                        prev_i = d_in[key]
                                        prev_o = d_out[key]
                                        new_i = mi & ~prev_i
                                        new_o = mo & ~prev_o
                                        if new_i != 0 or new_o != 0:
                                            tagged = True
                                            any_tag = True
                                            if prev_i == 0 and prev_o == 0:
                                                d_keys[n_keys] = key
                                                n_keys += 1
                                            if new_i != 0:
                                                d_in[key] = prev_i | mi
                                            if new_o != 0:
                                                d_out[key] = prev_o | mo

                    # ---- back in aRange ----
                    # ".1 rad = sufficiently large"
                    if tagged or d3 > _MIN_ANGLE_STEP:
                        if sp + 2 > st_cap:
                            raise RuntimeError("angle search stack overflow")
                        st_a1[sp] = mid
                        st_a2[sp] = a2
                        st_age[sp] = 0
                        sp += 1
                        st_a1[sp] = a1
                        st_a2[sp] = mid
                        st_age[sp] = 0
                        sp += 1
                    elif age < _AGE_LIMIT:
                        if sp + 2 > st_cap:
                            raise RuntimeError("angle search stack overflow")
                        st_a1[sp] = mid
                        st_a2[sp] = a2
                        st_age[sp] = age + 1
                        sp += 1
                        st_a1[sp] = a1
                        st_a2[sp] = mid
                        st_age[sp] = age + 1
                        sp += 1

                # ==== searchAngles, second half: merge into global tags ====
                # dAnyTag is cumulative across the nine offsets, so a later
                # offset that tags nothing still merges what an earlier one
                # found -- matching the C control flow.
                if any_tag:
                    d_any_tag = True
                if d_any_tag:
                    for ki in range(n_keys):
                        key = d_keys[ki]
                        mi = d_in[key]
                        if mi != 0:
                            prev = g_in[key]
                            bits = mi & ~prev
                            if bits != 0:
                                new_tag = True
                                g_in[key] = prev | mi
                                for c in range(15):
                                    if (bits >> c) & 1:
                                        sum_all_in[c] += all_class[c, key]
                                        sum_unk_in[c] += unk_class[c, key]
                        mo = d_out[key]
                        if mo != 0:
                            prev = g_out[key]
                            bits = mo & ~prev
                            if bits != 0:
                                new_tag = True
                                g_out[key] = prev | mo
                                a_ss = all_ss[key]
                                u_ss = unk_ss[key]
                                for c in range(15):
                                    if (bits >> c) & 1:
                                        sum_all_out[c] += a_ss - all_class[c, key]
                                        sum_unk_out[c] += u_ss - unk_class[c, key]

                if no_obs_err:
                    di = 2
                    ri = 2
                di += 1
            ri += 1

        # clear this distance's tags (only the keys actually touched)
        for ki in range(n_keys):
            key = d_keys[ki]
            d_in[key] = 0
            d_out[key] = 0

        # ---- back in dRange ----
        if age_d >= 0:
            # ".2 au = big"
            if new_tag or hi - lo > _MIN_DISTANCE_STEP:
                if dsp + 2 > dst_cap:
                    raise RuntimeError("distance search stack overflow")
                dst_lo[dsp] = d
                dst_hi[dsp] = hi
                dst_age[dsp] = 0
                dsp += 1
                dst_lo[dsp] = lo
                dst_hi[dsp] = d
                dst_age[dsp] = 0
                dsp += 1
            elif age_d < _AGE_LIMIT:
                if dsp + 2 > dst_cap:
                    raise RuntimeError("distance search stack overflow")
                dst_lo[dsp] = d
                dst_hi[dsp] = hi
                dst_age[dsp] = age_d + 1
                dsp += 1
                dst_lo[dsp] = lo
                dst_hi[dsp] = d
                dst_age[dsp] = age_d + 1
                dsp += 1

    return rand_state
