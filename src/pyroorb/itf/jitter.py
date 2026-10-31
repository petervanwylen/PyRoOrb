"""Astrometry jitter: perturb the KNOWN object's own observations (never ITF
candidates) to explore orbit space stochastically when the uncertainty is large.

Split out of the former ``itf/obsutil.py``, where it was called
"varobs" / "noise". Off by default (``ItfConfig.jitter``).
"""


def jitter_line(line, cfg):
    """``line`` with gaussian noise added to RA/Dec (and time for ground-based
    obs), per the per-observatory sigma table.  Roving/satellite/radar lines
    are left untouched."""
    import math as _m
    import random

    if line[14:15] in "vrswqt":
        return line
    obscode = line[77:80]
    sigma = cfg.jitter_sigma_table.get(obscode, cfg.jitter_default_sigma) / 3600.0
    try:
        precise_ra = line[38:44]
        if precise_ra == "      ":
            precise_ra = "0"
        precise_dec = line[51:56]
        if precise_dec == "     ":
            precise_dec = "0"
        ra = (int(line[32:34]) + int(line[35:37]) / 60
              + float(precise_ra) / 3600) * 15
        dec = ((int(line[45:47]) + int(line[48:50]) / 60
                + float(precise_dec) / 3600) * int(line[44:45] + "1"))
        day = float(line[23:32])
        ra_acc = line[43:44] != " "
        day_acc = line[31:32] != " "
        cos_dec = _m.cos(_m.radians(dec))
        ra = ra + random.gauss(0, sigma) / cos_dec
        dec = dec + random.gauss(0, sigma)
        if line[14:15] not in "VRSWQT":
            day = day + random.gauss(0, cfg.jitter_sigma_t)
        ra_h_decimal = ra / 15
        ra_h = _m.floor(ra_h_decimal)
        ra_m_decimal = (ra_h_decimal % 1) * 60
        ra_m = _m.floor(ra_m_decimal)
        ra_s_decimal = (ra_m_decimal % 1) * 60
        dec_d = int(dec)
        dec_sign = ('{0:+}'.format(dec))[0:1]
        dec_m_decimal = (abs(dec) % 1) * 60
        dec_m = int(dec_m_decimal)
        dec_s_decimal = (dec_m_decimal % 1) * 60
        if ra_acc:
            radec_txt = (f'{ra_h:02d} {ra_m:02d} {ra_s_decimal:06.3f}'
                         f'{dec_sign}{abs(dec_d):02d} {dec_m:02d} {dec_s_decimal:05.2f}')
        else:
            radec_txt = (f'{ra_h:02d} {ra_m:02d} {ra_s_decimal:05.2f} '
                         f'{dec_sign}{abs(dec_d):02d} {dec_m:02d} {dec_s_decimal:04.1f} ')
        day_txt = f'{day:09.6f}' if day_acc else f'{day:08.5f} '
        return line[0:23] + day_txt + radec_txt + line[56:]
    except Exception:
        return line


def maybe_jitter(line, cfg):
    """Apply :func:`jitter_line` only when the jitter switch is on."""
    if getattr(cfg, "jitter", False):
        return jitter_line(line, cfg)
    return line
