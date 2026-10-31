"""Text/HTML exporters for the web UI (WEB_UI_PLAN §2.7).

Two products, both pure reads of a completed ``FittedOrbit`` (no engine_api edits):

* ``pseudo_mpec(fit)``    -- a find_orb-style **pseudo-MPEC** HTML report: astrometry
  block, orbital-elements block (with 1-sigma uncertainties), observatory roll-up,
  and a 3-column residuals block. Layout ported (not copied) from find_orb's
  ``make_pseudo_mpec`` in ``ephem0.cpp`` (see PROVENANCE).
* ``ephemeris_text(fit, obscodes, ...)`` -- the find_orb-style fixed-width ephemeris
  table, wrapping ``ephemeris.compute_ephemeris`` + ``ephemeris.format_table``.

Kept framework-agnostic so a later FastAPI app can serve the very same strings.
"""
import os
import sys
import html
import math

_HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _num(v, fmt, dash="--"):
    """Format a possibly-None / NaN number, falling back to a dash placeholder."""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return dash
        return format(f, fmt)
    except (TypeError, ValueError):
        return dash


def _yymmdd(jd_utc):
    """Compact YYYYMMDD.dd calendar tag from a UTC JD, for the residual columns."""
    try:
        from astropy.time import Time
        iso = Time(float(jd_utc), format="jd", scale="utc").iso  # 'YYYY-MM-DD hh:mm:ss.sss'
        ymd = iso[:10].replace("-", "")
        frac = (float(jd_utc) + 0.5) % 1.0
        return f"{ymd}.{int(frac * 100):02d}"
    except Exception:
        return f"JD{_num(jd_utc, '.5f')}"


def _obs_display_line(o, r):
    """The observation as it should appear in the astrometry <pre> block: prefer the
    original OBS80 line; otherwise synthesise a readable date/RA/Dec/code line."""
    line = o.get("obs80_line")
    if line:
        return line.rstrip("\n")
    from pyroorb.ephemerides import _ra_hms, _dec_dms
    date = _yymmdd(r.get("jd_utc"))
    ra = _ra_hms(o["ra_deg"]) if o.get("ra_deg") is not None else "  -  "
    dec = _dec_dms(o["dec_deg"]) if o.get("dec_deg") is not None else "  -  "
    mag = _num(o.get("mag"), "4.1f", dash="    ")
    # One band character, like the verbatim OBS80 lines this block is interleaved
    # with -- an ADES band is two ('Lg', 'Pw'), which both reads inconsistently
    # beside them and shifts the obscode a column. Same collapse the 80-column
    # rendering uses (obs80.band_to_obs80), never a bare truncation: 'Lg' is
    # Rubin g, and its first character is the SYSTEM.
    from pyroorb.obs80 import band_to_obs80
    band = band_to_obs80(o.get("band"))
    return f"{date:>14}  {ra} {dec}  {mag} {band}      {r.get('obscode','   ')}"


# --------------------------------------------------------------------------
# pseudo-MPEC (HTML)
# --------------------------------------------------------------------------
_CSS = """body{font-family:monospace;background:#0d1117;color:#d7dee8;margin:1.2rem}
a{color:#6cb6ff;text-decoration:none}a:hover{text-decoration:underline}
h2{color:#e8eef6;border-bottom:1px solid #30363d;padding-bottom:.2rem;margin-top:1.6rem}
pre{white-space:pre-wrap;line-height:1.35}
.excluded{color:#8b3b3b}.excluded a{color:#a55}
.meta{color:#8b949e}"""


def _elements_block(d):
    """find_orb-style orbital-elements listing with per-element 1-sigma uncertainties."""
    def sig(name, fmt):
        s = d.get(f"{name} sigma")
        return f"  +/- {_num(s, fmt)}" if s not in (None, "") else ""

    is_closed = (d.get("e") is not None and float(d["e"]) < 1.0)
    lines = [
        f"Epoch {_num(d.get('epoch'), '.5f')} JD (TDB)   = {html.escape(str(d.get('epoch_iso','')))[:23]} TDB",
        f"   M  {_num(d.get('M'), '12.5f')} deg{sig('M', '.2e')}"
        if is_closed else
        f"   Tp {_num(d.get('Tp'), '.5f')} JD{sig('Tp', '.2e')}",
        f"   Peri.dist. q  {_num(d.get('q'), '13.8f')} au{sig('q', '.2e')}",
        f"   Semimajor  a  {_num(d.get('a'), '13.8f')} au{sig('a', '.2e')}",
        f"   Eccentr.   e  {_num(d.get('e'), '13.8f')}{sig('e', '.2e')}",
        f"   Incl.      i  {_num(d.get('i'), '13.8f')} deg{sig('i', '.2e')}",
        f"   Asc.node   N  {_num(d.get('asc_node'), '13.8f')} deg{sig('asc_node', '.2e')}",
        f"   Arg.peri.  w  {_num(d.get('arg_per'), '13.8f')} deg{sig('arg_per', '.2e')}",
        f"   Period     P  {_num(d.get('P'), '13.6f')} yr",
        f"   H {_num(d.get('H'), '5.2f')}   G {_num(d.get('G'), '4.2f')}"
        f"      Earth MOID {_num(d.get('moid_earth_au'), '.5f')} au",
        f"   Class: {html.escape(str(d.get('class','?')))}   "
        f"U {_num(d.get('U'), '.1f')}   Tj {_num(d.get('tisserand_jupiter'), '.3f')}"
        f"{'   PHA' if d.get('pha') else ''}",
    ]
    span = _num(d.get('arc_days'), '.0f')
    lines.append(f"   From {d.get('n_obs')} of {d.get('n_obs_total')} obs over {span} d, "
                 f"{d.get('n_oppositions')} opp; mean residual {_num(d.get('rms_residual'), '.3f')}\"")
    return "\n".join(lines)


def _residuals_block(residuals, ncol=3):
    """3-column residual table (find_orb's layout): date, code, RA\" Dec\" per obs;
    excluded obs wrapped in parentheses."""
    cells = []
    for r in residuals:
        date = _yymmdd(r.get("jd_utc"))[:9]
        code = (r.get("obscode") or "   ")[:3]
        dra = _num(r.get("d_ra_arcsec"), "6.2f", dash="   -- ")
        ddec = _num(r.get("d_dec_arcsec"), "6.2f", dash="   -- ")
        code_a = f'<a href="#stn_{html.escape(code)}">{html.escape(code)}</a>'
        if r.get("included"):
            cells.append(f"{date} {code_a} {dra} {ddec} ")
        else:
            cells.append(f'<span class="excluded">({date} {code_a} {dra} {ddec})</span>')
    # lay the cells out column-major like find_orb (fills down each column)
    n = len(cells)
    rows = (n + ncol - 1) // ncol
    out = []
    for rr in range(rows):
        line = []
        for cc in range(ncol):
            idx = cc * rows + rr
            if idx < n:
                line.append(cells[idx])
        out.append("   ".join(line))
    return "\n".join(out)


def _undetermined_caveat(fit):
    """One sentence above the elements when the user asked for an orbit the arc does not
    determine (engine_api.nominal_optin): a pseudo-MPEC is prose, so unlike an MPCORB line it
    CAN say so, and it travels as far as one."""
    info = getattr(fit, "nominal_optin", None)
    if not info:
        return ""
    why = ("the most probable orbit given the known asteroid population, at the posterior-median "
           "distance" if info.get("source") == "population" else "the nominal least-squares orbit")
    return ("<div class='meta'><b>Not determined:</b> this arc does not determine an orbit. The "
            f"elements below are {why}, shown at the user's request; other orbits fit the "
            "astrometry as well.</div>")


def pseudo_mpec(fit):
    """Render a find_orb-style pseudo-MPEC (HTML string) for a completed fit."""
    d = fit.as_dict()
    try:
        H, sH = fit.absolute_magnitude()
        d["H"] = H
    except Exception:
        pass
    residuals = fit.residuals()
    obs_records = getattr(fit, "_obs", [])
    name = html.escape(str(d.get("object", "object")))

    # astrometry block
    astro = []
    for k, r in enumerate(residuals):
        o = obs_records[k] if k < len(obs_records) else {}
        code = (r.get("obscode") or "   ")[:3]
        disp = html.escape(_obs_display_line(o, r))
        anchor = f'<a name="o{k:03d}"></a><a href="#r{k:03d}">obs {k + 1:>4}</a>'
        cls = "" if r.get("included") else ' class="excluded"'
        astro.append(f'<span{cls}>{anchor}  {disp}</span>')

    # residual anchors line up index-for-index with the astrometry block
    resids = []
    for k, r in enumerate(residuals):
        resids.append(f'<a name="r{k:03d}"></a>')  # placeholder anchors precede the table
    resid_table = _residuals_block(residuals)

    # observatory roll-up
    by_code = {}
    for r in residuals:
        c = (r.get("obscode") or "   ")[:3]
        e = by_code.setdefault(c, {"n": 0, "used": 0})
        e["n"] += 1
        if r.get("included"):
            e["used"] += 1
    stations = []
    for c in sorted(by_code):
        e = by_code[c]
        stations.append(f'<a name="stn_{html.escape(c)}"></a>{html.escape(c)}'
                        f'   {e["used"]}/{e["n"]} obs used')

    desigs = ", ".join(html.escape(x) for x in d.get("designations", [d.get("object")]))
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Pseudo-MPEC for {name}</title><style>{_CSS}</style></head><body>",
        f"<h2>Pseudo-MPEC for {name}</h2>",
        f"<div class='meta'>Designations: {desigs} &middot; "
        f"force model: {html.escape(str(d.get('force_model','?')))} &middot; "
        f"generated by PyRoOrb</div>",
        "<h2 id='astrometry'>Astrometry</h2><pre>",
        "\n".join(astro),
        "</pre>",
        "<h2 id='elements'>Orbital elements</h2>",
        _undetermined_caveat(fit),
        "<pre>",
        _elements_block(d),
        "</pre>",
        "<h2 id='stations'>Observatory details</h2><pre>",
        "\n".join(stations),
        "</pre>",
        "<h2 id='residuals'>Residuals (RA, Dec in arcsec)</h2>",
        "".join(resids),
        "<pre>",
        resid_table,
        "</pre>",
        "</body></html>",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# ephemeris (fixed-width text)
# --------------------------------------------------------------------------
def ephemeris_text(fit, obscodes, jd_start, jd_stop, step_days, site="500",
                   faint_limit=None, suppress_unobservable=False):
    """find_orb-style ephemeris table over [jd_start, jd_stop] at `step_days`,
    seen from MPC `site`. Returns a plain-text string (header + fixed-width rows).
    `faint_limit` (V mag) and `suppress_unobservable` are find_orb's display cuts --
    they hide epochs the site can't shoot and note the tally above the table."""
    from pyroorb import ephemerides as ephemeris
    rows = ephemeris.compute_ephemeris(fit, obscodes, jd_start, jd_stop, step_days,
                                       site=site, full=bool(suppress_unobservable))
    rows, hidden = ephemeris.filter_rows(rows, faint_limit=faint_limit,
                                         suppress_unobservable=suppress_unobservable)
    header = (f"Pseudo-MPEC ephemeris for {fit.designation}\n"
              f"force model: {fit.force_model}\n")
    return header + ephemeris.format_table(rows, site=site, hidden=hidden)
