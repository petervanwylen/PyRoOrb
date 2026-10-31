"""Shared presentation view-models for the PyRoOrb front ends.

Both the CLI report (``cli.main``) and the web JSON bridge
(``web/engine_api.serialise_fit``) render the SAME FittedOrbit. Historically each
maintained its own copy of the orbital-element display table and its own
value-coercion helper, and the two had already drifted (different element labels;
both silently omitting mean-motion ``n``). This module is the single source of
truth for that shared formatting, so the two renderings can differ in STYLE
(terse CLI vs. descriptive web) without ever drifting in CONTENT.

Pure stdlib; imports nothing from the engine, so either front end can use it.
"""
import math


# Orbital elements shown to humans, in display order. The engine computes 11
# elements (``orbital_elements.ELEM_NAMES``, which also carries mean-motion ``n``); the
# two front ends deliberately show the same 10 -- ``n`` is redundant with ``a``/
# ``P`` and is intentionally omitted from BOTH (each used to drop it silently).
# Each row carries two label styles so the renderings can stay terse vs.
# descriptive without the tables drifting apart:
#   name : key into FittedOrbit.elements()/.element_sigmas()/.as_dict()
#   cli  : fixed-width label for the aligned CLI report
#   web  : descriptive label for the browser element panel
#   unit : display unit (shared by both)
ELEM_ROWS = [
    {"name": "a",        "cli": "a  ", "web": "semi-major axis",  "unit": "au"},
    {"name": "e",        "cli": "e  ", "web": "eccentricity",     "unit": ""},
    {"name": "q",        "cli": "q  ", "web": "perihelion dist.", "unit": "au"},
    {"name": "Q",        "cli": "Q  ", "web": "aphelion dist.",   "unit": "au"},
    {"name": "i",        "cli": "i  ", "web": "inclination",      "unit": "deg"},
    {"name": "asc_node", "cli": "node", "web": "ascending node",  "unit": "deg"},
    {"name": "arg_per",  "cli": "peri", "web": "arg. perihelion", "unit": "deg"},
    {"name": "M",        "cli": "M  ", "web": "mean anomaly",     "unit": "deg"},
    {"name": "Tp",       "cli": "Tp ", "web": "perihelion time",  "unit": "JD"},
    {"name": "P",        "cli": "P  ", "web": "period",           "unit": "yr"},
]


def safe(v):
    """Coerce numpy / None / non-finite into a JSON- and print-safe scalar: None
    stays None, a finite number becomes a Python float, NaN/inf collapse to None,
    and anything non-numeric is returned unchanged."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def element_rows(as_dict):
    """The web element panel: one dict per displayed element, value + 1-sigma read
    from a ``FittedOrbit.as_dict()`` payload (keys ``name`` and ``f"{name} sigma"``)
    and run through :func:`safe`."""
    return [{
        "name": r["name"], "label": r["web"], "unit": r["unit"],
        "value": safe(as_dict.get(r["name"])),
        "sigma": safe(as_dict.get(f"{r['name']} sigma")),
    } for r in ELEM_ROWS]


def amr_line(report):
    """The CLI's AMR line for a :func:`nongrav.amr_report` mapping (None -> None).

    find_orb prints "Epoch 1997 Oct 13.0 TT; AMR 0.034 m^2/kg" on the elements' epoch line for
    its 1-parameter SRP force model only; we give it a line of its own under the nongrav block.
    The uncertainty follows the same three-way shape as the coefficients themselves: ">" for an
    axis the profile never bounded, "+plus / -minus" when the halves genuinely differ, else a
    plain "±" -- and nothing at all before the sigmas have been computed."""
    if not report:
        return None
    lo, hi = report.get("minus"), report.get("plus")
    tail = "(area/mass ratio implied by A1, as a perfect absorber"
    if lo is None or hi is None:                       # sigmas not computed (or not asked for)
        return f"AMR = {report['amr']:.4g} m²/kg  {tail})"
    if report.get("unconstrained"):                    # A1's profile never rose by 1: a lower bound
        return (f"AMR = {report['amr']:.4g} m²/kg  {tail}; unconstrained -- "
                f"± is at least {max(lo, hi):.2g})")
    if abs(hi - lo) > 0.15 * max(abs(hi), abs(lo), 1e-300):
        return f"AMR = {report['amr']:.4g} +{hi:.2g} / -{lo:.2g} m²/kg  {tail})"
    return f"AMR = {report['amr']:.4g} ± {max(lo, hi):.2g} m²/kg  {tail})"
