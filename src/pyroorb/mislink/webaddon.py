"""The mislinkage (chimera) score as a web add-on (web/addons.py): the fit-page note, the
watch-list page and their two API routes."""
from pathlib import Path

from pyroorb.web.addons import MenuLink, WebAddon
from pyroorb.console import for_log
from pyroorb.web import errors


def mislinkage(ctx, designation, session=None, fit_key=None, page=False):
    """Mislinkage score of a 3-5 night single-opposition arc (``mislink.score``).

    By default the object's PUBLISHED obs_sbn rows are scored (what the watch list scores).
    With ``page`` the displayed fit's own observations are scored instead -- pasted or added
    astrometry, exactly the nights on screen. Any arc outside what the model scores answers
    ``ok`` with a ``status`` naming why; a failure is logged and answered ``ok: False``."""
    from pyroorb.mislink import score as ms
    desig = " ".join(str(designation or "").split())
    if not desig:
        return {"ok": False, "error": "missing designation"}
    try:
        if page:
            fit = ctx.fit_for(session, fit_key)
            if fit is None:
                return {"ok": False, "expired": True,
                        "error": "the current fit has expired -- re-run it"}
            res = ms.score_observations(list(fit._obs), desig, ctx.obscodes())
        else:
            res = ms.score_designation(desig, obscodes=ctx.obscodes())
    except Exception as e:
        msg = f"mislinkage score failed: {errors.describe(e)}"[:300]
        print(f"mislinkage: {for_log(desig)} ({'page' if page else 'published'}): {msg}",
              flush=True)
        return {"ok": False, "error": msg}
    if res.get("status") in ("fit-failed", "unavailable", "error"):
        print(f"mislinkage: {for_log(desig)} ({res.get('source')}): {res.get('status')} "
              f"{res.get('error') or ''}".rstrip(), flush=True)
    if res.get("status") == "ok":
        res = {**res, "descriptions": descriptions()}
    return {"ok": True, **res}


def descriptions():
    from pyroorb.mislink import score as ms
    return {k: {"label": v[0], "unit": v[1], "text": v[2]} for k, v in ms.DESCRIPTIONS.items()}


_CANDIDATES = None


def candidates():
    """The committed mislinkage watch list (``mislink/candidates.json``), read once."""
    global _CANDIDATES
    if _CANDIDATES is None:
        from pyroorb.mislink import score as ms
        _CANDIDATES = {"ok": True, **ms.candidates()}
    return _CANDIDATES


def _post_score(ctx, handler, d):
    # a replica read plus up to five short refits: through the bounded worker pool
    return 200, ctx.pooled("mislinkage", lambda: mislinkage(
        ctx, d.get("designation"), d.get("session"), d.get("fit_key"), bool(d.get("page"))))


ADDON = WebAddon(
    name="mislink",
    static_dir=Path(__file__).resolve().parent / "static",
    pages={"/mislinkage-candidates": "mislinkage-candidates.html",
           "/mislinkage-candidates/": "mislinkage-candidates.html"},
    get={"/api/mislinkage_candidates": lambda ctx, handler, qs: (200, candidates())},
    post={"/api/mislinkage": _post_score},
    menu=[MenuLink(id="mislinkLink", href="/mislinkage-candidates", label="Mislinkage model ↗",
                   title="Single-opposition objects the mislinkage model flags as likely "
                         "holding two bodies (opens in a new tab)")],
    scripts=["mislinkexplain.js", "mislinknote.js"],
    styles=["mislink.css"],
)
