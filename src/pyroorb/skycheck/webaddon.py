"""The sky check as a web add-on (web/addons.py): the /skycheck page and its API routes.

Requests never wait on a download: the catalogue is brought up to date by a background thread
at start-up and every REFRESH_S after, and a request uses the newest finished build. Until the
first build exists the API says so. Each check runs through the server's bounded worker pool.
"""
import math
import threading
import time
from pathlib import Path

from pyroorb.web.addons import MenuLink, WebAddon

REFRESH_S = 60.0            # the background loop's tick (each step downloads only when due:
                            # the catalogue daily, the NEOCP half-hourly while asked for)
NEOCP_WANTED_S = 86400.0    # the NEOCP is kept current while a request asked for it this recently
MAX_OBS = 3000              # an identify request's observations (keeps a check inside the
                            # web deadline: ~25 ms a tracklet, a few hundred tracklets)
_STATE = {"thread": None, "last": None, "error": None, "neocp_asked": 0.0}


def _jsonable(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _refresh_loop():
    from pyroorb.skycheck import catalogue, neocp
    while True:
        try:                            # downloads only when the sources are a day old
            catalogue.update()
            catalogue.load(refresh=False)
            _STATE["last"], _STATE["error"] = time.time(), None
        except Exception as ex:                                   # noqa: BLE001
            _STATE["error"] = f"{type(ex).__name__}: {ex}"[:300]
            print(f"skycheck: catalogue refresh failed: {_STATE['error']}", flush=True)
        if time.time() - _STATE["neocp_asked"] < NEOCP_WANTED_S:
            try:                        # requests read only this cache (refresh=False)
                neocp.fetch()
            except Exception as ex:                               # noqa: BLE001
                print(f"skycheck: NEOCP refresh failed: {type(ex).__name__}: {ex}", flush=True)
        time.sleep(REFRESH_S)


def _start(ctx):
    if _STATE["thread"] is None:
        th = threading.Thread(target=_refresh_loop, name="skycheck-refresh", daemon=True)
        _STATE["thread"] = th
        th.start()


def _ready():
    from pyroorb.skycheck import catalogue
    return catalogue.current_build() is not None


def status():
    """Catalogue readiness plus the engine's defaults, which the page fills its fields from
    (a literal in the page would silently diverge when an engine default changes)."""
    from pyroorb import skycheck
    from pyroorb.skycheck import catalogue
    cur = catalogue.current_build()
    meta = catalogue._meta(cur) if cur is not None else {}
    return {"ok": True, "ready": cur is not None, "orbits": meta.get("rows"),
            "epoch_jd": meta.get("main_epoch"), "built": meta.get("built"),
            "refresh_error": _STATE["error"],
            "defaults": {"radius": skycheck.DEFAULT_RADIUS_ARCMIN,
                         "match": skycheck.MATCH_ARCSEC, "motion": skycheck.MOTION_ARCSEC,
                         "max_obs": MAX_OBS}}


_NOT_READY = {"ok": False, "error": "the orbit catalogue is still being prepared "
                                    "(first start: a few minutes) -- try again shortly"}


def _neocp(d):
    """The request's NEOCP choice (off unless asked); asking keeps the cache current."""
    want = d.get("neocp") is True
    if want:
        _STATE["neocp_asked"] = time.time()
    return want


def _float(v, default=None):
    try:
        return default if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return default


def field(d, obscodes):
    from pyroorb import skycheck
    if not _ready():
        return _NOT_READY
    try:
        res = skycheck.check_field(d.get("ra"), d.get("dec"), d.get("time") or time.time() /
                                   86400.0 + 2440587.5,
                                   radius_arcmin=_float(d.get("radius"),
                                                        skycheck.DEFAULT_RADIUS_ARCMIN),
                                   obscode=str(d.get("station") or "500").strip(),
                                   mag_limit=_float(d.get("mag_limit")),
                                   comets=d.get("comets", True) is not False,
                                   neocp=_neocp(d), obscodes=obscodes,
                                   refresh=False)
    except (ValueError, KeyError) as ex:
        return {"ok": False, "error": f"could not read the request: {ex}"[:300]}
    return _jsonable({"ok": True, "time_utc": res.time_utc, "ra": res.ra, "dec": res.dec,
                      "radius_arcmin": res.radius_arcmin, "obscode": res.obscode,
                      "objects": res.objects, "notes": res.notes,
                      "catalogue": res.catalogue})


def identify(d, obscodes):
    import pyroorb
    from pyroorb import skycheck
    if not _ready():
        return _NOT_READY
    text = d.get("text") or ""
    try:
        obs = pyroorb.read_observations(text, obscodes=obscodes)
    except Exception as ex:                                       # noqa: BLE001
        return {"ok": False, "error": f"could not read the astrometry: {ex}"[:300]}
    if not len(obs):
        return {"ok": False, "error": "no observations found in the text"}
    if len(obs) > MAX_OBS:
        return {"ok": False, "error": f"{len(obs)} observations: at most {MAX_OBS} per check"}
    res = skycheck.identify(obs, radius_arcmin=_float(d.get("radius"),
                                                      skycheck.DEFAULT_RADIUS_ARCMIN),
                            match_arcsec=_float(d.get("match"), skycheck.MATCH_ARCSEC),
                            motion_arcsec=_float(d.get("motion"), skycheck.MOTION_ARCSEC),
                            comets=d.get("comets", True) is not False,
                            neocp=_neocp(d), obscodes=obscodes, refresh=False)
    out = []
    for tr in res:
        first = obs[tr.indices[0]]
        out.append({"label": tr.label, "obscode": tr.obscode, "n_obs": len(tr.indices),
                    "t_first": tr.t_first, "t_last": tr.t_last,
                    "ra": first["ra_deg"], "dec": first["dec_deg"],
                    "identification": tr.identification, "ambiguous": tr.ambiguous,
                    "candidates": tr.candidates[:10]})
    return _jsonable({"ok": True, "tracklets": out, "n_obs": len(obs)})


ADDON = WebAddon(
    name="skycheck",
    static_dir=Path(__file__).resolve().parent / "static",
    pages={"/skycheck": "skycheck.html", "/skycheck/": "skycheck.html"},
    get={"/api/skycheck/status": lambda ctx, handler, qs: (200, status())},
    post={"/api/skycheck/field": lambda ctx, handler, d: (
              200, ctx.pooled("skycheck-field", lambda: field(d, ctx.obscodes()))),
          "/api/skycheck/identify": lambda ctx, handler, d: (
              200, ctx.pooled("skycheck-identify", lambda: identify(d, ctx.obscodes())))},
    menu=[MenuLink(id="skycheckLink", href="/skycheck", label="Sky check ↗",
                   title="Known objects in a field, or the known object each tracklet of "
                         "some astrometry is (opens in a new tab)")],
    start=_start,
)
