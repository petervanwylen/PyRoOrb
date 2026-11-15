"""Command line: ``pyroorb-skycheck field|identify|update`` (also ``python -m pyroorb.skycheck``).

    pyroorb-skycheck field "04 09 32.4" "+14 41 33" "2026-03-20T02:30:30" --station G96
    pyroorb-skycheck identify night.obs [--json]
    pyroorb-skycheck update
"""
import argparse
import json
import math
import sys


def _jsonable(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def main(argv=None):
    from pyroorb import skycheck
    ap = argparse.ArgumentParser(prog="pyroorb-skycheck",
                                 description="Which known objects are in a field, or which "
                                             "known object each tracklet of a file is.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--radius", type=float, default=skycheck.DEFAULT_RADIUS_ARCMIN,
                       help="search radius, arcmin (default %(default)s)")
        p.add_argument("--no-comets", action="store_true", help="leave comets out")
        p.add_argument("--neocp", action="store_true",
                       help="also check the NEO Confirmation Page's objects (times within "
                            "the last 30 days)")
        p.add_argument("--no-refresh", action="store_true",
                       help="no downloads: the catalogue and NEOCP orbits as last fetched")
        p.add_argument("--json", action="store_true", help="JSON output")

    f = sub.add_parser("field", help="objects within a radius of a position at a time")
    f.add_argument("ra", help="right ascension: degrees, or 'HH MM SS.s'")
    f.add_argument("dec", help="declination: degrees, or '+DD MM SS'")
    f.add_argument("time", help="UTC: ISO, 'YYYY MM DD.ddddd' or a Julian date")
    f.add_argument("--station", default="500", help="MPC observatory code (default 500)")
    f.add_argument("--mag-limit", type=float, default=None, help="faintest magnitude shown")
    common(f)

    i = sub.add_parser("identify", help="the known object each tracklet of a file is")
    i.add_argument("file", help="MPC 80-column or ADES (PSV/XML) astrometry")
    i.add_argument("--match", type=float, default=skycheck.MATCH_ARCSEC,
                   help="mean offset within which a candidate matches, arcsec "
                        "(default %(default)s)")
    i.add_argument("--motion", type=float, default=skycheck.MOTION_ARCSEC,
                   help="rms about that offset within which the motion agrees, arcsec "
                        "(default %(default)s)")
    i.add_argument("--all", action="store_true", help="list every candidate, not the best 8")
    common(i)

    sub.add_parser("update", help="download the MPC's orbit files and rebuild the catalogue")
    args = ap.parse_args(argv)

    if args.cmd == "update":
        from pyroorb.skycheck import catalogue
        cat = catalogue.load(refresh=True, verbose=True)
        print(f"{len(cat):,} orbits, epoch JD {cat.meta.get('main_epoch')}")
        return 0
    if args.cmd == "field":
        res = skycheck.check_field(args.ra, args.dec, args.time, radius_arcmin=args.radius,
                                   obscode=args.station, mag_limit=args.mag_limit,
                                   comets=not args.no_comets, neocp=args.neocp,
                                   refresh=not args.no_refresh)
        if args.json:
            print(json.dumps(_jsonable({"time_utc": res.time_utc, "ra": res.ra, "dec": res.dec,
                                        "radius_arcmin": res.radius_arcmin,
                                        "obscode": res.obscode, "objects": res.objects,
                                        "notes": res.notes, "catalogue": res.catalogue}),
                             indent=1))
        else:
            print(res.table())
        return 0
    res = skycheck.identify(args.file, radius_arcmin=args.radius, match_arcsec=args.match,
                            motion_arcsec=args.motion, comets=not args.no_comets,
                            neocp=args.neocp, refresh=not args.no_refresh)
    if args.json:
        print(json.dumps(_jsonable([tr.__dict__ for tr in res]), indent=1))
        return 0
    n_id = sum(1 for tr in res if tr.identification)
    print(f"{len(res)} tracklet(s), {n_id} identified")
    for tr in res:
        print(tr.table(limit=None if args.all else 8))
    return 0


if __name__ == "__main__":
    from pyroorb.envfile import load_env_file
    load_env_file()
    from pyroorb.console import utf8_stdio
    utf8_stdio()
    sys.exit(main())
