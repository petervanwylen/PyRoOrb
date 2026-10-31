"""Command-line interface, matching the reference ``digest2`` program's
options, config-file semantics, and output formatting (``d2cli.c`` +
``digest2.c: setup()``/``fmtScores()``).

Differences from the reference C program, by design:

* The population model and obscodes table fall back to the copies
  bundled in ``pydigest2/data/`` when not found alongside the input
  (see :mod:`pydigest2.paths`) -- the original requires a local
  ``digest2.model.csv``/``digest2.obscodes`` (auto-downloading only the
  latter). This changes nothing about *how a tracklet is scored*, only
  where the files are found when the caller didn't say.
* ``-m``'s "compile a fast-loading cache" step writes this package's
  own ``.npz`` cache format (see :mod:`pydigest2.model`), not the
  original's raw-``fwrite`` binary layout -- an internal speed
  optimization, not an interchange format, in either implementation.
* A ``.psv`` input file is read as ADES pipe-separated-value (no C
  reference exists for this format; see :mod:`pydigest2.observations`).
* ``--cpu``/``-u`` selects worker threads when the compiled kernel is
  available and worker *processes* otherwise (the interpreted kernel
  holds the GIL, so threads would serialize); see
  :mod:`pydigest2.engine`. As in the reference program, the default is
  every available core.

Tracklets are scored in batches rather than one at a time, so a run can
use several cores without holding a whole input file in memory. Output
order always matches input order, independent of batch size and worker
count -- and so do the scores themselves, since each tracklet carries
its own independent RNG stream.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence, TextIO

from .classes import CLASS_ABBR, CLASS_HEADING
from .config import Config, ConfigError, apply_config_lines, parse_limit_spec
from .engine import (
    Digest2Engine, TrackletProblem, score_many, validate_tracklet,
)
from .intake import InvalidRun, iter_ades_xml_tracklets, iter_mpc80_tracklets
from .model import load_model, save_model_cache
from .obscodes import SiteTable
from .observations import parse_ades_psv
from .output import format_header_lines, format_mpc_desig, format_score_line, strtok_first_token
from .paths import find_model_path, find_obscodes_path

OBSCODES_URL = "https://minorplanetcenter.net/iau/lists/ObsCodes.html"

_HELP_EPILOG = """\
Config file keywords:
   headings
   noheadings
   rms
   rmsPrime
   noThreshold
   norms
   raw
   noid
   repeatable
   random
   poss
   obserr

Orbit classes:
""" + "\n".join(f"   {abbr:<5} {heading}" for abbr, heading in zip(CLASS_ABBR, CLASS_HEADING))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pydigest2",
        usage=(
            "%(prog)s [options] <obs file> [obs file2 ...]   score observations\n"
            "       %(prog)s [options] -             score observations from stdin\n"
            "       %(prog)s -m <cache file>          generate a fast-load cache from the model CSV\n"
            "       %(prog)s -h or --help             display help and quick reference\n"
            "       %(prog)s -v or --version           display program version and model date"
        ),
        description="Statistical ranging classification of short-arc asteroid astrometry.",
        epilog=_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    p.add_argument("-h", "--help", action="help", help="display help and quick reference")
    p.add_argument("-v", "--version", action="store_true", help="display program version and model date")
    p.add_argument("-c", "--config", dest="config", metavar="<config file>")
    p.add_argument("-m", "--model", dest="model", metavar="<model file>")
    p.add_argument("-o", "--obscodes", dest="obscodes", metavar="<obscode file>")
    p.add_argument("-p", "--config-path", dest="config_path", metavar="<path>")
    p.add_argument("-u", "--cpu", dest="cpu", type=int, metavar="<n-cores>")
    p.add_argument("-l", "--limit", dest="limit", metavar="<class>/<score>=<limit>")
    p.add_argument("obs_files", nargs="*", metavar="<obs file>")
    return p


def _joined(explicit: Optional[str], default_name: str, config_dir: Optional[str]) -> Optional[str]:
    if explicit is not None:
        return explicit
    if config_dir is not None:
        return str(Path(config_dir) / default_name)
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.version:
        _print_version(args)
        return 0

    try:
        limit_spec = parse_limit_spec(args.limit) if args.limit else None
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1

    model_path_opt = _joined(args.model, "digest2.model.csv", args.config_path)
    obscodes_path_opt = _joined(args.obscodes, "digest2.obscodes", args.config_path)
    config_path_opt = _joined(args.config, "digest2.config", args.config_path)

    if not args.obs_files:
        # No observation files: -m and/or -o alone perform their standalone
        # actions (generate a cache / fetch obscodes) and exit.
        if args.model is None and args.obscodes is None:
            parser.print_usage(sys.stderr)
            return 1
        if args.model is not None:
            csv_source = _joined(None, "digest2.model.csv", args.config_path) or find_model_path()
            _generate_model_cache(csv_source, model_path_opt)
        if args.obscodes is not None:
            _fetch_obscodes(obscodes_path_opt)
        return 0

    try:
        model_path = model_path_opt or find_model_path()
        obscodes_path = obscodes_path_opt or find_obscodes_path()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    site_table = SiteTable.from_file(obscodes_path)
    cfg = Config()
    try:
        if config_path_opt is not None:
            if not Path(config_path_opt).is_file():
                print(f"Open {config_path_opt} failed.", file=sys.stderr)
                return 1
            apply_config_lines(cfg, Path(config_path_opt).read_text().splitlines(), site_table)
        else:
            default_cfg = Path("digest2.config")
            if default_cfg.is_file():
                apply_config_lines(cfg, default_cfg.read_text().splitlines(), site_table)
        if limit_spec is not None:
            cfg.limit_class, cfg.limit_raw, cfg.limit_value = limit_spec
            cfg.validate_limit()
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        model = load_model(model_path)
    except Exception as e:  # noqa: BLE001 -- surface any load failure as a CLI error
        print(f"Read {model_path} failed: {e}", file=sys.stderr)
        return 1

    engine = Digest2Engine(model, site_table, cfg)

    out = sys.stdout
    for line in format_header_lines(cfg):
        print(line, file=out)

    for fn in args.obs_files:
        _process_file(fn, engine, cfg, out, args.cpu)
    return 0


# Tracklets are scored in batches so that a run can use several cores
# (see --cpu) without having to hold a whole input file in memory. Output
# stays in input order regardless of batch size or worker count.
_BATCH = 256


def _process_file(fn: str, engine: Digest2Engine, cfg: Config, out: TextIO,
                   n_jobs: Optional[int]) -> None:
    ext = Path(fn).suffix.lower().lstrip(".")
    if fn != "-" and ext == "xml":
        _process_tracklet_stream(iter_ades_xml_tracklets(fn), is_ades=True,
                                  engine=engine, cfg=cfg, out=out, n_jobs=n_jobs)
    elif fn != "-" and ext == "psv":
        batch = [(desig, desig, olist) for desig, olist in parse_ades_psv(fn).items()]
        _flush_batch(batch, is_ades=True, engine=engine, cfg=cfg, out=out, n_jobs=n_jobs)
    elif fn == "-":
        _process_mpc80_lines(sys.stdin, engine, cfg, out, n_jobs)
    else:
        try:
            with open(fn, "r") as f:
                _process_mpc80_lines(f, engine, cfg, out, n_jobs)
        except OSError:
            print(f"Open {fn} failed.", file=sys.stderr)


def _process_mpc80_lines(lines, engine: Digest2Engine, cfg: Config, out: TextIO,
                          n_jobs: Optional[int]) -> None:
    _process_tracklet_stream(iter_mpc80_tracklets(lines), is_ades=False,
                              engine=engine, cfg=cfg, out=out, n_jobs=n_jobs)


def _process_tracklet_stream(stream, *, is_ades: bool, engine: Digest2Engine,
                              cfg: Config, out: TextIO, n_jobs: Optional[int]) -> None:
    batch: List[tuple] = []
    for item in stream:
        if isinstance(item, InvalidRun):
            # flush first so the message lands in the right place in the output
            _flush_batch(batch, is_ades=is_ades, engine=engine, cfg=cfg,
                          out=out, n_jobs=n_jobs)
            print(f"{item.n_lines} lines skipped for .", file=out)
            continue
        display = item.desig if is_ades else format_mpc_desig(item.desig)
        message_desig = item.desig if is_ades else strtok_first_token(item.desig)
        batch.append((display, message_desig, item.observations))
        if len(batch) >= _BATCH:
            _flush_batch(batch, is_ades=is_ades, engine=engine, cfg=cfg,
                          out=out, n_jobs=n_jobs)
    _flush_batch(batch, is_ades=is_ades, engine=engine, cfg=cfg, out=out, n_jobs=n_jobs)


def _flush_batch(batch: List[tuple], *, is_ades: bool, engine: Digest2Engine,
                  cfg: Config, out: TextIO, n_jobs: Optional[int]) -> None:
    """Score a batch of tracklets (in parallel when asked for) and print
    the results in the order they were read."""
    if not batch:
        return

    checked = [(display, message, olist, validate_tracklet(olist))
               for display, message, olist in batch]
    todo = [(display, olist, is_ades)
            for display, _, olist, problem in checked if problem is None]
    # score_many preserves input order, so results zip back positionally
    results = iter(score_many(todo, engine, n_jobs=n_jobs) if todo else ())

    for display, message, _olist, problem in checked:
        if problem is not None:
            print(f"{message} {problem.reason}", file=out)
            continue
        _, result = next(results)
        if isinstance(result, TrackletProblem):
            print(f"{message} {result.reason}", file=out)
            continue
        line = format_score_line(display, result, cfg)
        if line is not None:
            print(line, file=out)
    batch.clear()


def _generate_model_cache(csv_path: str, target_path: str) -> None:
    try:
        model = load_model(csv_path, use_cache=False)
        save_model_cache(model, target_path)
        print(f"Wrote model cache to {target_path}")
    except Exception as e:  # noqa: BLE001
        print(f"Write model cache failed: {e}", file=sys.stderr)


def _fetch_obscodes(target_path: Optional[str]) -> None:
    target = target_path or "digest2.obscodes"
    try:
        with urllib.request.urlopen(OBSCODES_URL, timeout=30) as resp:
            data = resp.read()
        Path(target).write_bytes(data)
        print(f"Wrote {target}")
    except Exception as e:  # noqa: BLE001
        print(f"Cannot access URL {OBSCODES_URL}: {e}", file=sys.stderr)


def _print_version(args) -> None:
    from . import __version__
    print(f"pydigest2 version {__version__}")
    print("Public domain algorithm; Python port.")
    try:
        model_path = _joined(args.model, "digest2.model.csv", args.config_path) or find_model_path()
        st = Path(model_path).stat()
        import time
        print(f"{model_path}: {st.st_size} bytes {time.ctime(st.st_mtime)}")
    except FileNotFoundError:
        print("digest2.model.csv not present.")


if __name__ == "__main__":
    sys.exit(main())
