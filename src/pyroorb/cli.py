"""Command-line / batch front-end for the orbit engine: argument parsing, the
identification-check driver, the multi-object batch runner, the egress glue, and `main`.

This is the user-interface layer -- it orchestrates the engine (fit_object) and the I/O
(obs_sources) and renders results, but contains no orbit-determination logic of its own.
It runs as the `pyroorb` command (or `python -m pyroorb.fit_object`, whose `__main__`
delegates here); the ident-check and batch drivers re-invoke that same module in child
processes (`_child_cmd`).

Imports the engine from fit_object and the data plumbing from obs_sources/guards; nothing
imports cli back at module load time (fit_object pulls it in only inside its main()), so
there is no import cycle.
"""
import os, sys, time
import numpy as np
from astropy.time import Time

import pyroorb.nongrav as _ngm          # SRP1AU + the AMR find_orb reports for its SRP model
from pyroorb.fit_object import _nongrav_asteroids_gate as _ng_ast_gate
from pyroorb.fit_object import (fit_orbit, fetch_mpc_seed, refine_asteroids, refine_nongrav,
                        refit_constrained)
from pyroorb.obs_sources import (load_obscodes, fetch_observations, pop_unknown_obscodes,
                         fetch_combined, ensure_unpacked, read_obs_file, _group_by_designation,
                         fetch_failure_message)
from pyroorb.guards import is_pluto, _pluto_candidates, PLUTO_BAN_MSG
from pyroorb.itf.verdict import SearchVerdict
from pyroorb.obs80 import pack_designation
from pyroorb.orbit_common import OPP_GAP_DAYS
# The ident rule itself lives in the engine module (ident_check), not here: the web UI gates its
# "Submit linkage" button on the SAME verdict this batch prints. A UI offering what --ident-check
# files as a non-match would be the worst kind of divergence. Old names kept for callers/tests.
from pyroorb.ident_check import MPC_MIN_RETENTION, examine_or_no
_examine_or_no = examine_or_no
from pyroorb import presentation

# The ident-check / batch drivers re-invoke the CLI in child processes (subprocess isolation
# with a hard timeout). A child runs the package module, never a file path: running a file from
# inside the package would put the package directory on the child's import path, where every
# module could load a second time under its bare name.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _child_cmd(*args):
    """argv for one child run of the CLI: ``python -m pyroorb.fit_object ARGS``."""
    return [sys.executable, "-m", "pyroorb.fit_object", *args]


def _child_env():
    """The child's environment: the package's parent directory on PYTHONPATH, so a child
    imports the same pyroorb as its parent whether or not the package is installed."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_SRC] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


USAGE = """usage: pyroorb DESIG [DESIG ...] [options]      fit one object (several = one linkage)
       pyroorb --batch FILE [options]                  fit every object in an astrometry file
       pyroorb DESIG1 DESIG2 --ident-check OUT         test a proposed identification

  --force                 experimental brute-force linker (probe a hypothesis only)
  --in-format obs80|ades  force the input format of --batch / --paste files
  --out-format LIST       write mpcorb,sof,elem,covar exports
  --itf-search            search the ITF for unlinked tracklets (needs your own SBN database)
  --mag-audit, --impact, --keyholes    opt-in reports
  --no-radar              skip the radar delay/Doppler refinement

Environment: PYROORB_VERBOSE=1 traces arc extension. See the README for the Python API.
"""


def _opt(args, name, default=None):
    """Pull a `--name value` option out of `args` in place; return its value."""
    if name in args:
        i = args.index(name)
        val = args[i + 1] if i + 1 < len(args) else default
        del args[i:i + 2]
        return val
    return default


def _read_pairs(path):
    """Read a pairs file: one identification candidate per line, two designations
    separated by a TAB, a comma, or a run of 2+ spaces (each designation contains a single
    space, so we split on 2+ whitespace, not 1). Blank lines and `#` comments are ignored."""
    import re
    pairs = []
    with open(path, encoding="utf-8") as fp:
        for ln in fp:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if "\t" in ln:
                parts = ln.split("\t")
            elif "," in ln:
                parts = ln.split(",")
            else:
                parts = re.split(r"\s{2,}", ln)        # two designations split by 2+ spaces
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def _classify_path(out_path):
    """Companion classification-file path for a --ident-check output:
    `idents.txt` -> `idents.classify.txt`."""
    root, ext = os.path.splitext(out_path)
    return root + ".classify" + (ext or ".txt")


def _fetch_ident_pair(desigs, obscodes):
    """Fetch a candidate identification pair for --ident-check.

    Fit on the CANONICAL ADES obs (`fetch_observations`, the same source the CLI / web UI /
    fixtures use), so the ident verdict reflects the fit the engine reports everywhere else.
    ident_check.identify() then echoes the dropped tracklets VERBATIM under `except:`, which
    needs each obs's original 80-col line -- and ADES obs carry none. So we ALSO pull the OBS80
    records and graft `obs80_line` onto each ADES obs.

    Match within each (obscode, packed) group -- NOT on an exact jd key: the OBS80 time field is
    truncated to 5 decimal days (~0.9 s), so its jd disagrees with the ADES obsTime by up to
    ~half a second (2022 MT4's B74 tracklet: 2459706.62584 vs .625845), which an exact key
    misses. Equal-count groups (the norm -- the same obs in two formats) zip by sorted time;
    the rare unequal group falls back to nearest-time within a few seconds. An obs that fails to
    match keeps obs80_line=None (identify() then declines the echo) -- never a fabricated line."""
    ades, meta = fetch_observations(desigs, obscodes)
    if not ades:
        return ades, meta
    if meta.get("format") == "OBS80":
        return ades, meta            # already OBS80 (ADES unavailable) -- lines present, no graft
    try:
        o80, _ = fetch_combined(desigs, obscodes)
    except Exception:
        return ades, meta            # no OBS80 to graft; identify() declines the except: echo
    from collections import defaultdict
    GRAFT_TOL_DAYS = 5.0e-5          # ~4 s: bridges the OBS80 time truncation, under tracklet cadence
    a_groups, o_groups = defaultdict(list), defaultdict(list)
    for o in ades:
        a_groups[(o["obscode"], o.get("packed"))].append(o)
    for o in o80:
        o_groups[(o["obscode"], o.get("packed"))].append(o)
    for key, alist in a_groups.items():
        olist = o_groups.get(key)
        if not olist:
            continue
        alist.sort(key=lambda o: o["jd_utc"])
        olist = sorted(olist, key=lambda o: o["jd_utc"])
        if len(alist) == len(olist):                 # same obs, two formats -> position-match
            for a, b in zip(alist, olist):
                a.setdefault("obs80_line", b.get("obs80_line"))
        else:                                        # rare: an obs in one format only
            for a in alist:
                jd = float(a["jd_utc"])
                b = min(olist, key=lambda o: abs(float(o["jd_utc"]) - jd))
                if abs(float(b["jd_utc"]) - jd) <= GRAFT_TOL_DAYS:
                    a.setdefault("obs80_line", b.get("obs80_line"))
    return ades, meta


def _ident_check_one(d1, d2, out_path, obscodes, force=False):
    """Fit ONE candidate pair in-process and APPEND a clean identification to `out_path`
    (ident_check.identify; the except: block echoes verbatim obs80 lines grafted onto the
    canonical ADES fit -- see _fetch_ident_pair). Prints a one-line verdict to stderr and
    returns the classification code. This is the unit the batch runner shells out to -- one
    subprocess per pair -- so a single pathological fit can be SIGKILLed without stalling the
    batch (a spin inside the integrator can't be broken by a Python-level signal).

    The pair is fitted on the CANONICAL ADES-weighted obs (the same source the CLI / web UI /
    fixtures use). An over-confident survey sigma that would spuriously clip a whole arc is now
    handled inside the fit by the sigma.txt reject-rate relaxation
    (arc_extend._relax_overconfident_obscodes), so the verdict no longer needs the old
    OBS80-weighting fallback -- 2015 KR274 = 2025 PY146 links 49/55 on ADES directly, and
    2025 KV14 = 2025 NH520 stays a clean y2."""
    from pyroorb import ident_check
    # Accept packed OR unpacked designations on either side (the --pairs file and the
    # positional CLI both flow through here); normalize ONLY for the fetch so the verdict
    # and classify rows still echo the tokens exactly as supplied.
    obs, meta = _fetch_ident_pair([ensure_unpacked(d1), ensure_unpacked(d2)], obscodes)
    # NOT a non-match: a verdict computed on a half-fetched pair (one side's fetch timed
    # out, rows for the other arrived) would file a confident "no" on the strength of a
    # dropped connection -- and a "no" here is a durable classification row. The distinct
    # code keeps the outage out of the identification file, counts as a skip (not a
    # match) to every reader, and marks the row as re-runnable.
    _fmsg = fetch_failure_message(meta)
    if _fmsg:
        print(f"[ident-check] {d1} + {d2}: fetch failed -- {_fmsg}", file=sys.stderr)
        print("[ident-check-class] no-fetch", file=sys.stderr); return "no-fetch"
    if not obs:
        print(f"[ident-check] {d1} + {d2}: no observations", file=sys.stderr)
        print("[ident-check-class] no", file=sys.stderr); return "no"
    fit = fit_orbit(obs, obscodes, desig=meta["designation"], meta=meta, verbose=False, force=force)
    text = ident_check.identify(fit)
    stat = (f"{fit.n_observations()}/{fit.n_total_observations()} obs, "
            f"{fit.n_excluded()} dropped, rms {fit.rms_residual():.2f}\"")
    if text:
        # SYSTEMATIC-residual audit (one cheap pass over the residuals the fit already produced --
        # no per-object re-fit): a link whose kept obs fit at an acceptable OVERALL rms but show a
        # per-apparition residual TREND is two arcs the main linker forced together (e.g. 2014 HV328
        # = 2025 OV408 at 0.50"). Downgrade to 'y-systematic' -- "be skeptical" -- and do NOT write
        # it as a confirmed identification; a human reviews it (it may still be real, like the
        # genuine 2025 OT173 = 2014 EN155 whose 2014 nights trend).
        # ...unless BOTH sides are SHORT arcs, in which case the trend is decisive rather than
        # suspicious: with no curvature on either side there is nothing for a reviewer to weigh
        # the systematic against, a nongrav large enough to walk the residuals would have been
        # auto-detected, and the joint orbit plainly does not thread both arcs -- a flat 'no'
        # (ident_check.short_arc_systematic; 2025 OC732 + 2014 HJ409).
        fatal_sys = ident_check.short_arc_systematic(fit)
        sysdesc = None if fatal_sys else ident_check.systematic_misfit(fit)
        if fatal_sys:
            code = "no"
            print(f"[ident-check] {d1} + {d2}: no clean match ({stat}; {fatal_sys})",
                  file=sys.stderr)
        elif sysdesc:
            code = "y-systematic"
            print(f"[ident-check] {d1} + {d2}: SYSTEMATIC residual ({stat}; {sysdesc}) -- not "
                  f"auto-confirmed [{code}]", file=sys.stderr)
        else:
            # classify by the number of EXCEPTED obs (the except-block lines -- the header
            # carries ' = ', a photometric FLAG starts with '#'): 'y' = clean / <=1 excepted,
            # 'y2' = needs tracklet exception(s). identify() lists only wholly-dropped
            # tracklets with every obs > EXCEPT_MIN_RESID_ARCSEC; isolated outliers the fit
            # excludes are bad OBSERVATIONS, not exceptions, and don't push 'y' to 'y2'.
            # ...counted here from the rendered record, but the y/y2 THRESHOLD lives in
            # ident_check (EXCEPT_CLEAN_MAX_OBS), so the web UI's y2 caution splits at the same
            # place this code does (ident_check.exception_shape / linkage_verdict).
            n_drops = sum(1 for ln in text.splitlines() if " = " not in ln and not ln.startswith("#"))
            code = "y" if n_drops <= ident_check.EXCEPT_CLEAN_MAX_OBS else "y2"
            # THIN-COLLAPSED-ARC false-positive flag (advisory): the drops make the code look like a
            # clean y/y2, but if the match rests on ONE designation collapsed to a single sub-day
            # tracklet -- its own longer arc gutted -- the link isn't independently confirmed on that
            # side and is more likely a false positive than a real ID. Never a rejection: still
            # emitted, but the classify code gains a '-falsepos?' suffix and the written record a
            # '# FLAG' comment so a human looks before submitting. (ident_check.thin_arc_falsepos.)
            fpdesc = ident_check.thin_arc_falsepos(fit)
            if fpdesc:
                code = code + "-falsepos?"
                text = (f"# FLAG false-positive risk: {fpdesc} -- examine before submitting\n"
                        + text)
                print(f"[ident-check] {d1} + {d2}: MATCH but FALSE-POSITIVE RISK ({stat}; {fpdesc}) "
                      f"[{code}]", file=sys.stderr)
            else:
                print(f"[ident-check] {d1} + {d2}: MATCH ({stat}) [{code}]", file=sys.stderr)
            with open(out_path, "a", encoding="utf-8") as out:
                out.write(text + "\n"); out.flush()
    else:
        # Not a CLEAN identification -> AMBIGUOUS link worth a human look ('y-examine') or a clear
        # non-match ('no'). The decision (retention / short+short / baseline) lives in the pure
        # _examine_or_no helper so it is unit-testable offline.
        code, reason = _examine_or_no(fit)
        label = "EXAMINE" if code == "y-examine" else "no clean match"
        tag = f" [{code}]" if code != "no" else ""
        if code == "y-examine":
            # A 'y-examine' reason is the grounds for NOT calling it a non-match -- an exculpation
            # -- so print it as that, beside the defect identify() actually refused on. Printed
            # bare, it read as the doubt ("EXAMINE (...; a baseline that long pins the orbit)").
            why = {}
            ident_check.exception_shape(fit, _why=why)
            refused = f"refused: {why['reason']}; " if why.get("reason") else ""
            print(f"[ident-check] {d1} + {d2}: {label} ({stat}; {refused}kept as a candidate "
                  f"because {reason}){tag}", file=sys.stderr)
        else:
            print(f"[ident-check] {d1} + {d2}: {label} ({stat}; {reason}){tag}", file=sys.stderr)
    # SHORT-LINK suffix: an identification carried by fewer than ident_check.THIN_LINK_MIN_OBS
    # observations is barely over-determined -- every other gate judges what the fit DROPPED, so a
    # tiny link that drops nothing passes them all on almost no evidence. Advisory ONLY (still
    # written, still a y/y2), but the code says '-short' so a bulk scan can pull those out for a
    # human look. (J99K17K = K16L89L: an immaculate-looking 9/9 @ 0.56" 'y'.)
    if code != "no":
        thin = ident_check.thin_link(fit)
        if thin:
            code = code + "-short"
            print(f"[ident-check] {d1} + {d2}: SHORT link ({thin}) -> [{code}]", file=sys.stderr)
    # Photometric mismatch suffix: a cross-opposition implied-H disagreement (the `_mag_stats` dH x p
    # grid) marks an otherwise-positive classification for a closer look -- 'y'/'y2'/'y-examine' gain
    # a suffix by tier: '-mag?' (advisory), '-mag' (flag), '-mag!' (strong). Advisory ONLY (we never
    # reject on photometry -- some survey mags are junk), but a brightness disagreement shouldn't hide
    # behind a clean code. (2016 LW90 = 2002 GX31: a tight 0.27" link but implied H~7.2 vs H~9.0 ->
    # 'y-mag'.) 'no' is already a non-match, so it is left alone.
    if code != "no":
        st = ident_check._mag_stats(fit)
        if st and st["tier"] is not None:
            code = code + "-mag" + st["tier"]
            print(f"[ident-check] {d1} + {d2}: photometric FLAG dH={st['dH']:.2f} mag "
                  f"p={st['p']:.1g} -> [{code}]", file=sys.stderr)
    print(f"[ident-check-class] {code}", file=sys.stderr)
    return code


def _is_written_match(code):
    """Is this classify code one of the identifications actually WRITTEN to the file -- a 'y'/'y2'
    carrying any advisory suffix ('-falsepos?', '-short', '-mag?'/'-mag'/'-mag!')?

    'y-examine' and 'y-systematic' also begin with 'y' but are NOT written (they are flagged for a
    human), so they are excluded by name. The summary counters key on this rather than on an exact
    'y'/'y2', which counted every advisory-suffixed match as a non-match."""
    return code.split("-")[0] in ("y", "y2") and not code.startswith(("y-examine", "y-systematic"))


def run_ident_check(pairs, out_path, obscodes, force=False, pair_timeout=120):
    """Batch driver: fit each candidate pair and append clean identifications to `out_path`.

    Each pair runs in its OWN subprocess with a hard `pair_timeout` (seconds). A bogus /
    chimera pair can spin forever inside an uninterruptible integrator call -- which a
    Python signal can't break -- so we isolate it in a child process and SIGKILL it on
    timeout. A timed-out pair is, by definition, not a clean identification, so it is
    skipped. `pair_timeout=0` runs everything in-process with no timeout.

    Also writes a companion CLASSIFICATION file next to `out_path` -- one row per pair, in
    input order: `d1<TAB>d2<TAB>code<TAB>secs` (secs = wall-clock for that pair, so a bulk scan
    can surface the slow-non-match tail without re-timing):
      * `no`        -- not an identification;
      * `y`         -- a clean match (at most one dropped obs);
      * `y2`        -- a match needing a clean tracklet exception or two;
      * `y-examine` -- NOT a clean match, but the joint fit retains a strong majority
                       (>= MPC_MIN_RETENTION) of BOTH designations' obs and fits sub-arcsec, so
                       the rejections just aren't a clean tracklet drop -- worth a human look.
                       (A chimera linked only by discarding the bulk of one arc is a 'no'.)
      * `y-systematic` -- the kept obs fit at an acceptable OVERALL rms but a per-apparition
                       residual TREND betrays two arcs forced together (ident_check.systematic_misfit):
                       be skeptical -- a human reviews it; not auto-confirmed. When BOTH sides are
                       SHORT arcs (ident_check.short_arc_systematic) there is nothing for a
                       reviewer to weigh the trend against, and it is a plain `no` instead.
    A `y`/`y2` may carry advisory suffixes (still written, flagged for review): `-falsepos?` when
    the match rests on ONE side collapsed to a single sub-day tracklet (ident_check.thin_arc_falsepos
    -- more likely a false positive than a real ID), `-short` when fewer than
    ident_check.THIN_LINK_MIN_OBS obs support the joint orbit at all (barely over-determined, so
    the fit tests the identification hardly at all -- ident_check.thin_link), and/or
    `-mag?`/`-mag`/`-mag!` for a cross-opposition brightness mismatch.
    Only `y` and `y2` (with any advisory suffix) are written to the identifications file;
    `y-examine`/`y-systematic` are flagged here only."""
    import subprocess
    classify_path = _classify_path(out_path)

    def _write_class(d1, d2, code, secs):
        # Columns: d1 <TAB> d2 <TAB> code <TAB> wall-seconds. The `secs` column lets a bulk scan
        # spot the pairs that ate the most time (the slow-non-match tail) without re-timing; the
        # pairs reader only consumes columns 0-1, so the extra field is backward-compatible.
        with open(classify_path, "a", encoding="utf-8") as cf:
            cf.write(f"{d1}\t{d2}\t{code}\t{secs:.1f}\n"); cf.flush()

    if not pair_timeout:                               # no-timeout: plain in-process loop
        for d1, d2 in pairs:
            t0 = time.perf_counter()
            try:
                code = _ident_check_one(d1, d2, out_path, obscodes, force=force) or "no"
            except Exception as ex:
                print(f"[ident-check] {d1} + {d2}: error {str(ex)[:70]}", file=sys.stderr); code = "no"
            _write_class(d1, d2, code, time.perf_counter() - t0)
        print(f"[ident-check] done -> {out_path}  (classification -> {classify_path})", file=sys.stderr)
        return
    n_match = n_examine = n_skip = n_timeout = n_err = 0
    for d1, d2 in pairs:
        cmd = _child_cmd(d1, d2, "--ident-check", out_path) + (["--force"] if force else [])
        t0 = time.perf_counter()
        try:
            r = subprocess.run(cmd, timeout=pair_timeout, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               env=_child_env())
            verdict = next((l for l in r.stderr.splitlines() if l.startswith("[ident-check]")), None)
            code = next((l.split()[-1] for l in r.stderr.splitlines()
                         if l.startswith("[ident-check-class]")), "no")
            if _is_written_match(code):
                n_match += 1
            elif code.startswith("y-examine"):
                n_examine += 1
            else:
                n_skip += 1
            print(verdict or f"[ident-check] {d1} + {d2}: {code}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            n_timeout += 1; code = "no"
            print(f"[ident-check] {d1} + {d2}: TIMEOUT (>{int(pair_timeout)}s) -- skipped (bogus pair?)",
                  file=sys.stderr)
        except Exception as ex:
            n_err += 1; code = "no"
            print(f"[ident-check] {d1} + {d2}: error {str(ex)[:70]}", file=sys.stderr)
        _write_class(d1, d2, code, time.perf_counter() - t0)
    print(f"[ident-check] done: {n_match} identified, {n_examine} examine, {n_skip} no-match, "
          f"{n_timeout} timeout, {n_err} errors -> {out_path}  (classification -> {classify_path})",
          file=sys.stderr)


def _jyear(jd):
    """Calendar year of a JD (for labelling an apparition), e.g. 2013.7 -> 2013."""
    return int(Time(float(jd), format="jd").decimalyear)


def run_mag_audit(desigs, obscodes, force=False, out=None):
    """Audit each SINGLE designation for a significant implied-H change BETWEEN its apparitions
    (ident_check.opposition_mag_audit). A single body shares one absolute magnitude across its
    oppositions, so an apparition that sits well off the others is worth a look -- a bad-photometry
    apparition, an unrecognised contaminant tracklet folded into the arc, or a genuine brightness
    change. One report line per object; those whose apparitions disagree are flagged
    [MAG?]/[MAG]/[MAG!] by tier. Advisory ONLY -- it fits each object and inspects the per-apparition
    brightness, never alters an orbit. Returns the number of objects flagged."""
    from pyroorb import ident_check
    out = out or sys.stdout
    tier_lbl = {"?": "MAG?", "": "MAG", "!": "MAG!"}
    n_flag = 0
    for desig in desigs:
        try:
            obs, meta = fetch_observations([ensure_unpacked(desig)], obscodes)
        except Exception as ex:
            print(f"[mag-audit] {desig}: fetch error {str(ex)[:60]}", file=out); continue
        _fmsg = fetch_failure_message(meta)
        if _fmsg:                # an outage is not "no observations" (see main())
            print(f"[mag-audit] {desig}: fetch failed -- {str(_fmsg)[:100]}", file=out); continue
        if not obs:
            print(f"[mag-audit] {desig}: no observations", file=out); continue
        try:
            fit = fit_orbit(obs, obscodes, desig=meta["designation"], meta=meta,
                            verbose=False, force=force)
        except Exception as ex:
            print(f"[mag-audit] {desig}: fit error {str(ex)[:60]}", file=out); continue
        name = meta.get("designation", desig)
        result = ident_check.opposition_mag_audit(fit)
        if result is None:
            print(f"[mag-audit] {name}: <2 apparitions with usable photometry -- skipped", file=out)
            continue
        apps, w = result["apps"], result["worst"]
        appstr = ", ".join(f"{_jyear(a['jd0'])} H~{a['H']:.1f}(n{a['n']})" for a in apps)
        flagged = result["tier"] is not None
        flag = f"[{tier_lbl[result['tier']]}]" if flagged else "[ok]"
        n_flag += int(flagged)
        print(f"[mag-audit] {name}: {len(apps)} app  worst dH={w['dH']:.2f} p={w['p']:.1g} "
              f"{flag}  ({appstr})", file=out)
    print(f"[mag-audit] done: {n_flag}/{len(desigs)} flagged", file=out)
    return n_flag


def run_batch(text, obscodes, force=False, only=None, verbose=False, in_format=None):
    """Fit EVERY designation in an astrometry blob independently (in-process).

    Identification-agnostic: obs are grouped by the designation each carries, and
    each group is fit on its own (no attempt to link different designations -- that
    needs a separate identifications file). Returns (fits, failures, format_label)
    where `fits` is a list of FittedOrbit (input-designation order) and `failures`
    is a list of (designation, error) for groups that could not be fit. `only`
    restricts the run to a single designation (packed or unpacked)."""
    radar_rows = []
    obs, fmt = read_obs_file(text, obscodes, in_format=in_format, radar_out=radar_rows)
    groups = _group_by_designation(obs)
    if only is not None:
        want = {only.strip(), ensure_unpacked(only)}
        groups = {d: g for d, g in groups.items()
                  if d in want or pack_designation(d) in want}
    fits, failures = [], []
    for d, g in groups.items():
        meta = {"designation": d, "designations": [d], "combined": False,
                "other_designations": [], "n_requested": 1, "format": fmt}
        rd = [r for r in radar_rows if r.get("desig") == d]
        if rd:
            meta["radar"] = rd     # the file's radar rows ride the matching group's fit
        try:
            f = fit_orbit(g, obscodes, desig=d, meta=meta, verbose=verbose, force=force)
            fits.append(f)
            print(f"[batch] {d}: {f.n_observations()}/{f.n_total_observations()} obs, "
                  f"{f.n_oppositions()} opp, rms {f.rms_residual():.3f}\"", file=sys.stderr)
        except Exception as ex:
            failures.append((d, f"{type(ex).__name__}: {str(ex)[:70]}"))
            print(f"[batch] {d}: FIT FAILED ({type(ex).__name__}: {str(ex)[:70]})",
                  file=sys.stderr)
    return fits, failures, fmt


_EGRESS_ALIASES = {"mpcorb": "mpcorb", "sof": "sof", "elem": "elem",
                   "elements": "elem", "8line": "elem", "eightline": "elem",
                   "8-line": "elem", "covar": "covar", "covar.json": "covar",
                   "covariance": "covar"}

# find_orb's covar.json describes ONE object -- a state vector and its parameter
# covariance -- and there is no concatenation of two of them that is still the format.
# So it is a single-object egress only; --batch refuses it by name rather than writing
# a file that looks like JSON and isn't.
_SINGLE_OBJECT_FORMATS = ("covar",)


def _parse_egress_formats(spec):
    """Split a comma-separated --out-format spec into canonical names
    (mpcorb / sof / elem / covar), accepting common aliases (8line, elements, ...)."""
    return [_EGRESS_ALIASES.get(f.strip().lower(), f.strip().lower())
            for f in spec.split(",") if f.strip()]


def _rows_for_fit(fit, fmts, written_jd):
    """Per-object egress rows for the requested formats (no headers): {fmt: rowstr}."""
    from pyroorb import egress
    row = {}
    if "mpcorb" in fmts:
        try:
            row["mpcorb"] = egress.to_mpcorb(fit, written_jd=written_jd)
        except ValueError as ex:
            if "no MPCORB form" not in str(ex):
                raise
            # An unbound orbit has no MPCORB line (find_orb writes none): skip the row, keep the
            # object's other formats, and say why rather than fail the whole batch.
            print(f"{fit.designation}: MPCORB row skipped -- {ex}", file=sys.stderr)
    if "sof" in fmts:
        row["sof"] = egress.to_sof(fit, with_header=False, written_jd=written_jd)
    if "elem" in fmts:
        row["elem"] = egress.to_eightline(fit)
    if "covar" in fmts:
        row["covar"] = egress.to_covar_json(fit)
    return row


def _assemble_batch_output(fmts, ordered_rows):
    """Assemble the final multi-object text: one header per format + its data rows.
    MPCORB is a plain row list; SOF is its self-describing template + records; the
    8-line element block has no shared header, so blocks are blank-line separated."""
    from pyroorb import egress
    chunks = []
    for fmt_name in fmts:
        if fmt_name == "mpcorb":
            rows = [r["mpcorb"] for r in ordered_rows if r.get("mpcorb")]
            chunks.append("\n".join([egress.MPCORB_HEADER] + rows))
        elif fmt_name == "sof":
            rows = [r["sof"] for r in ordered_rows if r.get("sof")]
            chunks.append("\n".join([egress.SOF_TEMPLATE] + rows))
        elif fmt_name == "elem":
            blocks = [r["elem"] for r in ordered_rows if r.get("elem")]
            chunks.append("\n\n".join(blocks))
        elif fmt_name == "covar":
            blocks = [r["covar"] for r in ordered_rows if r.get("covar")]
            chunks.append("\n\n".join(blocks))       # one object; see _SINGLE_OBJECT_FORMATS
        else:
            print(f"(unknown --out-format {fmt_name!r}; expected mpcorb, sof, elem or covar)",
                  file=sys.stderr)
    return "\n".join(c for c in chunks if c)


def run_batch_cli(text, batch_file, obscodes, fmts, out_file, in_format,
                  force, only, batch_timeout, emit_json, verbose):
    """Drive `--batch`: fit each designation, emit one MPCORB/SOF row per object.

    Three roles: (1) the JSON-emitting CHILD (`--batch-emit-json`, used by the
    isolated parent); (2) the isolated PARENT (`--batch-timeout`>0), one subprocess
    per designation hard-killed on timeout so a single hanging object can't stall the
    batch; (3) the plain IN-PROCESS run (default)."""
    from pyroorb import egress; import json as _json
    written_jd = egress._now_jd()

    # (1) CHILD: fit the requested designation(s), print one JSON row per fit.
    if emit_json:
        fits, _failures, _fmt = run_batch(text, obscodes, force=force, only=only,
                                          verbose=verbose, in_format=in_format)
        for f in fits:
            print(_json.dumps({"designation": f.designation,
                               "rows": _rows_for_fit(f, fmts, written_jd)}))
        return

    # (2) ISOLATED PARENT: subprocess per designation with a hard timeout.
    if batch_timeout > 0 and not only:
        obs, _fmt = read_obs_file(text, obscodes, in_format=in_format)
        designations = list(_group_by_designation(obs).keys())
        import subprocess
        ordered_rows, n_ok, n_to, n_err = [], 0, 0, 0
        for d in designations:
            cmd = _child_cmd("--batch", batch_file, "--batch-one", d,
                             "--batch-emit-json", "--out-format", ",".join(fmts))
            if force:
                cmd.append("--force")
            if in_format:
                cmd += ["--in-format", in_format]
            try:
                r = subprocess.run(cmd, timeout=batch_timeout, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace",
                                   env=_child_env())
                got = False
                for ln in r.stdout.splitlines():
                    ln = ln.strip()
                    if not ln.startswith("{"):
                        continue
                    try:
                        obj = _json.loads(ln)
                    except Exception:
                        continue
                    ordered_rows.append(obj["rows"]); got = True
                if got:
                    n_ok += 1
                    print(f"[batch] {d}: fit", file=sys.stderr)
                else:
                    n_err += 1
                    print(f"[batch] {d}: no orbit (fit failed in child)", file=sys.stderr)
            except subprocess.TimeoutExpired:
                n_to += 1
                print(f"[batch] {d}: TIMEOUT (>{int(batch_timeout)}s) -- skipped", file=sys.stderr)
            except Exception as ex:
                n_err += 1
                print(f"[batch] {d}: error {str(ex)[:70]}", file=sys.stderr)
        print(f"[batch] done: {n_ok} fit, {n_to} timeout, {n_err} failed", file=sys.stderr)
    else:
        # (3) IN-PROCESS.
        fits, failures, _fmt = run_batch(text, obscodes, force=force, only=only,
                                         verbose=verbose, in_format=in_format)
        ordered_rows = [_rows_for_fit(f, fmts, written_jd) for f in fits]
        print(f"[batch] done: {len(fits)} fit, {len(failures)} failed", file=sys.stderr)

    output = _assemble_batch_output(fmts, ordered_rows)
    if out_file:
        with open(out_file, "w", encoding="utf-8") as fp:
            fp.write(output + "\n")
        print(f"[batch] wrote {','.join(fmts)} ({len(ordered_rows)} object(s)) -> {out_file}",
              file=sys.stderr)
    elif output:
        print(output)


_LD_KM = 384400.0   # one lunar distance, for human-scale miss distances


def _print_impact(fit, window_days):
    """Phase-1 impact LOCATION report: propagate the fitted orbit forward and
    say WHERE/WHEN it reaches the ground, if it does. Wraps `impact.find_impact` (validated to
    ~km on 2008 TC3) -- the CLI twin of the web Impact pane. A non-impacting object prints its
    explicit miss distance; a crossing is never fabricated."""
    from pyroorb import impact
    result = impact.find_impact(fit, window_days=window_days)
    if result.get("impact"):
        print(f"\nimpact   : ⚠ PREDICTED GROUND IMPACT  (forward {window_days:.0f}-day window)")
        for c in result.get("crossings", []):
            tag = f"@{c['altitude_km']:.0f} km"
            print(f"  {tag:<9}: {c['time_iso'][:23]} UTC  lat {c['lat_deg']:+.3f}°  "
                  f"lon {c['lon_deg']:+.3f}°  v={c['geo_speed_km_s']:.1f} km/s  "
                  f"slope {c['entry_slope_deg']:.1f}°")
        if result.get("ca_min_altitude_km") is not None:
            print(f"  perigee  : {result['ca_min_altitude_km']:.0f} km altitude at "
                  f"{result.get('ca_time_iso', '?')[:19]} UTC")
        fp = impact.impact_footprint(fit, window_days=window_days)
        if fp.get("ok"):
            sl = fp["semilinear"]
            print(f"  1-σ region: {2*sl['semi_major_km']:.2f} × {2*sl['semi_minor_km']:.2f} km "
                  f"(semilinear), long axis bearing {sl['azimuth_deg']:.0f}° at the 100 km entry")
        return
    print(f"\nimpact   : no Earth impact  (forward {window_days:.0f}-day window)")
    reason = result.get("reason", "")
    if "ca_geocentric_km" in result:
        miss = result["ca_geocentric_km"]
        print(f"  {reason}: closest approach {miss:,.0f} km ({miss / _LD_KM:.1f} LD), "
              f"perigee altitude {result.get('ca_min_altitude_km', float('nan')):.0f} km "
              f"at {result.get('ca_time_iso', '?')[:19]} UTC")
    elif result.get("min_geocentric_km") is not None:
        miss = result["min_geocentric_km"]
        print(f"  {reason}: nearest {miss:,.0f} km ({miss / _LD_KM:.1f} LD) in window")
    else:
        print(f"  {reason}")


def _print_one_encounter(e, date=None):
    """One b-plane encounter result (shared by the encounter- and scan-mode IP reports)."""
    when = (date or e.get("ca_time_iso") or "?")[:19]
    k = e.get("closest_approach_sigma")
    if e.get("resolved"):
        ip, v = e.get("ip"), e.get("impact_velocity_km_s")
        odds = f" (~1 in {1.0 / ip:,.0f})" if ip and ip > 0 else ""
        print(f"  {when} UTC: IP = {ip:.2e}{odds}"
              + (f", v_impact {v:.1f} km/s" if v else "")
              + (f", closest on LOV {k:+.2f}σ" if k is not None else ""))
    else:
        line = f"  {when} UTC: no resolved impact"
        if e.get("ip_upper"):
            line += f", IP < {e['ip_upper']:.1e}"
        if e.get("note"):
            line += f"  ({e['note']})"
        print(line)


def _print_impact_probability(fit, window_days, encounter_jd):
    """EXPERIMENTAL Phase-2 impact PROBABILITY report: the LOV b-plane IP
    (`impact.impact_probability`). Scans `window_days` forward from today, or targets one
    `encounter_jd`. A research prototype -- prints the same caveat and authority pointers the web
    pane carries, and REFUSES (rather than guess) on any orbit too poorly constrained for the
    method (U>=5): a 500-object JPL-Sentry validation found our IP never matches within
    +-20% in that regime. Only a well-pinned orbit (U<5, e.g. the certain impactor 2008 TC3)
    reports a number."""
    from pyroorb import impact
    result = impact.impact_probability(fit, window_days=window_days, encounter_jd=encounter_jd)
    print("\nimpact-IP: EXPERIMENTAL Line-of-Variations b-plane probability")
    print("           research prototype -- for operational hazard assessment defer to "
          "JPL CNEOS Sentry\n           (cneos.jpl.nasa.gov/sentry) / ESA NEODyS "
          "(neo.ssa.esa.int)")
    if not result.get("ok", True):
        print(f"  not assessable: {result.get('reason', '')}")
        return
    if result.get("assessed") is False:
        print(f"  {result.get('confidence_note', 'too poorly constrained to assess')}")
        return
    if result.get("mode") == "encounter":
        _print_one_encounter(result)
        return
    for e in result.get("encounters", []):
        _print_one_encounter(e, date=e.get("approach_date"))
    ip = result.get("ip")
    if result.get("resolved") and ip is not None:
        odds = f" (~1 in {1.0 / ip:,.0f})" if ip > 0 else ""
        print(f"  TOTAL over searched window: IP = {ip:.2e}{odds}")
    if result.get("note"):
        print(f"  {result['note']}")


def _print_planetocentric(fit, center):
    """Planetocentric ORBIT report: the fitted object is asserted to orbit
    `center` (a giant planet). Reframes the converged heliocentric state onto the planet and
    prints classical planetocentric elements + a satellite class (planetocentric.report)."""
    from pyroorb import planetocentric
    print()
    print(planetocentric.report(fit, center))


def _print_linkage(fit, do_submit=False):
    """--linkage: report the MPC IDENTIFICATION implied by a joint fit of several designations
    (`fit_object.py "1999 TV43" OOn0024 --linkage`), and with --linkage-submit transmit it.

    The document, the verdict that gates it and the transmission path are the ones behind the web
    UI's "Submit linkage" button -- ident_check.linkage_verdict decides whether this is a candidate
    at all, itf.linkage.build_linkage assembles it (filing NEOCP/ITF temporary ids under `trksubs`
    and stamping identification_type "neocp"), and itf.linkage.submit_linkage does the pre-submit
    DES=DES dedup and the ledger write. Two designations typed at the CLI and two typed into the
    web search box must not disagree about what they imply."""
    from pyroorb import ident_check
    from pyroorb.itf.config import ItfConfig
    from pyroorb.itf.linkage import (linkage_from_fit, submit_linkage,
                             comet_refusal, included_obs80)
    verdict = ident_check.linkage_verdict(fit)
    print("\nlinkage  : MPC identification implied by this joint fit")
    if not verdict["offer"]:
        print(f"  NOT offered: {verdict['reason']}")
        print("  (the orbit may be fine; the IDENTIFICATION is what the engine declines to back)")
        return
    # Said UNPREFIXED: "no linkage" would misstate a comet, which IS a linkage --
    # just not one the MPC's identification pipeline can carry.  One home for the
    # sentence (itf.linkage.comet_refusal), one wording on every surface.
    comet_why = comet_refusal("\n".join(included_obs80(fit)) + "\n")
    if comet_why:
        print(f"  NOT submittable: {comet_why}")
        return
    doc, err = linkage_from_fit(fit, ItfConfig(submit=do_submit))
    if doc is None:
        print(f"  no linkage: {err}")
        return
    print(f"  {'clean identification' if verdict['clean'] else verdict['reason']}")
    for c in verdict["cautions"]:
        print(f"  CAUTION: {c}")
    if not verdict.get("submittable", True):
        # Reported, never sent -- the same rule that leaves the web UI's button down. The CAUSE
        # is the verdict's to state, not this line's: it used to assert "an identification with
        # exceptions", which is true of a 'y2' and false of the other two ways to land here (a
        # trending link, and a shape the engine could not measure at all). Telling someone their
        # link has exceptions, and sending them to Jira for it, when what actually happened is
        # that a check crashed, is worse than saying nothing.
        why = {"y2": "an identification with exceptions cannot be sent this way -- take it to "
                     "the MPC as a Jira ticket",
               "y-systematic": "the engine does not transmit an identification whose own "
                               "residuals trend",
               "unknown": "the engine could not determine whether this identification needs "
                          "exceptions, so it will not put its name to it"}
        print("  NOT transmitted: "
              + why.get(verdict.get("code"), "the engine does not back this identification"))
        return
    if do_submit and verdict["cautions"]:
        print("  -- transmitting anyway (--linkage-submit was given explicitly)")
    submit_linkage(doc, ItfConfig(submit=do_submit), do_submit=do_submit)


def _print_keyholes(fit, years):
    """EXPERIMENTAL Phase-3 KEYHOLE report: resonant-return keyholes on the
    deepest forward Earth encounter (`keyholes.find_keyholes`). Locates which keyholes the
    Line-of-Variations threads (resonance, return year, b-plane width) and the impact
    probability through each. Single-deflector only -- a research prototype; for operational
    keyhole monitoring defer to JPL Sentry / ESA NEODyS."""
    from pyroorb import keyholes
    r = keyholes.find_keyholes(fit, years=years)
    print("\nkeyholes : EXPERIMENTAL resonant-return keyhole analysis (Valsecchi b-plane)")
    print("           research prototype, single deflecting encounter -- for operational keyhole\n"
          "           monitoring defer to JPL CNEOS Sentry / ESA NEODyS")
    if not r.get("ok", True):
        print(f"  not assessable: {r.get('reason', '')}")
        return
    if r.get("assessed") is False:
        print(f"  {r.get('confidence_note', 'too poorly constrained to assess')}")
        return
    enc = r.get("deflecting_encounter")
    if not enc:
        print(f"  {r.get('note', 'no deflecting encounter found')}")
        return
    print(f"  deflector: {enc['date'][:19]} UTC, perigee {enc['perigee_km']:,.0f} km"
          + (f", U∞={enc['u_infinity_km_s']:.2f} km/s" if enc.get("u_infinity_km_s") else "")
          + (f", θ={enc['theta_deg']:.0f}°" if enc.get("theta_deg") else ""))
    khs = r.get("keyholes", [])
    if not khs:
        print(f"  {r.get('note', 'no keyhole threaded by the Line of Variations')}")
        return
    for kh in khs:
        when = (kh.get("return_date") or "?")[:10]
        if kh.get("resolved"):
            ip = kh.get("ip"); odds = f" (~1 in {1.0/ip:,.0f})" if ip else ""
            w = kh.get("keyhole_width_km")
            print(f"  {kh['resonance']:>5} keyhole → {when} return: IP = {ip:.2e}{odds}"
                  + (f", width {w:.2f} km" if w is not None else "")
                  + f", LOV {kh.get('lov_center_sigma', 0):+.2f}σ")
        else:
            up = kh.get("ip_upper")
            print(f"  {kh['resonance']:>5} keyhole → {when} return: no resolved impact"
                  + (f", IP < {up:.1e}" if up else "")
                  + (f"  ({kh['note']})" if kh.get("note") else ""))
    if r.get("ip_total"):
        print(f"  TOTAL keyhole IP: {r['ip_total']:.2e}")
    if r.get("caveat"):
        print(f"  {r['caveat']}")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help") or "--help" in args:
        print(USAGE)
        return
    force = "--force" in args
    if "--no-radar" in args:        # skip the radar delay/Doppler refine (PYROORB_RADAR=0)
        args.remove("--no-radar")
        os.environ["PYROORB_RADAR"] = "0"
    # --ident-check OUT [--pairs FILE]: batch identification mode. Reads candidate pairs (from
    # --pairs FILE, or the two positional designations) and appends each clean ID to OUT.
    ident_check_out = _opt(args, "--ident-check")
    if ident_check_out is not None:
        pairs_file = _opt(args, "--pairs")
        pair_timeout = float(_opt(args, "--pair-timeout", "120") or 120)
        obscodes = load_obscodes()
        if pairs_file:                       # BATCH: one subprocess per pair (hard timeout)
            pairs = _read_pairs(pairs_file)
            if not pairs:
                print("usage: fit_object.py --ident-check OUT --pairs FILE", file=sys.stderr); return
            run_ident_check(pairs, ident_check_out, obscodes, force=force, pair_timeout=pair_timeout)
        else:                                # SINGLE pair, in-process (also the subprocess unit)
            d = [a for a in args if not a.startswith("--")]
            if len(d) < 2:
                print("usage: fit_object.py --ident-check OUT --pairs FILE   (or two designations)",
                      file=sys.stderr); return
            _ident_check_one(d[0], d[1], ident_check_out, obscodes, force=force)
        return
    # --mag-audit DESIG... | --mag-audit --list FILE: audit each SINGLE designation for a
    # significant implied-H change BETWEEN its apparitions (a single body shares one absolute
    # magnitude across oppositions, so an apparition sitting well off the others -- junk survey
    # photometry, a contaminant tracklet, or a real brightness change -- is worth a look). Reads
    # designations from --list FILE (one per line, '#' comments) and/or the positional args; writes
    # an advisory report to stdout or --out-file. Never alters an orbit.
    if "--mag-audit" in args:
        args.remove("--mag-audit")
        list_file = _opt(args, "--list")
        out_file = _opt(args, "--out-file")
        obscodes = load_obscodes()
        desigs = []
        if list_file:
            with open(list_file, encoding="utf-8") as fp:
                desigs += [ln.strip() for ln in fp if ln.strip() and not ln.startswith("#")]
        desigs += [a for a in args if not a.startswith("--")]
        if not desigs:
            print("usage: fit_object.py --mag-audit DESIG [DESIG ...]   (or --mag-audit --list FILE)",
                  file=sys.stderr); return
        out = open(out_file, "w", encoding="utf-8") if out_file else sys.stdout
        try:
            run_mag_audit(desigs, obscodes, force=force, out=out)
        finally:
            if out_file:
                out.close()
        return
    # --itf-search: search the ITF for tracklets belonging to the given designation(s)
    # (native itfsearch, src/pyroorb/itf/): PyRoOrb fit -> ephemeris corridor -> candidate
    # tracklets from itf.itf_tracklets (--itf-candidate-source scan for the legacy obs_sbn scan) ->
    # trial fits -> recursion -> acceptance gates. A found linkage is printed as the MPC
    # identification JSON; it is transmitted only with --itf-submit (dry-run otherwise).
    # Options: --itf-max-uncert ARCSEC (corridor cap, default ItfConfig.max_uncert_deg),
    # --itf-rank-mode ml|certainty|legacy (default ml), --itf-jitter (variant-orbit
    # jitter on the seed), --itf-trace [--itf-trace-level 0-4] (the live decision
    # trace and its verbosity; 0 is off), --itf-mag-limit MAG (corridor faint cut, applied
    # exactly; default ItfConfig.mag_limit, deepened per object -- see ItfConfig.mag_floor_margin) / --itf-thorough (sweep to the deeper legacy 24.0) /
    # --itf-mag-fallback (EXPERIMENTAL, off by default: re-sweep deeper when
    # nothing links), --itf-prefer complete|submittable (which result leads when
    # the search holds both a linkage and a fuller 0.6; default complete, or
    # submittable with --itf-submit), at parity with `python -m pyroorb.itf search` and
    # the web modal.
    if "--itf-search" in args:
        args.remove("--itf-search")
        from pyroorb.itf.config import ItfConfig, PREFER_CHOICES, default_prefer
        from pyroorb.itf.search import alternative_note, itf_search_live
        from pyroorb.itf.trace import as_tracer, L_FIREHOSE
        itf_candidate_source = _opt(args, "--itf-candidate-source") or "index"
        itf_rank_mode = _opt(args, "--itf-rank-mode") or "ml"
        itf_max_unc = _opt(args, "--itf-max-uncert")
        itf_submit = "--itf-submit" in args
        if itf_submit:
            args.remove("--itf-submit")
        itf_jitter = "--itf-jitter" in args
        if itf_jitter:
            args.remove("--itf-jitter")
        # corridor faint cut: the same three knobs the other two surfaces have
        itf_mag_limit = _opt(args, "--itf-mag-limit")
        itf_thorough = "--itf-thorough" in args
        if itf_thorough:
            args.remove("--itf-thorough")
        itf_mag_fb = "--itf-mag-fallback" in args
        if itf_mag_fb:
            args.remove("--itf-mag-fallback")
        itf_prefer = _opt(args, "--itf-prefer")
        if itf_prefer is not None and itf_prefer not in PREFER_CHOICES:
            print(f"--itf-prefer: expected one of {', '.join(PREFER_CHOICES)}, got "
                  f"{itf_prefer!r}", file=sys.stderr)
            return
        # --itf-trace (or PYROORB_VERBOSE=1): stream the live decision trace.
        # --itf-trace-level N picks its verbosity (0 off, 1 headline .. 4
        # firehose, default 2 narrative -- see itf/trace.py), with the SAME
        # meaning as `python -m pyroorb.itf search --trace-level`: 0 is an explicit
        # "no trace" and wins over --itf-trace and PYROORB_VERBOSE, exactly as
        # it does there.  (The web has no 0: the page always records the trace
        # for its viewer, so its selector offers 1-4.)
        # Refused outright when it is not a number or is out of range, as
        # argparse `choices=[0,1,2,3,4]` refuses it on the other CLI -- a bare
        # int() raised a ValueError traceback, and clamping a typo to the
        # nearest level answers a question nobody asked.
        itf_trace_level = _opt(args, "--itf-trace-level")
        if itf_trace_level is not None:
            try:
                itf_trace_level = int(itf_trace_level)
                if not 0 <= itf_trace_level <= L_FIREHOSE:
                    raise ValueError(itf_trace_level)
            except (TypeError, ValueError):
                print(f"--itf-trace-level: expected 0-{L_FIREHOSE}, got "
                      f"{itf_trace_level!r}", file=sys.stderr)
                return
        itf_trace = "--itf-trace" in args
        if itf_trace:
            args.remove("--itf-trace")
        if itf_trace_level == 0:
            itf_trace, itf_trace_level = False, None
        else:
            itf_trace = (itf_trace or itf_trace_level is not None
                         or os.environ.get("PYROORB_VERBOSE") == "1")
        # Every --itf-* option this surface knows has been consumed above, so
        # one still here is a typo or a retired spelling.  It must be refused by
        # name: this parser is hand-rolled, and the line below would otherwise
        # drop the flag silently and read its VALUE as a designation to fit.
        unknown = [a for a in args if a.startswith("--itf-")]
        if unknown:
            print(f"--itf-search: unknown option {', '.join(unknown)} (known: "
                  f"--itf-candidate-source, --itf-rank-mode, --itf-max-uncert, "
                  f"--itf-jitter, --itf-submit, --itf-mag-limit, --itf-thorough, "
                  f"--itf-mag-fallback, --itf-prefer, --itf-trace, --itf-trace-level)",
                  file=sys.stderr)
            return
        desigs = [a for a in args if not a.startswith("--")]
        if not desigs:
            print("usage: fit_object.py DESIG [DESIG ...] --itf-search", file=sys.stderr)
            return
        itf_cfg = ItfConfig(candidate_path=itf_candidate_source, ranking_mode=itf_rank_mode,
                            jitter=itf_jitter, submit=itf_submit,
                            prefer=itf_prefer or default_prefer(itf_submit))
        if itf_thorough:
            itf_cfg.apply_thorough()      # ONE definition -- see ItfConfig
        if itf_mag_limit is not None:      # an explicit cut wins over --itf-thorough
            try:
                itf_cfg.set_mag_limit(float(itf_mag_limit))
            except (TypeError, ValueError):
                print(f"--itf-mag-limit: expected a magnitude, got "
                      f"{itf_mag_limit!r}", file=sys.stderr)
                return
        if itf_mag_fb:
            itf_cfg.mag_limit_fallback = True
        result = itf_search_live(
            desigs, cfg=itf_cfg,
            max_uncert_deg=(float(itf_max_unc) / 3600.0 if itf_max_unc else None),
            do_submit=itf_submit,
            trace=(as_tracer(print, itf_trace_level)
                   if itf_trace and itf_trace_level is not None
                   else print if itf_trace else None))
        print(f"ITF search: status={result.verdict.legacy_status} {result.reason}")
        if alternative_note(result):
            print(alternative_note(result))
        for w in result.warnings or []:
            print(f"WARNING: {w}")
        if result.itf_obs80:
            print("linked ITF astrometry:" if result.verdict is SearchVerdict.LINKED else
                  "POSSIBLE (unconfirmed, NOT submitted) ITF astrometry:")
            print(result.itf_obs80)
        return
    # --batch FILE: fit every designation in an astrometry FILE (OBS80 or ADES) and write
    # one MPCORB (and/or SOF) row per designation. Identification-agnostic -- obs are grouped
    # by the designation each carries; no cross-designation linking (that needs a separate
    # IDs file). Options: --in-format obs80|ades (override auto-detect), --out-format
    # mpcorb,sof (default mpcorb), --out-file PATH (default stdout), --batch-one DESIG
    # (restrict to one designation), --batch-timeout SECS (>0: isolate each designation in a
    # subprocess with a hard kill, so one hanging object can't stall the batch).
    batch_file = _opt(args, "--batch")
    if batch_file is not None:
        in_format = _opt(args, "--in-format")
        out_format = _opt(args, "--out-format") or "mpcorb"
        out_file = _opt(args, "--out-file")
        only = _opt(args, "--batch-one")
        batch_timeout = float(_opt(args, "--batch-timeout", "0") or 0)
        emit_json = "--batch-emit-json" in args     # internal: child emits JSON rows for the parent
        fmts = _parse_egress_formats(out_format)
        bad = [f for f in fmts if f in _SINGLE_OBJECT_FORMATS]
        if bad:
            print(f"--batch cannot write {','.join(bad)}: it is a single-object format",
                  file=sys.stderr)
            return
        obscodes = load_obscodes()
        verbose = bool(os.environ.get("PYROORB_VERBOSE"))
        with open(batch_file, encoding="utf-8") as fp:
            text = fp.read()
        run_batch_cli(text, batch_file, obscodes, fmts, out_file, in_format,
                      force, only, batch_timeout, emit_json, verbose)
        return
    if "--finite-diff" in args:                 # legacy finite-difference Jacobian
        os.environ["PYROORB_FINITE_DIFF_JAC"] = "1"
    if "--no-mpc-seed" in args:                 # force the full IOD/subarc/extend path even on a
        os.environ["PYROORB_MPC_SEED"] = "0"         # well-observed, multi-opp object (skip the MPC seed)
    # --out-format mpcorb,sof,elem,covar [--out-file PATH]: egress the fitted orbit into the
    # interchange formats. Comma-separated; writes to PATH or stdout. `covar`
    # is find_orb's covar.json and is single-object only (see _SINGLE_OBJECT_FORMATS).
    out_format = _opt(args, "--out-format")
    out_file = _opt(args, "--out-file")
    # ephemeris options: --ephem [--ephem-start ISO] [--ephem-days N] [--ephem-step D] [--site CODE]
    #                    [--ephem-faint MAG] [--ephem-suppress]
    ephem = "--ephem" in args
    eph_start = _opt(args, "--ephem-start")
    eph_days = float(_opt(args, "--ephem-days", "30") or 30)
    eph_step = float(_opt(args, "--ephem-step", "1") or 1)
    eph_site = _opt(args, "--site", "500") or "500"
    # find_orb's two display cuts: hide epochs predicted fainter than MAG (bare
    # --ephem-faint takes find_orb's own 22.0), and hide epochs --site can't shoot
    # (below the horizon / daylight / behind the Moon -- topocentric codes only).
    eph_faint = None
    if "--ephem-faint" in args:
        import pyroorb.ephemerides as _eph
        i = args.index("--ephem-faint")
        nxt = args[i + 1] if i + 1 < len(args) else None
        try:                                   # --ephem-faint MAG
            eph_faint = float(nxt)
            del args[i:i + 2]
        except (TypeError, ValueError):        # bare flag -> find_orb's default limit
            eph_faint = _eph.FAINT_LIMIT_DEFAULT
            del args[i]
    eph_suppress = "--ephem-suppress" in args
    if eph_suppress:
        args.remove("--ephem-suppress")
    # --nongrav MODEL: after the gravity fit converges, turn on a non-gravitational
    # model (comet2/comet3/yarkovsky/srp) and re-solve for its A1/A2/A3 (find_orb-style).
    nongrav_model = _opt(args, "--nongrav")
    # --nongrav-jackknife: after the fit, add the delete-1-tracklet robustness sigma (a few
    # dozen refits, ~10-30s) next to the formal profile +/-. Opt-in because of the cost.
    want_jackknife = "--nongrav-jackknife" in args
    if want_jackknife:
        args.remove("--nongrav-jackknife")
    # --constrain "e=0.2,i=10": hold one or more elements fixed (find_orb limited_orbit).
    # Elements: a/q/Q (au), e (-), i (deg). e=1 -> parabolic, e=0 -> circular.
    constrain_spec = _opt(args, "--constrain")
    # --asteroids [--asteroids-path PATH]: after the gravity fit, run the 16-asteroid
    # perturber POLISH (refine_asteroids, force=True -> bypasses the main-belt/2-opp gate).
    # Opt-in and run-scoped: the gravity fit is unchanged, only a final polish is added.
    # --asteroids-path overrides the kernel location WITHOUT enabling whole-fit env mode.
    asteroids_flag = "--asteroids" in args
    asteroids_path = _opt(args, "--asteroids-path")
    if asteroids_path:
        import pyroorb.force_assist as _fa
        _fa.ASTEROIDS_OVERRIDE = asteroids_path
    if asteroids_flag:
        args.remove("--asteroids")
    # --impact [--impact-days N]: Phase-1 impact LOCATION -- propagate forward and report
    # where/when the object reaches the ground (the CLI twin of the web Impact
    # pane). --impact-ip adds the EXPERIMENTAL Phase-2 LOV b-plane impact PROBABILITY, scanning
    # --impact-ip-days forward from today (default 365) or targeting one --impact-encounter
    # "YYYY-MM-DD..." UTC approach. Both are opt-in, leaf, post-fit products -- the default
    # pipeline and the regression suite never call them.
    impact_days = float(_opt(args, "--impact-days", "30") or 30)
    impact_ip_days = float(_opt(args, "--impact-ip-days", "365") or 365)
    impact_encounter = _opt(args, "--impact-encounter")
    # --keyholes [--keyhole-years N]: EXPERIMENTAL Phase-3 resonant-return keyholes (the layer
    # the 1-D --impact-ip method cannot reach -- an impact behind a deep prior encounter).
    keyhole_years = float(_opt(args, "--keyhole-years", "12") or 12)
    keyhole_flag = "--keyholes" in args
    if keyhole_flag:
        args.remove("--keyholes")
    impact_ip_flag = "--impact-ip" in args
    if impact_ip_flag:
        args.remove("--impact-ip")
    impact_flag = ("--impact" in args) or impact_ip_flag
    if "--impact" in args:
        args.remove("--impact")
    # --linkage: fit several designations together (including NEOCP temporary ids) and report the
    # MPC IDENTIFICATION that joint fit implies -- the same document the web UI's "Submit linkage"
    # button sends, gated by the same ident_check.linkage_verdict. Dry-run print unless
    # --linkage-submit, which is the CLI twin of clicking that button.
    linkage_submit = "--linkage-submit" in args
    if linkage_submit:
        args.remove("--linkage-submit")
    linkage_flag = ("--linkage" in args) or linkage_submit
    if "--linkage" in args:
        args.remove("--linkage")
    # --center <planet>: the "told-the-center" planetocentric path. The object
    # is asserted to orbit this planet (e.g. a Saturn satellite); seed + report the orbit about
    # it instead of heliocentrically. Validated for the giant planets. Opt-in, post-fit report.
    center = _opt(args, "--center")
    if center is not None:
        import pyroorb.planetocentric as _pc
        try:
            center = _pc.resolve_planet(center)
        except ValueError as ex:
            print(f"ERROR: {ex}", file=sys.stderr)
            sys.exit(2)
    # Auto-detect whether the object is a giant-planet satellite (find_orb's
    # find_best_fit_planet) and, if so, re-fit + report it planetocentrically. ON BY DEFAULT;
    # --no-auto-center disables it. (--auto-center is accepted as a harmless no-op for back-compat.)
    no_auto_center = "--no-auto-center" in args
    if no_auto_center:
        args.remove("--no-auto-center")
    if "--auto-center" in args:
        args.remove("--auto-center")
    desigs = [a for a in args
              if a not in ("--force", "--finite-diff", "--ephem", "--asteroids",
                           "--no-mpc-seed")] or ["2023 QJ3"]
    # Pluto ban (early, by typed designation): it is a DE440 perturber and cannot be
    # fit unperturbed by itself. A self-identifying-by-number paste/fetch is caught
    # again inside fit_orbit; this just skips the network for an obvious 'Pluto'/134340.
    if any(is_pluto(d) for d in desigs):
        print(PLUTO_BAN_MSG, file=sys.stderr)
        sys.exit(2)
    obscodes = load_obscodes()
    # Defaults to ADES (modern; per-obs rmsRA/rmsDec); OBS80 only as fallback.
    obs, meta = fetch_observations(desigs, obscodes)
    # A fetch that FAILED is not an object with no observations -- and a PARTIAL failure
    # is worse than a total one: a linkage where one designation's fetch timed out still
    # returns rows, and fitting them would silently answer from half the requested
    # evidence while reporting success. The web routes already refuse this
    # (engine_api.run_fit / fitcache.get_by_desig); the CLI must not be the one surface
    # that quietly proceeds. obs_sources records which designations failed and why.
    _fmsg = fetch_failure_message(meta)
    if _fmsg:
        print(f"ERROR: {_fmsg}", file=sys.stderr)
        sys.exit(3)
    # Second Pluto gate: the fetched obs may self-identify as 134340 even when the box
    # didn't say 'Pluto' (an odd MPC round-trip). Ban it again before any fit work.
    if any(is_pluto(c) for c in _pluto_candidates(meta=meta, obs=obs)):
        print(PLUTO_BAN_MSG, file=sys.stderr)
        sys.exit(2)
    _unknown = pop_unknown_obscodes()
    if _unknown:
        codes = ", ".join(f"{c} ({n})" for c, n in sorted(_unknown.items()))
        print(f"WARNING: unrecognised observatory code(s): {codes} — those observations "
              f"were dropped. Your ObsCodes.htm may be out of date; refetch with "
              f"PYROORB_OBSCODES_REFRESH=1.", file=sys.stderr)
    fmt = meta.get("format", "ADES")
    if meta["combined"]:
        print(f"object   : {meta['designation']}  (COMBINED; linked: "
              f"{', '.join(meta['other_designations'])})  "
              f"{len(meta['designations'])} designations, {len(obs)} obs  [{fmt}]"
              + ("  [--force]" if force else ""))
    else:
        p = meta["designation"]
        print(f"object   : {p}  (packed {pack_designation(p)!r})  [{fmt}]")
    dup = meta.get("duplicates_dropped") or []
    if dup:
        pairs = sorted({(d["designation"], d["kept_designation"]) for d in dup})
        print(f"dedup    : {len(dup)} cross-listed obs dropped (same station, instant and "
              f"position under two designations): "
              + ", ".join(f"{a} -> kept as {b}" for a, b in pairs))

    verbose = bool(meta["combined"] or os.environ.get("PYROORB_VERBOSE"))
    fit = fit_orbit(obs, obscodes, desig=meta["designation"], meta=meta,
                    verbose=verbose, force=force, center=center,
                    auto_center=(False if no_auto_center else None),
                    # the planetocentric path ignores the MPC seed (MPC has no heliocentric
                    # orbit for a satellite); skip the needless network fetch when --center is set.
                    mpc_seed=(None if center else fetch_mpc_seed(obs, meta, verbose)))
    # fit_orbit runs auto-detect internally (on by default); pick up a detected satellite center
    # so the planetocentric report prints below.
    if center is None:
        center = getattr(fit, "center", None)
        if center is not None:
            print(f"auto-center: detected satellite of {center.capitalize()}")
    if asteroids_flag:
        fit = refine_asteroids(fit, force=True, verbose=True)
    if nongrav_model and nongrav_model != "none":
        # explicit model request -> (1) DEEP coordinate-profiled seed so the joint LM descends to the
        # true minimum instead of stopping short, and (2) the RATIONAL profile-likelihood sigma
        # (Delta-chi2=1), which flags a degenerate/unconstrained coefficient instead of a fake tight one.
        fit = refine_nongrav(fit, nongrav_model, verbose=True, profile_sigmas=True,
                             asteroids=_ng_ast_gate(fit),
                             radial_seed_all=True)
    if constrain_spec:
        cons = {}
        for tok in constrain_spec.replace(";", ",").split(","):
            if "=" in tok:
                k, _, v = tok.partition("=")
                cons[k.strip()] = float(v)
        if cons:
            fit = refit_constrained(fit, cons, verbose=True)
    print(f"obs used : {fit.n_observations()}/{fit.n_total_observations()} over "
          f"{fit.arc_length_days():.1f}-day arc, {fit.n_oppositions()} opposition(s), "
          f"codes {fit.observatory_codes()}"
          + (f"  ({fit.n_excluded()} rejected)" if fit.n_excluded() else ""))
    nsrc = {r["uncert_src"] for r in fit.residuals()}
    print(f"fit      : RMS = {fit.rms_residual():.3f}\"  (normalized {fit.rms_normalized():.2f}σ)  "
          f"n_resids={fit.n_resids()}  uncert={'/'.join(sorted(nsrc))}"
          + (f"  debiased {fit.n_debiased()}/{fit.n_total_observations()}" if fit.n_debiased() else ""))
    print(f"           force model {fit.force_model}, IOD {fit.iod_method}, seed score {fit.seed_d:.2f}")
    if not fit.converged:
        print("    ⚠ final LM stopped at its iteration cap, not at a minimum -- the elements "
              "are a mid-iteration state and the sigmas are not formal uncertainties")
    if fit.nongrav_model:
        # RATIONAL profile-likelihood interval (Delta-chi2=1), computed on demand here. An
        # UNCONSTRAINED axis shows ">" (its interval never rose by 1 -- the coefficient is
        # degenerate, the bound is only a lower limit); a meaningfully ASYMMETRIC interval shows
        # "+plus / -minus" (the -/+ half-widths differ on a degenerate valley), else a plain "± sig".
        rsig = fit.rational_nongrav_sigmas()
        rint = fit.rational_nongrav_intervals()
        unc = set(fit.nongrav_unconstrained)
        def _err(nm):
            lo, hi = rint.get(nm, (rsig[nm], rsig[nm]))
            if nm in unc:
                return f"> {max(lo, hi):.1e} (unconstrained)"
            if abs(hi - lo) > 0.15 * max(hi, lo, 1e-300):     # genuinely asymmetric
                return f"+{hi:.1e} / -{lo:.1e}"
            return f"± {max(lo, hi):.1e}"
        ps = "  ".join(f"{nm} = {fit.nongrav_params[nm]:+.3e} {_err(nm)}"
                       for nm in fit.nongrav_params)
        print(f"nongrav  : model {fit.nongrav_model} (au/day²)  {ps}")
        # AMR: find_orb prints the implied area/mass ratio on the elements' Epoch line for its
        # 1-parameter SRP model ONLY ("AMR 0.034 m^2/kg", elem_out.cpp) -- same rule here, so
        # nothing is claimed for srp2/srp3, whose radial term is not the whole photon push.
        amr = presentation.amr_line(_ngm.amr_report(
            fit.nongrav_model, fit.nongrav_params, rint, rsig, unc))
        if amr:
            print(f"    {amr}")
        if want_jackknife:
            # Delete-1-tracklet JACKKNIFE: the robustness sigma the covariance/profile cannot give.
            # It removes a whole tracklet (the unit a systematic lives in) and refits -- so a
            # coefficient resting on a few high-leverage tracklets, or fitting a correlated
            # systematic, shows a LOW jackknife z here even when the profile z is high. NOT ROBUST
            # marks a coefficient whose leave-one-tracklet cloud SPANS ZERO -- the sign itself does
            # not survive dropping one tracklet. It is a sign test, not a z threshold: the "< ~2
            # jk-sigma" clause this comment used to name was removed in 34ce636, because a real
            # coefficient with one high-leverage tracklet can sit at a low z without its sign ever
            # being in doubt.
            jk = fit.nongrav_jackknife_detail()
            if jk:
                def _jkcell(nm):
                    d = jk[nm]
                    if not d.get("assessed"):
                        return f"{nm} not assessed ({d.get('n',0)} converged of {d.get('n_tracklets','?')})"
                    return (f"{nm} ± {d['se']:.1e} (robust z={d['z']:.1f}"
                            + (", NOT ROBUST -- sign not stable" if d['spans_zero'] else "") + ")")
                jps = "  ".join(_jkcell(nm) for nm in fit.nongrav_params if nm in jk)
                nt = next(iter(jk.values())).get("n_tracklets", "?")
                print(f"    jackknife (delete-1 of {nt} tracklets, robust MAD scale): {jps}")
                if fit.nongrav_not_robust:
                    print(f"    ⚠ {', '.join(fit.nongrav_not_robust)}: the sign does not survive "
                          f"dropping one tracklet -- likely a systematic or a lone high-leverage "
                          f"tracklet, not a force")
            else:
                print("    jackknife: not enough tracklets to assess robustness")
        if fit.nongrav_warning:                       # advisory only -- the model was still fitted
            print(f"    ⚠ {fit.nongrav_warning}")
    if fit.constraints:
        print("constr   : held fixed -> "
              + "  ".join(f"{k}={v:g}" for k, v in fit.constraints.items())
              + "  (find_orb limited_orbit)")
    if fit.asteroids:
        print("asteroids: 16 massive perturbers active (sb441-n16)")
    if getattr(fit, "radar", None):
        import pyroorb.radar as _radar
        rd = fit.radar
        if rd.get("applied"):
            print(f"radar    : {rd['n_used']}/{rd['n']} delay/Doppler measurements in the fit "
                  f"(median |O-C| {rd['post_median_z']:.2f}σ)")
        else:
            print(f"radar    : {rd['n']} measurements present but NOT fit "
                  f"({rd.get('note', 'unavailable')})")
        for rr in rd["rows"]:
            unit = "µs" if rr["kind"] == "delay" else "Hz"
            resid = rr["resid"] * (1e6 if rr["kind"] == "delay" else 1.0)
            mark = "" if rr["included"] else "   x excluded"
            print(f"    {rr['kind']:7s} {Time(rr['jd_utc'], format='jd').iso[:16]}  "
                  f"{_radar.station_name(rr['tx'])}→{_radar.station_name(rr['rx'])}"
                  f"  O-C {resid:+10.3f} {unit} ({rr['z']:+.1f}σ){mark}")
    # Systematic-residual audit (same check the --ident-check batch uses, here for ANY object): an
    # acceptable overall RMS can still hide an apparition whose residuals TREND across the arc -- a
    # contaminated tracklet or a mislink the fit absorbed. Surface it so it isn't taken on faith.
    import pyroorb.ident_check as _idc
    _sys = _idc.systematic_misfit(fit)
    if _sys:
        # same sentence the web banner uses (app.js flashFitBanners), so the two surfaces agree
        print(f"WARNING  : residuals trend instead of scatter -- {_sys}. Nights sitting coherently "
              f"off the orbit is the usual sign of a wrong linkage or a contaminated night; check "
              f"the residuals night by night before trusting this orbit")
    if fit.history:
        stages = " -> ".join(f"{h['stage']}:{h['n_obs']}obs/{h['rms']:.2f}\"" for h in fit.history)
        print(f"arc-ext  : {stages}")
    for r in fit.excluded_observations():
        print(f"  x rejected {r['obscode']} {Time(r['jd_utc'],format='jd').iso[:10]} "
              f"({r['designation']})  O-C=({r['d_ra_arcsec']:+.2f}\",{r['d_dec_arcsec']:+.2f}\")"
              f"  = {r['resid_sigmas']:.1f}σ")
    print(f"epoch    : JD {fit.epoch_jd():.5f} TDB  ({fit.epoch_iso()[:10]})  frame {fit.frame()}")
    H, sH = fit.absolute_magnitude()
    if H is not None:
        bands = sorted({r['band'] for r in fit.magnitude_residuals()})
        print(f"mag      : H = {H:.2f} ± {sH:.2f}  (G={fit.slope_parameter()}, "
              f"{fit.n_magnitudes()} mags, bands {bands} → V)")
    moid = fit.moid_earth(); U = fit.uncertainty_parameter()
    print(f"class    : {fit.orbit_class()}"
          + ("  [PHA]" if fit.is_pha() else "")
          + f"  |  Earth MOID = {moid:.4f} au" + (f" ({moid*389.0:.1f} LD)" if moid else "")
          + f"  |  T_J = {fit.tisserand_jupiter():.2f}"
          + (f"  |  U = {U:.1f}" if U is not None else ""))
    # Unbound orbits (hyperbolic.py): the same two informational records the web note shows.
    _uc = getattr(fit, "unbound_check", None)
    if _uc and _uc.get("kept") == "bound":
        why = (f"only {_uc['nights']} nights over {_uc['arc_days']:.1f} d -- too few to believe an "
               f"unbound orbit" if _uc.get("damaged")
               else (f"the data disfavour it at only {_uc['sigma']:.1f} sigma"
                     if _uc.get("sigma") is not None
                     else "holding e below 1 barely changes the residuals"))
        print(f"unbound  : an unbound solution also fits (e = {_uc['e_free']:.5f}, rms "
              f"{_uc['rms_free']:.3f}\" vs {_uc['rms_bound']:.3f}\" bound); bound orbit shown -- {why}")
    el, sg = fit.elements(), fit.element_sigmas()         # same source the web report reads
    if el["e"] > 1.0:
        from pyroorb import hyperbolic
        _rem = hyperbolic.inbound_reminder(el, sg["e"], fit.epoch_tdb)
        if _rem:
            print(f"note     : {_rem['text']}")
    print("\nelements (value ± 1σ):")
    for r in presentation.ELEM_ROWS:
        val, sig = el[r["name"]], sg[r["name"]]
        if presentation.safe(val) is None and el["e"] > 1.0:
            print(f"  {r['cli']} = {'undefined':>14s}   (an unbound orbit has no {r['web']})")
            continue
        print(f"  {r['cli']} = {val:14.6f} ± {sig:.6f} {r['unit']}")
    print(f"\ncovariance (elements) is {fit.covariance_matrix().shape} matrix; "
          f"correlation(a,e) = {fit.correlation_matrix()[0,1]:+.3f}")

    if out_format:
        from pyroorb import egress
        # Same format->writer path as --batch: build this one fit's rows and assemble them,
        # so single-object and batch egress can't drift (one shared header/row dispatch). One
        # written_jd stamps every format's Twritten consistently.
        fmts = _parse_egress_formats(out_format)
        text = _assemble_batch_output(fmts, [_rows_for_fit(fit, fmts, egress._now_jd())])
        if out_file:
            with open(out_file, "w", encoding="utf-8") as fp:
                fp.write(text + "\n")
            print(f"\nwrote {out_format} -> {out_file}")
        elif text:
            print("\n" + text)

    if ephem:
        from pyroorb import ephemerides as ephemeris
        start = (Time(eph_start).jd if eph_start else fit.epoch_jd())
        rows = ephemeris.compute_ephemeris(fit, obscodes, start, start + eph_days,
                                           eph_step, site=eph_site)
        rows, hidden = ephemeris.filter_rows(rows, faint_limit=eph_faint,
                                             suppress_unobservable=eph_suppress)
        print()
        print(ephemeris.format_table(rows, site=eph_site, hidden=hidden))

    if impact_flag:
        _print_impact(fit, impact_days)
    if impact_ip_flag:
        enc_jd = float(Time(impact_encounter, scale="utc").jd) if impact_encounter else None
        _print_impact_probability(fit, impact_ip_days, enc_jd)
    if keyhole_flag:
        _print_keyholes(fit, keyhole_years)
    if linkage_flag:
        _print_linkage(fit, do_submit=linkage_submit)
    if center is not None:
        _print_planetocentric(fit, center)


if __name__ == "__main__":
    from pyroorb.console import utf8_stdio
    utf8_stdio()
    main()
