"""Pre-provision the JPL ephemeris kernels PyRoOrb integrates in.

Downloads (idempotently -- skips anything already present) into ``PYROORB_EPHEM_DIR``
(default: ``datadirs.table_dir('ephem')``):

  - DE440 planets    linux_p1550p2650.440   ~102 MB   REQUIRED
  - asteroid kernel  sb441-n16.bsp          ~616 MB   optional (16 perturbers)

The extra-perturber kernel (pyroorb-perturbers-1900-2100-20km.bsp, ~33 MB: 343 asteroids,
1900-2100, the 16 of them 1800-2200; the automatic perturber screen and, without sb441-n16,
the 16 themselves) ships
inside the package, so nothing is fetched for it; only an install without that copy builds it
here from JPL's sb441-n373 (~1 min, reading ~165 MB).

Why a separate command rather than the engine's lazy download: on a hosted box the
~700 MB asteroid fetch should happen ONCE, before traffic, into a persistent
volume -- not mid-fit (slow, and the asteroid/epsilon coupling wants the kernel
in place before the first integration). Run it at deploy time against the mounted
volume; subsequent boots find the files and skip.

    pyroorb-fetch-data                  # DE440 + debias tables
    pyroorb-fetch-data --asteroids      # + sb441-n16 (the 16 over DE441's whole span, 616 MB)

It reuses the engine's own downloaders so the URLs/paths never drift from the
engine. Honours PYROORB_EPHEM_DIR -- point it at the volume mount on a host.
"""
import argparse
import sys

if __name__ == "__main__":
    # Run as a program: load the `.env` settings file before force_assist and debias read
    # their folders (PYROORB_EPHEM_DIR / PYROORB_DEBIAS_DIR) at import.
    from pyroorb.envfile import load_env_file
    load_env_file()
    from pyroorb.console import utf8_stdio
    utf8_stdio()

import pyroorb.force_assist as fa  # noqa: E402
import pyroorb.debias as db  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asteroids", action="store_true",
                    help="also fetch the ~616 MB asteroid perturber kernel (full fidelity)")
    ap.add_argument("--no-perturbers", action="store_true",
                    help="do not check for (or build) the extra-perturber kernel; it ships with "
                         "the package, so this only matters for an install without that copy")
    ap.add_argument("--perturbers", action="store_true",
                    help="fail if the extra-perturber kernel is absent and cannot be built")
    ap.add_argument("--no-debias", action="store_true",
                    help="skip the ~10 MB EFCC18 debias tables (fetched by default)")
    args = ap.parse_args()

    print(f"ephem dir: {fa.EPHEM_DIR}")
    print("ensuring DE440 planetary ephemeris (~102 MB) ...")
    p = fa.ensure_planets_kernel(verbose=True)
    print(f"  planets: {p}")

    if not args.no_debias:
        print(f"debias dir: {db.DEBIAS_DIR}")
        print("ensuring EFCC18 debias tables (~10 MB) ...")
        if db.ensure_debias_tables(verbose=True):
            print(f"  debias: {db.DEBIAS_DIR}")
        else:
            print("  debias: download failed; fits will run undebiased")

    if args.asteroids:
        print("ensuring asteroid perturber kernel (~616 MB) ...")
        a = fa.ensure_asteroid_kernel(verbose=True, n16=True)
        if a:
            print(f"  asteroids: {a}")
            print("  -> the web UI and fits use it automatically for objects that need it;\n"
                  "     for every command-line fit, set PYROORB_ASTEROIDS_PATH to this path")
        else:
            print("  asteroids: download failed; fits will run gravity-only")
            sys.exit(1)
    if not args.no_perturbers:
        from pathlib import Path
        from pyroorb import perturbers, perturber_kernel
        have = perturbers.kernel_path()
        if have:
            print(f"  perturbers: {have} (present)")
        else:
            # only a stripped install lacks the bundled copy: make it from JPL instead
            dst = Path(fa.EPHEM_DIR) / perturbers.KERNEL_NAME
            print("building the asteroid-perturber kernel from JPL sb441-n373 (~1 min) ...")
            try:
                perturber_kernel.build(dst, p)
                print(f"  perturbers: {dst}")
            except Exception as ex:
                print(f"  perturbers: build failed ({type(ex).__name__}: {ex}); "
                      "fits skip the perturber screen (re-run to retry)")
                if args.perturbers:
                    sys.exit(1)
    print("done.")


if __name__ == "__main__":
    main()
