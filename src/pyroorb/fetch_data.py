"""Pre-provision the JPL ephemeris kernels PyRoOrb integrates in.

Downloads (idempotently -- skips anything already present) into ``PYROORB_EPHEM_DIR``
(default: ``datadirs.table_dir('ephem')``):

  - DE440 planets    linux_p1550p2650.440   ~102 MB   REQUIRED
  - asteroid kernel  sb441-n16.bsp          ~616 MB   optional (16 perturbers)

Why a separate command rather than the engine's lazy download: on a hosted box the
~700 MB asteroid fetch should happen ONCE, before traffic, into a persistent
volume -- not mid-fit (slow, and the asteroid/epsilon coupling wants the kernel
in place before the first integration). Run it at deploy time against the mounted
volume; subsequent boots find the files and skip.

    pyroorb-fetch-data                # DE440 + debias tables (gravity-grade fits)
    pyroorb-fetch-data --asteroids    # + the 616 MB perturber kernel

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
        a = fa.ensure_asteroid_kernel(verbose=True)
        if a:
            print(f"  asteroids: {a}")
            print("  -> set PYROORB_ASTEROIDS_PATH to this path to enable whole-fit perturbers")
        else:
            print("  asteroids: download failed; fits will run gravity-only")
            sys.exit(1)
    print("done.")


if __name__ == "__main__":
    main()
