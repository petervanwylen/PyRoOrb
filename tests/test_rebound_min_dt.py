"""The rebound fallback floors IAS15's step like the ASSIST backend does.

Without the floor a trial state bound within a hair of a planet's point mass makes a
single sim.integrate() call take millions of steps per simulated day -- effectively a
hang, and one no between-call budget or deadline can interrupt. The timing case runs in
a subprocess so a regression fails on a timeout instead of wedging the test run.
"""
import math
import subprocess
import sys
from pathlib import Path

import pyroorb.forward_model as fm

PKG = Path(fm.__file__).resolve().parent          # src/pyroorb
SRC = PKG.parent


def test_base_sim_and_its_copies_carry_the_floor():
    assert fm.IAS15_MIN_DT > 0
    base = fm.build_base_sim(2460000.5)
    assert base.ri_ias15.min_dt == fm.IAS15_MIN_DT
    assert base.copy().ri_ias15.min_dt == fm.IAS15_MIN_DT   # every sweep integrates a copy


def test_both_backends_share_one_floor():
    src = (PKG / "force_assist.py").read_text()
    assert "from pyroorb.forward_model import IAS15_MIN_DT" in src


_TIGHT_ORBIT = r"""
import math, sys, numpy as np
sys.path.insert(0, %r)
import pyroorb.forward_model as fm
t0 = 2460000.5
base = fm.build_base_sim(t0)
earth, sun = base.particles[3], base.particles[0]   # Sun, Mercury, Venus, Earth(-Moon)
r = 1e-7                                             # au from the Earth point mass, bound
v = math.sqrt(fm.MU_SUN * earth.m / r)
state = np.array([earth.x - sun.x + r, earth.y - sun.y, earth.z - sun.z,
                  earth.vx - sun.vx, earth.vy - sun.vy + v, earth.vz - sun.vz])
times = t0 + np.array([1.0, 10.0, 30.0])
fm.predict_radec_batch(base, t0, state, times, [np.zeros(3)] * 3)
print("ok")
"""


def test_a_state_bound_to_a_planet_point_mass_returns():
    out = subprocess.run([sys.executable, "-c", _TIGHT_ORBIT % str(SRC)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
