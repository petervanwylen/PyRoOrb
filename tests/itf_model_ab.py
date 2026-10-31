"""Re-run the three v3 failures with BOTH models, interleaved.

The failure reason was "N opps without progress, abandoning" -- the
no_progress_sec budget, which is WALL CLOCK.  A slow DB burns it without the
search making progress, so a transient network problem produces exactly this
symptom.  v2 and v3 were run ~40 minutes apart, so the original comparison did
not control for that.  Alternating them per fixture does.
"""
import os, sys, time
import warnings; warnings.filterwarnings('ignore')
from pyroorb.obs_sources import load_obscodes
from pyroorb.itf.config import ItfConfig
from pyroorb.itf.harness import run_recovery
from pyroorb.itf.candidates import DbSource
from pyroorb.itf.dbio import connect_pub
from _itf_fixtures import load_fixture, load_obs

MODELS = {'v2': 'itf/training_data/itf_match_v2.json',
          'v3': 'itf/training_data/itf_match_v3.json'}
FIX = ['2015_BE559', '2022_GL24', '2022_WY29']
REPEATS = int(os.environ.get('REPEATS', '2'))

oc = load_obscodes()
for name in FIX:
    fx = load_fixture(f'tests/itf_fixtures/{name}.json')
    obs, meta = load_obs(fx, oc)
    want = fx['n_fake_tracklets']
    for rep_i in range(REPEATS):
        for mname, mpath in MODELS.items():
            # ml_second_model_path=None: a pure-MODEL A/B must not inherit the
            # deployed v2 second opinion, or it measures the pair
            cfg = ItfConfig(ml_model_path=mpath, ml_second_model_path=None)
            base = DbSource(connect_pub(cfg), cfg,
                            reconnect=lambda c=cfg: connect_pub(c))
            t0 = time.time()
            rep = run_recovery(fx['designation'], cfg=cfg, base_source=base,
                               obscodes=oc, obs_meta=(obs, meta),
                               seed_nights=fx.get('seed_nights'),
                               log=lambda *a: None)
            ok = rep.recovered_tracklets >= want
            print(f"{name:>12} run{rep_i} {mname}: "
                  f"{'OK  ' if ok else 'MISS'} trk={rep.recovered_tracklets}/{want} "
                  f"apps={sorted(rep.recovered_apparitions)} "
                  f"{time.time()-t0:6.1f}s  {rep.reason[:60]}", flush=True)
