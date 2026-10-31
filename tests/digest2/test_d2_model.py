from pathlib import Path

from pyroorb.digest2.model import load_model

DATA = Path(__import__("pyroorb.digest2", fromlist=["_"]).__file__).parent / "data" / "digest2.model.csv"


def test_load_real_model_csv_shapes():
    model = load_model(DATA, use_cache=False)
    from pyroorb.digest2.constants import D2CLASSES, EX, HX, IX, QX
    assert model.all_ss.shape == (QX, EX, IX, HX)
    assert model.unk_ss.shape == (QX, EX, IX, HX)
    assert model.all_class.shape == (D2CLASSES, QX, EX, IX, HX)
    assert model.unk_class.shape == (D2CLASSES, QX, EX, IX, HX)


def test_model_population_is_nonnegative_and_nonzero():
    model = load_model(DATA, use_cache=False)
    assert (model.all_ss >= 0).all()
    assert model.all_ss.sum() > 0
    assert model.unk_ss.sum() > 0


def test_class_population_never_exceeds_whole_solar_system():
    model = load_model(DATA, use_cache=False)
    # a class's "All" population at any bin can't exceed the SS total there
    assert (model.all_class <= model.all_ss[None, ...] + 1e-6).all()


def test_model_cache_round_trip(tmp_path):
    model = load_model(DATA, use_cache=False)
    from pyroorb.digest2.model import save_model_cache
    import numpy as np

    cache_path = tmp_path / "model.npz"
    save_model_cache(model, cache_path)
    reloaded = load_model(cache_path)
    assert np.array_equal(model.all_ss, reloaded.all_ss)
    assert np.array_equal(model.all_class, reloaded.all_class)


def test_a_read_only_model_directory_caches_in_the_user_cache(tmp_path, monkeypatch):
    """An installed package is read-only: the parsed-model cache goes to the user cache
    instead of failing silently and re-parsing the CSV in every process."""
    import os
    import shutil
    from pyroorb.digest2 import model as M
    ro = tmp_path / "site-packages" / "data"
    ro.mkdir(parents=True)
    csv = ro / "digest2.model.csv"
    shutil.copy(DATA, csv)
    monkeypatch.setenv("PYROORB_CACHE_DIR", str(tmp_path / "cache"))
    real_access = os.access
    monkeypatch.setattr(M.os, "access", lambda p, mode: Path(p) != ro and real_access(p, mode))
    target = M._cache_path_for(csv)
    assert target == tmp_path / "cache" / "digest2" / "digest2.model.csv.npz"
    M.load_model(csv)
    assert target.is_file() and not (ro / "digest2.model.csv.npz").exists()
    assert M._cache_path_for(csv) == target
    # a writable source checkout keeps the cache beside the CSV, as before
    monkeypatch.setattr(M.os, "access", real_access)
    assert M._cache_path_for(csv) == ro / "digest2.model.csv.npz"
