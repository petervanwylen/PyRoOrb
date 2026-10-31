"""The deployed ranker is ONE v4 model, and an UNSET ml_model_path must get
everything a v4 model needs: its own feature frame (honest scale + decision
context) and the continuous LOV offset pass that feeds it.  Both used to key on the
path being set, so a None default would have been served the pre-floor v3
frame with every LOV feature missing -- silently, since the scores still come
out as numbers."""
from pyroorb.itf.config import ItfConfig
from pyroorb.itf.training import _is_v4, default_model_path, load_ranker
from pyroorb.itf.search import _ranker_wants_lov


def test_the_default_ranker_is_v4_and_single():
    cfg = ItfConfig()
    assert cfg.ml_model_path is None and cfg.ml_second_model_path is None
    assert default_model_path().endswith("itf_match_v4.json")
    assert load_ranker(None)[1].get("v4")


def test_an_unset_model_path_gets_the_v4_frame_and_the_lov_pass():
    assert _is_v4(None)
    assert _ranker_wants_lov(ItfConfig())
    assert "lov_resid" in load_ranker(None)[1]["features"]
