"""The comet A3/A1 advisory (nongrav.comet_ratio_warning): a comet delivery whose out-of-plane A3
exceeds COMET_A3_A1_ADVISORY_RATIO x |A1| -- outside every published comet solution -- carries a
note that the fitted ratio is not trusted. Advisory only; it never changes the model."""
import pyroorb.nongrav as ng


def test_a_ratio_past_the_published_range_is_flagged():
    # 2P/Encke on its 2018-11..2026-11 arc: A3 22x A1
    w = ng.comet_ratio_warning({"A1": 6.0e-11, "A2": 5.8e-12, "A3": 1.31e-9})
    assert w is not None
    assert "don't trust the fitted A1 to A3 ratio" in w
    assert "22x |A1|" in w
    assert "comet2" in w


def test_a_ratio_inside_the_published_range_is_not_flagged():
    # 21P (JPL fits A3 too): |A3| ~ 0.56 |A1|; and exactly at the bar
    assert ng.comet_ratio_warning({"A1": 4.6e-9, "A2": -8.9e-10, "A3": -2.57e-9}) is None
    bar = ng.COMET_A3_A1_ADVISORY_RATIO
    assert ng.comet_ratio_warning({"A1": 1e-9, "A3": bar * 1e-9}) is None


def test_the_sign_of_either_term_does_not_matter():
    assert ng.comet_ratio_warning({"A1": -1e-10, "A3": 5e-10}) is not None
    assert ng.comet_ratio_warning({"A1": 1e-10, "A3": -5e-10}) is not None


def test_models_without_both_terms_draw_no_note():
    assert ng.comet_ratio_warning({"A1": 1e-9}) is None
    assert ng.comet_ratio_warning({"A1": 1e-9, "A2": 1e-8}) is None
    assert ng.comet_ratio_warning({"A1": 0.0, "A3": 1e-9}) is None
    assert ng.comet_ratio_warning({}) is None
    assert ng.comet_ratio_warning(None) is None
