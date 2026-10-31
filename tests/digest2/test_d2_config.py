import pytest

from pyroorb.digest2.config import Config, ConfigError, apply_config_lines, parse_limit_spec
from pyroorb.digest2.obscodes import SiteTable, parse_cod3


def test_default_config_matches_digest2_defaults():
    cfg = Config()
    assert cfg.headings is True
    assert cfg.rms is True
    assert cfg.raw is False
    assert cfg.noid is True
    assert cfg.repeatable is False
    assert cfg.class_columns == [0, 1, 2, 3]
    assert cfg.class_compute == list(range(15))


def test_poss_example_matches_default_behavior():
    # NOTE: OPERATION.md's own "Example 1" writes "Neo" (mixed case), but
    # digest2's config-file class matching is case-sensitive (strcmp) --
    # that's a typo in the docs, not real behavior to replicate. "NEO" is
    # the correct spelling, as used everywhere else in the docs/source.
    cfg = Config()
    st = SiteTable()
    apply_config_lines(cfg, ["Int", "NEO", "N22", "N18", "poss"], st)
    assert cfg.class_columns == [0, 1, 2, 3]
    assert cfg.class_possible is True
    assert cfg.class_compute == list(range(15))


def test_restricted_classes_shrink_class_compute():
    cfg = Config()
    st = SiteTable()
    apply_config_lines(cfg, ["# just three", "NEO", "Hun", "JTr"], st)
    assert cfg.class_columns == [1, 5, 13]
    assert cfg.class_possible is False
    assert cfg.class_compute == [1, 5, 13]


def test_raw_noid_toggle_first_occurrence_resets_other():
    cfg = Config()
    st = SiteTable()
    apply_config_lines(cfg, ["raw"], st)
    assert cfg.raw is True
    assert cfg.noid is False


def test_noheadings_and_norms():
    cfg = Config()
    st = SiteTable()
    apply_config_lines(cfg, ["noheadings", "norms"], st)
    assert cfg.headings is False
    assert cfg.rms is False


def test_repeatable_and_random_keywords():
    cfg = Config()
    st = SiteTable()
    apply_config_lines(cfg, ["repeatable"], st)
    assert cfg.repeatable is True
    apply_config_lines(cfg, ["random"], st)
    assert cfg.repeatable is False


def test_obserr_default_and_site_specific():
    cfg = Config()
    st = SiteTable()
    apply_config_lines(cfg, ["obserr = 0.7", "obserrF51=.3", "obserr704 = 1"], st)
    from pyroorb.digest2.constants import ARCSEC_RAD
    assert abs(cfg.default_obserr_rad - 0.7 * ARCSEC_RAD) < 1e-15
    assert abs(st[parse_cod3("F51")].obs_err - 0.3 * ARCSEC_RAD) < 1e-15
    assert abs(st[parse_cod3("704")].obs_err - 1.0 * ARCSEC_RAD) < 1e-15


def test_obserr_too_large_is_rejected():
    cfg = Config()
    st = SiteTable()
    with pytest.raises(ConfigError):
        apply_config_lines(cfg, ["obserr=15"], st)


def test_unrecognized_line_is_rejected():
    cfg = Config()
    st = SiteTable()
    with pytest.raises(ConfigError):
        apply_config_lines(cfg, ["not-a-real-keyword"], st)


def test_config_ignores_blank_lines_and_comments():
    cfg = Config()
    st = SiteTable()
    apply_config_lines(cfg, ["", "# a comment", "rms"], st)
    assert cfg.rms is True


def test_parse_limit_spec():
    idx, is_raw, value = parse_limit_spec("NEO/raw=50")
    assert (idx, is_raw, value) == (1, True, 50)
    idx, is_raw, value = parse_limit_spec("MB1/noid=1")
    assert (idx, is_raw, value) == (7, False, 1)


def test_parse_limit_spec_rejects_bad_syntax():
    with pytest.raises(ConfigError):
        parse_limit_spec("garbage")
    with pytest.raises(ConfigError):
        parse_limit_spec("NEO/raw=0")
    with pytest.raises(ConfigError):
        parse_limit_spec("NEO/raw=101")
    with pytest.raises(ConfigError):
        parse_limit_spec("NotAClass/raw=50")


def test_validate_limit_requires_class_in_compute_set():
    cfg = Config()
    cfg.class_compute = [1, 7]
    cfg.limit_class = 4  # MC, not configured
    cfg.limit_raw = False
    with pytest.raises(ConfigError):
        cfg.validate_limit()


def test_validate_limit_requires_matching_score_kind():
    cfg = Config()
    cfg.raw = False
    cfg.noid = True
    cfg.limit_class = 1
    cfg.limit_raw = True  # asking to limit on 'raw' but raw isn't being output
    with pytest.raises(ConfigError):
        cfg.validate_limit()
