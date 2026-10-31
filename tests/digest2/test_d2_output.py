from pyroorb.digest2.config import Config
from pyroorb.digest2.output import format_header_lines, format_mpc_desig, format_score_line, strtok_first_token
from pyroorb.digest2.ranging import ScoreResult


def test_format_mpc_desig_numbered_object():
    # permanent-number field (cols 1-5) non-blank -> shown in the
    # provisional-designation slot instead, 2 leading spaces + 5 chars
    assert format_mpc_desig("NE00030     ") == "  NE000"


def test_format_mpc_desig_provisional_only():
    assert format_mpc_desig("     K16S99K") == "K16S99K"


def test_strtok_first_token():
    assert strtok_first_token("     K16S99K") == "K16S99K"
    assert strtok_first_token("NE00030     ") == "NE00030"


def test_default_header_lines():
    # digest2's actual default shows only noid (not raw+noid dual columns),
    # so the dashes sub-header line is skipped -- matches OPERATION.md's
    # own "Basic operation" example verbatim.
    cfg = Config()
    lines = format_header_lines(cfg)
    assert lines == ["Desig.    RMS Int NEO N22 N18 Other Possibilities"]


def test_dual_column_header_shows_dashes_line():
    cfg = Config()
    cfg.raw = True  # both raw and noid now on
    lines = format_header_lines(cfg)
    assert lines[0] == "-------  ----   Int     NEO     N22     N18   ---------------"
    assert lines[1] == "Desig.    RMS Raw NID Raw NID Raw NID Raw NID Other Possibilities"


def test_restricted_class_header_no_dashes_no_other_possibilities():
    cfg = Config()
    cfg.class_columns = [1, 7]
    cfg.class_compute = [1, 7]
    cfg.class_possible = False
    cfg.raw = False
    cfg.noid = True
    lines = format_header_lines(cfg)
    assert len(lines) == 1  # no dashes line when raw&noid aren't both on
    assert lines[0] == "Desig.    RMS NEO MB1"


def test_noheadings_produces_no_lines():
    cfg = Config()
    cfg.headings = False
    assert format_header_lines(cfg) == []


def _score(rms=0.5, **kw):
    raw = {i: 0.0 for i in range(15)}
    noid = {i: 0.0 for i in range(15)}
    raw.update(kw.get("raw", {}))
    noid.update(kw.get("noid", {}))
    return ScoreResult(rms=rms, rms_prime=0.0, raw_scores=raw, noid_scores=noid)


def test_format_score_line_default_columns_and_possibilities():
    cfg = Config()
    result = _score(rms=0.15, raw={0: 100, 1: 100, 2: 36, 3: 0}, noid={0: 100, 1: 100, 2: 36, 3: 0})
    line = format_score_line("NE00030", result, cfg)
    assert line == "NE00030  0.15 100 100  36   0"


def test_format_score_line_other_possibilities_rounding():
    cfg = Config()
    result = _score(rms=0.42, raw={0: 24, 1: 23, 2: 4, 3: 0}, noid={0: 24, 1: 23, 2: 4, 3: 0},
                     **{})
    # noid drives "other possibilities" text by default (noid=True, raw=False by default... but
    # here raw was toggled on by Config default False/True: default cfg.raw=False, cfg.noid=True)
    result.noid_scores[4] = 7  # MC
    result.noid_scores[5] = 3  # Hun
    result.noid_scores[8] = 0.4  # Pal -> "<1"
    line = format_score_line("NE00269", result, cfg)
    assert "(MC 7)" in line
    assert "(Hun 3)" in line
    assert "(Pal <1)" in line
    assert "(Pho" not in line  # zero score -> omitted entirely


def test_format_score_line_limit_suppresses_output():
    cfg = Config()
    cfg.limit_class = 1
    cfg.limit_raw = False
    cfg.limit_value = 50
    result = _score(raw={1: 10}, noid={1: 10})
    assert format_score_line("X", result, cfg) is None
    result2 = _score(raw={1: 60}, noid={1: 60})
    assert format_score_line("X", result2, cfg) is not None


def test_format_score_line_rms_overflow_fallback():
    cfg = Config()
    cfg.class_columns = []
    cfg.class_compute = []
    result = _score(rms=1234.5)
    line = format_score_line("X", result, cfg)
    assert "**.**" in line
