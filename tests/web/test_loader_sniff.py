"""The paste loader's format sniff: a '|' in OBS80 column 14 is not ADES PSV.

Column 14 (Note1, the program code) of an 80-column line legitimately carries '|'.
The paste loader used to call a blob ADES whenever its first real line held ANY '|',
so a W84 packet whose first line had one went to the ADES parser and came back
"no usable observations found" -- while the same lines loaded fine by designation and
through the CLI's file reader, which asks for a '|' header naming an ADES field.

    python -m pytest web/tests/test_loader_sniff.py
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
from pyroorb import ades
from pyroorb.web import loader
from pyroorb.web import engine_api

# cols 1-5 number, 6-12 provisional, 13 discovery, 14 Note1, 15 Note2, 16-32 date,
# 33-44 RA, 45-56 Dec, 66-70 mag, 71 band, 78-80 station
_LINE = (" " * 5 + "K24A00A" + " " + " " + "C" + "2024 01 15.12345 "
         + "12 34 56.789" + "+12 34 56.78" + " " * 9 + "20.1 " + "G" + " " * 6 + "W84")
_PIPE = _LINE[:13] + "|" + _LINE[14:]


def test_fixture_lines_are_80_columns():
    assert len(_LINE) == 80 and len(_PIPE) == 80 and _PIPE[13] == "|"


def test_pipe_in_note1_on_the_first_line_is_still_obs80():
    obscodes = engine_api._obscodes()
    for text in (_PIPE, _PIPE + "\n" + _LINE, "\n# comment\n" + _PIPE + "\n" + _LINE):
        assert not loader._looks_like_ades(text), text
        obs, fmt = loader._parse(text, obscodes)
        assert fmt == "OBS80"
        assert len(obs) == text.count("W84"), (len(obs), text)


def test_ades_psv_and_xml_are_still_ades():
    psv = ("# version=2017\n# observatory\n! mpcCode 695\n"
           "permID |provID   |obsTime                 |ra          |dec         "
           "|rmsRA |rmsDec|mag |band|stn |mode|astCat\n"
           "       |2001 KQ76|2001-05-24T00:19:31.600Z|180.000000  |10.000000   "
           "|0.500 |0.500 |20.0|V   |695 |CCD |Gaia2\n")
    assert loader._looks_like_ades(psv)
    obs, fmt = loader._parse(psv, engine_api._obscodes())
    assert fmt == "ADES" and len(obs) == 1
    assert loader._looks_like_ades('<?xml version="1.0"?>\n<ades version="2017"/>')
    assert loader._looks_like_ades("<obsBlock></obsBlock>")
