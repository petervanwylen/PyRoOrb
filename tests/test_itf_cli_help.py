"""Every ITF subcommand's help renders.

A bare '%' in an argparse help string is a format error: Python 3.14 refuses it when the
option is defined (the whole CLI then fails to start), older versions when the help is
printed. Printing every subcommand's help catches it on any version."""
import pytest

from pyroorb.itf.__main__ import main

SUBCOMMANDS = ["search", "train-model", "train-audit", "train-compare", "train-eval",
               "train-recalibrate", "recovery-suite"]


@pytest.mark.parametrize("sub", SUBCOMMANDS)
def test_subcommand_help_renders(sub, capsys):
    with pytest.raises(SystemExit) as exit_:
        main([sub, "--help"])
    assert exit_.value.code == 0
    assert "usage:" in capsys.readouterr().out
