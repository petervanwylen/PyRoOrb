"""The regression suite's timing guard never runs under continuous integration.

Its baselines are CPU seconds from one development machine; a CI runner is a
different machine, so only the correctness guards may fail a CI build."""
import pytest

from test_regression import _speed_guarded


class _Config:
    def __init__(self, speed=False, worker=False):
        self._speed = speed
        if worker:
            self.workerinput = {"workerid": "gw0"}

    def getoption(self, name):
        assert name == "--speed"
        return self._speed


@pytest.mark.parametrize("value", ["true", "1", "True", "yes"])
def test_ci_turns_the_speed_guard_off_even_with_speed(monkeypatch, value):
    monkeypatch.setenv("CI", value)
    assert _speed_guarded(_Config(speed=True)) is False
    assert _speed_guarded(_Config()) is False


@pytest.mark.parametrize("value", [None, "", "0", "false"])
def test_outside_ci_serial_runs_and_speed_keep_the_guard(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CI", raising=False)
    else:
        monkeypatch.setenv("CI", value)
    assert _speed_guarded(_Config()) is True
    assert _speed_guarded(_Config(speed=True, worker=True)) is True
    assert _speed_guarded(_Config(worker=True)) is False
