"""Control timing diagnostics remain readable while a stage is stalled."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from vuram_policy import control_timing


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(wall=10., cpu=1.)
    monkeypatch.setattr(control_timing, 'time', SimpleNamespace(
        monotonic=lambda: clock.wall, thread_time=lambda: clock.cpu))
    return clock


def test_reports_live_stage_and_wall_vs_cpu_time(clock):
    timing = control_timing.ControlTiming()
    with timing.measure('policy_inference'):
        clock.wall += .125
        clock.cpu += .003
        summary = timing.summary()  # a watchdog may read before stage completion
        assert 'phase=policy_inference phase_age=125.0ms' in summary
    summary = timing.summary()
    assert 'phase=between stages' in summary
    assert 'policy_inference=125.0/3.0ms' in summary


def test_keeps_only_latest_measurement_of_each_stage(clock):
    timing = control_timing.ControlTiming()
    for elapsed in (.050, .010):
        with timing.measure('wait_next_tick'):
            clock.wall += elapsed
    summary = timing.summary()
    assert summary.count('wait_next_tick=') == 1
    assert 'wait_next_tick=10.0/0.0ms' in summary


def test_exception_still_records_completed_stage_timing(clock):
    timing = control_timing.ControlTiming()
    with pytest.raises(RuntimeError):
        with timing.measure('command_publish'):
            clock.wall += .101
            raise RuntimeError('late command')
    assert 'command_publish=101.0/0.0ms' in timing.summary()
