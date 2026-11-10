"""Telemetry qualification and diagnostics without ROS or a connected vehicle."""
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from vuram_policy.telemetry import StreamMonitor


def feed(monitor, rate=30., duration=2., start=10., stamp_start=1000.):
    for offset in np.arange(round(duration * rate) + 1) / rate:
        monitor.push(stamp_start + offset, start + offset)
    return start + offset, stamp_start + offset


def ready(monitor, now, ros_now, min_rate=20., max_gap=.1):
    return monitor.ready_reason(now, ros_now, min_rate, max_gap, .2)


def test_normal_30hz_is_ready_and_summary_has_both_clocks():
    monitor = StreamMonitor()
    now, stamp = feed(monitor)
    assert ready(monitor, now, stamp) is None
    assert ready(monitor, now + .02, stamp + .02) is None
    summary = monitor.summary(now)
    assert 'receive=30.00Hz source=30.00Hz' in summary
    assert 'max_gap(receive/source)=0.033/0.033s' in summary
    assert 'coverage=' in summary


def test_low_4hz_stream_cannot_start_on_one_fresh_packet():
    monitor = StreamMonitor()
    now, stamp = feed(monitor, rate=4.)
    assert 'receive rate too low: 4.00Hz' in ready(monitor, now, stamp)
    assert 'receive timeout' in monitor.freshness_reason(now + .21, stamp + .21, .2)


def test_no_data_and_one_lucky_packet_never_qualify():
    monitor = StreamMonitor()
    assert 'no samples' in ready(monitor, 10., 1000.)
    monitor.push(1000., 10.)
    assert 'coverage=0.000s' in ready(monitor, 10., 1000.)
    # Freshness is deliberately usable before a complete startup window.
    assert monitor.freshness_reason(10., 1000., .2) is None
    assert 'last packet age=0.250s' in monitor.freshness_reason(10.25, 1000.25, .2)


def test_exact_duplicate_push_is_idempotent():
    monitor = StreamMonitor()
    now, stamp = feed(monitor)
    before = monitor.summary(now)
    monitor.push(stamp, now)
    assert monitor.summary(now) == before
    assert ready(monitor, now, stamp) is None


def test_same_source_stamp_at_new_receipts_does_not_count_as_valid_data():
    monitor = StreamMonitor()
    now, stamp = feed(monitor)
    for elapsed in np.arange(1, 10) / 30.:
        monitor.push(stamp, now + elapsed)
    reason = monitor.freshness_reason(now + elapsed, stamp + elapsed, .2)
    assert 'timestamp stalled' in reason
    assert 'last advance age=0.300s' in reason
    assert 'source=0.00Hz' in monitor.summary(now + elapsed)
    assert ready(monitor, now + elapsed, stamp + elapsed) is not None


@pytest.mark.parametrize('delta, expected', [(.3, 'header is stale: age=0.300s'),
                                             (-.3, 'header timestamp is ahead')])
def test_header_delay_and_future_time_are_distinct_from_receipt_timeout(delta, expected):
    monitor = StreamMonitor('imu')
    now, stamp = feed(monitor)
    reason = monitor.freshness_reason(now, stamp + delta, .2)
    assert expected in reason
    assert reason.startswith('imu ')


def test_receive_gap_is_rejected_even_when_average_rate_is_high():
    monitor = StreamMonitor()
    feed(monitor, rate=100., duration=1.)
    # A 150 ms hole, then enough fast samples for a high window-average rate.
    for offset in np.arange(115, 161) / 100.:
        monitor.push(1000. + offset, 10. + offset)
    reason = ready(monitor, 11.6, 1001.6)
    assert 'receive gap too large: 0.150s' in reason
    # The fault naturally ages out after a complete healthy window.
    for offset in np.arange(161, 231) / 100.:
        monitor.push(1000. + offset, 10. + offset)
    assert ready(monitor, 12.3, 1002.3) is None


def test_source_clock_rate_and_gap_are_checked_independently():
    monitor = StreamMonitor()
    for index in range(31):
        monitor.push(1000. + index / 10., 10. + index / 30.)
    assert 'source timestamp rate too low: 10.00Hz' in ready(monitor, 11., 1003.)
    monitor = StreamMonitor()
    # Receipt cadence is regular. One forward source-clock jump violates the
    # source gap limit while its average rate still exceeds 20 Hz.
    for index in range(101):
        monitor.push(1000. + index / 100. + (.15 if index >= 50 else 0.),
                     10. + index / 100.)
    assert 'source timestamp gap too large: 0.160s' in ready(monitor, 11., 1001.15)


def test_timestamp_rollback_is_reported_and_startup_can_recover():
    monitor = StreamMonitor()
    now, stamp = feed(monitor, duration=1.)
    monitor.push(stamp - .1, now + .01)
    assert 'timestamp moved backwards by 0.100000s' in monitor.freshness_reason(
        now + .01, stamp - .1, .2)
    # A changed source epoch becomes healthy after a full advancing window.
    for index in range(1, 32):
        monitor.push(stamp - .1 + index / 30., now + .01 + index / 30.)
    assert ready(monitor, now + .01 + 31 / 30., stamp - .1 + 31 / 30.) is None


def test_repeated_timestamp_resets_startup_coverage_then_recovers():
    monitor = StreamMonitor()
    now, stamp = feed(monitor, duration=1.)
    monitor.push(stamp, now + .01)
    assert 'coverage=0.000s' in ready(monitor, now + .01, stamp)
    for index in range(1, 32):
        monitor.push(stamp + index / 30., now + .01 + index / 30.)
    assert ready(monitor, now + .01 + 31 / 30., stamp + 31 / 30.) is None


def test_retention_is_time_based_at_high_frequency_and_keeps_predecessor():
    monitor = StreamMonitor(window=1.)
    now, stamp = feed(monitor, rate=2000., duration=5.)
    retained = monitor._snapshot()[0]
    assert len(retained) > 4000  # No fixed-size deque truncating coverage.
    cutoff = now - 2.
    assert retained[0][1] < cutoff <= retained[1][1]
    assert ready(monitor, now, stamp) is None


def test_threaded_writer_and_readers_take_consistent_snapshots():
    monitor = StreamMonitor()
    errors = []
    started = threading.Event()

    def write():
        try:
            started.set()
            feed(monitor, rate=1000., duration=3.)
        except Exception as exc:
            errors.append(exc)

    def read():
        started.wait(2.)
        try:
            for _ in range(300):
                snapshot = monitor._snapshot()[0]
                if snapshot:
                    stamp, now = snapshot[-1]
                    monitor.summary(now)
                    monitor.freshness_reason(now, stamp, .2)
                    ready(monitor, now, stamp)
        except Exception as exc:
            errors.append(exc)

    workers = [threading.Thread(target=write)] + [threading.Thread(target=read) for _ in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10.)
        assert not worker.is_alive()
    assert not errors
    assert ready(monitor, 13., 1003.) is None


@pytest.mark.parametrize('window', [0., -1., float('inf'), float('nan')])
def test_invalid_monitor_window_rejected(window):
    with pytest.raises(ValueError):
        StreamMonitor(window=window)


def test_invalid_timestamps_and_backwards_receipt_clock_rejected():
    monitor = StreamMonitor()
    with pytest.raises(ValueError, match='finite'):
        monitor.push(float('nan'), 10.)
    monitor.push(1000., 10.)
    with pytest.raises(ValueError, match='received_monotonic moved backwards'):
        monitor.push(1001., 9.)
