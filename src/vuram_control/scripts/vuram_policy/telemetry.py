"""Thread-safe telemetry checks using source and monotonic receipt clocks.

No ROS imports are needed. Startup requires a whole window of advancing source
timestamps; a single fresh packet cannot qualify a slow or interrupted stream.
"""
from collections import deque
import math
import threading


class StreamMonitor:
    """Monitor one callback stream without imposing a sample-count limit.

    ``push`` accepts the message header time and ``time.monotonic()`` at receipt.
    Public readers take one immutable snapshot under the lock. A timestamp reset
    invalidates startup qualification only until a new healthy window is seen.
    """

    def __init__(self, name='odometry', window=1.0):
        self.name = str(name)
        self.window = self._positive(window, 'window')
        self._samples = deque()
        self._segment_start = None
        self._last_advance = None
        self._rollback = None
        self._lock = threading.Lock()

    @staticmethod
    def _positive(value, name):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('{} must be positive and finite'.format(name))
        return value

    def push(self, stamp, received_monotonic):
        """Record a callback; only an identical (stamp, receipt) is ignored.

        Repeated stamps at later receipt times are retained as evidence of a
        frozen source, and restart the continuous startup qualification period.
        Receipt times must be monotonic, as they are in a serialized ROS callback.
        """
        stamp, received = float(stamp), float(received_monotonic)
        if not math.isfinite(stamp) or not math.isfinite(received):
            raise ValueError('source and receipt timestamps must be finite')
        with self._lock:
            previous = self._samples[-1] if self._samples else None
            if previous == (stamp, received):
                return
            if previous is not None and received < previous[1]:
                raise ValueError('received_monotonic moved backwards')
            if previous is None:
                self._segment_start = received
                self._last_advance = received
            elif stamp <= previous[0]:
                self._segment_start = received
                if stamp < previous[0]:
                    self._rollback = (received, previous[0] - stamp)
                    # A reset establishes a new epoch. It must still produce a
                    # complete healthy window before startup can recover.
                    self._last_advance = received
            else:
                self._last_advance = received
            self._samples.append((stamp, received))
            cutoff = received - 2.0 * self.window
            # Keep one predecessor of the retained time interval so a gap
            # spanning the boundary remains observable at any input frequency.
            while len(self._samples) > 2 and self._samples[1][1] < cutoff:
                self._samples.popleft()

    def _snapshot(self):
        with self._lock:
            return (tuple(self._samples), self._segment_start,
                    self._last_advance, self._rollback)

    def _freshness(self, snapshot, now, ros_now, timeout):
        samples, _, advanced, rollback = snapshot
        if not samples:
            return '{} has no samples'.format(self.name)
        stamp, received = samples[-1]
        receive_age = now - received
        if receive_age > timeout:
            return ('{} receive timeout: last packet age={:.3f}s '
                    '(limit {:.3f}s)').format(self.name, receive_age, timeout)
        if rollback is not None and now - rollback[0] < self.window:
            return ('{} timestamp moved backwards by {:.6f}s '
                    '(event age={:.3f}s)').format(self.name, rollback[1], now - rollback[0])
        advance_age = now - advanced
        if advance_age > timeout:
            return ('{} timestamp stalled: last advance age={:.3f}s '
                    '(limit {:.3f}s)').format(self.name, advance_age, timeout)
        header_age = ros_now - stamp
        if header_age > timeout:
            return ('{} header is stale: age={:.3f}s '
                    '(limit {:.3f}s)').format(self.name, header_age, timeout)
        if header_age < -timeout:
            return ('{} header timestamp is ahead of the ROS clock: '
                    'lead={:.3f}s (limit {:.3f}s)').format(self.name, -header_age, timeout)
        return None

    def freshness_reason(self, now, ros_now, timeout):
        """Return a current fault, or None, without requiring startup coverage."""
        timeout = self._positive(timeout, 'timeout')
        return self._freshness(self._snapshot(), float(now), float(ros_now), timeout)

    def _metrics(self, snapshot, continuous=False):
        samples, segment_start, _, _ = snapshot
        if continuous:
            samples = tuple(sample for sample in samples if sample[1] >= segment_start)
        if not samples:
            return dict(receive_rate=0., source_rate=0., receive_gap=0.,
                        source_gap=0., coverage=0.)
        # Anchor the averaging interval on the most recent callback. Freshness
        # separately checks its age; polling between callbacks must not change
        # an otherwise healthy stream's measured frequency or qualification.
        cutoff = samples[-1][1] - self.window
        start = 0
        while start + 1 < len(samples) and samples[start + 1][1] < cutoff:
            start += 1
        samples = samples[start:]
        count = len(samples) - 1
        coverage = samples[-1][1] - samples[0][1]
        source_span = samples[-1][0] - samples[0][0]
        receive_gaps = [right[1] - left[1] for left, right in zip(samples, samples[1:])]
        source_gaps = [right[0] - left[0] for left, right in zip(samples, samples[1:])]
        # A repeated/backwards source timestamp is not an effective new state.
        # Raw receipt Hz remains useful for diagnosing a frozen publisher.
        source_valid = bool(count) and all(gap > 0 for gap in source_gaps)
        return dict(receive_rate=count / coverage if coverage > 0 else 0.,
                    source_rate=count / source_span if source_valid else 0.,
                    receive_gap=max(receive_gaps, default=0.),
                    source_gap=max(source_gaps, default=0.), coverage=coverage)

    def ready_reason(self, now, ros_now, min_rate, max_gap, timeout):
        """Return why startup should remain neutral, or None when qualified.

        Both receipt-clock and source-clock rates/gaps must meet the limits over
        at least ``window`` seconds of strictly advancing source timestamps.
        """
        min_rate = self._positive(min_rate, 'min_rate')
        max_gap = self._positive(max_gap, 'max_gap')
        timeout = self._positive(timeout, 'timeout')
        snapshot = self._snapshot()
        reason = self._freshness(snapshot, float(now), float(ros_now), timeout)
        if reason:
            return reason
        metrics = self._metrics(snapshot, continuous=True)
        # Subtractions of ROS epoch timestamps and synthetic decimal samples can
        # round a boundary by sub-microseconds; do not reject exact-rate streams.
        epsilon = 1e-6
        if metrics['coverage'] + epsilon < self.window:
            return ('{} needs {:.3f}s of continuous advancing timestamps: '
                    'coverage={:.3f}s').format(self.name, self.window, metrics['coverage'])
        if metrics['receive_rate'] + epsilon < min_rate:
            return ('{} receive rate too low: {:.2f}Hz < {:.2f}Hz').format(
                self.name, metrics['receive_rate'], min_rate)
        if metrics['source_rate'] + epsilon < min_rate:
            return ('{} source timestamp rate too low: {:.2f}Hz < {:.2f}Hz').format(
                self.name, metrics['source_rate'], min_rate)
        if metrics['receive_gap'] > max_gap + epsilon:
            return ('{} receive gap too large: {:.3f}s > {:.3f}s').format(
                self.name, metrics['receive_gap'], max_gap)
        if metrics['source_gap'] > max_gap + epsilon:
            return ('{} source timestamp gap too large: {:.3f}s > {:.3f}s').format(
                self.name, metrics['source_gap'], max_gap)
        return None

    def summary(self, now):
        """Return concise receipt/source Hz, maximum gaps, coverage and age."""
        snapshot = self._snapshot()
        metrics = self._metrics(snapshot)
        age = float(now) - snapshot[0][-1][1] if snapshot[0] else float('inf')
        return ('{}: receive={:.2f}Hz source={:.2f}Hz '
                'max_gap(receive/source)={:.3f}/{:.3f}s coverage={:.3f}s '
                'receive_age={:.3f}s').format(
                    self.name, metrics['receive_rate'], metrics['source_rate'],
                    metrics['receive_gap'], metrics['source_gap'], metrics['coverage'], age)
