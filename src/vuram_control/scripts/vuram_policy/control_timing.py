"""Small bounded timing snapshot for diagnosing control deadline failures."""
from contextlib import contextmanager
import threading
import time


class ControlTiming:
    def __init__(self):
        self._lock = threading.Lock()
        self._phase = 'starting'
        self._started = time.monotonic()
        self._last = {}

    @contextmanager
    def measure(self, phase):
        started, cpu_started = time.monotonic(), time.thread_time()
        with self._lock:
            self._phase, self._started = phase, started
        try:
            yield
        finally:
            wall, cpu = time.monotonic() - started, time.thread_time() - cpu_started
            with self._lock:
                self._last[phase] = (wall, cpu)
                self._phase, self._started = 'between stages', time.monotonic()

    def summary(self):
        # Never hold the lock during a measured operation, including inference
        # and sleep, so the watchdog can identify an operation still in flight.
        with self._lock:
            phase, started, last = self._phase, self._started, tuple(self._last.items())
        stages = ', '.join('{}={:.1f}/{:.1f}ms'.format(name, wall * 1000., cpu * 1000.)
                           for name, (wall, cpu) in last)
        return 'phase={} phase_age={:.1f}ms last_wall/cpu=[{}]'.format(
            phase, (time.monotonic() - started) * 1000., stages)
