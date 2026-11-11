"""Background snapshot preparation; policy adoption stays at lap boundaries."""
import json
import os
import threading


class PolicyReloader:
    """Prepare immutable snapshots off the control thread; adopt only on request."""

    def __init__(self, model_root, load_policy, model_dir='', warn=lambda msg: None):
        self.model_root = os.path.abspath(os.path.expanduser(model_root))
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir)) if model_dir else ''
        self.load_policy, self.warn = load_policy, warn
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._prepared = None
        self._highest_version = -1
        self._attempted = None
        self._thread = None

    def poll_once(self):
        if self.model_dir:
            with self._lock:
                if self._highest_version >= 0:
                    return
            snapshot, version = self.model_dir, None
        else:
            with open(os.path.join(self.model_root, 'latest.json')) as stream:
                pointer = json.load(stream)
            version = int(pointer['version'])
            snapshot = os.path.expanduser(pointer['path'])
            if not os.path.isabs(snapshot):
                snapshot = os.path.join(self.model_root, snapshot)
            elif not os.path.isdir(snapshot):
                snapshot = os.path.join(self.model_root, os.path.basename(snapshot))
            with self._lock:
                if version <= self._highest_version:
                    return
        snapshot = os.path.abspath(snapshot)
        signature = (snapshot, version, tuple(
            (os.stat(os.path.join(snapshot, name)).st_mtime_ns,
             os.stat(os.path.join(snapshot, name)).st_size)
            for name in ('deploy.pt', 'spec.json')))
        if signature == self._attempted:
            return
        self._attempted = signature
        policy = self.load_policy(snapshot)
        if version is not None and int(policy.version) != version:
            raise ValueError('latest.json version differs from snapshot manifest')
        with self._lock:
            if int(policy.version) > self._highest_version:
                self._prepared = policy
                self._highest_version = int(policy.version)

    def take_ready(self):
        with self._lock:
            candidate, self._prepared = self._prepared, None
        return candidate

    def _run(self):
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                self.warn('policy preparation failed; keeping current policy: {}'.format(exc))
            self._stop.wait(1.)

    def start(self):
        self._thread = threading.Thread(target=self._run, name='policy-loader', daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.)

