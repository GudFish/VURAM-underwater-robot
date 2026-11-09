"""One TCP owner, serialized commands and monotonic deadlines (also under /clock)."""
import json
import socket
import threading
import time
from collections import deque

MAX_LINE_BUFFER = 1 << 20


class Command:
    def __init__(self, payload, timeout, callback):
        self.payload = payload
        self.deadline = time.monotonic() + timeout
        self.callback = callback
        self.event = threading.Event()
        self.ok = False
        self.error = ''

    def finish(self, ok, error=''):
        if self.event.is_set():
            return
        self.ok, self.error = ok, error
        if self.callback:
            self.callback(ok, error)
        self.event.set()


class JsonClient:
    def __init__(self, host, port, on_message, on_connection, on_error,
                 min_delay=1., max_delay=30.):
        self.host, self.port = host, port
        self.on_message, self.on_connection, self.on_error = on_message, on_connection, on_error
        self.min_delay, self.max_delay = min_delay, max_delay
        self.lock = threading.Lock()
        self.queue = deque()
        self.connected = False
        self.stop = threading.Event()
        self.socket = None

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def close(self):
        self.stop.set()
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def submit(self, payload, timeout=2., callback=None):
        command = Command(payload, timeout, callback)
        with self.lock:
            if self.connected:
                self.queue.append(command)
            else:
                command.finish(False, 'DVL not connected')
        return command

    def _run(self):
        delay = self.min_delay
        while not self.stop.is_set():
            pending = None
            try:
                self.socket = socket.create_connection((self.host, self.port), timeout=3.)
                self.socket.settimeout(0.05)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self.lock:
                    self.connected = True
                delay = self.min_delay
                self.on_connection(True)
                buffer = b''
                while not self.stop.is_set():
                    if pending and time.monotonic() >= pending.deadline:
                        pending.finish(False, 'response timeout; reconnecting to discard late ACK')
                        raise OSError('command response timeout')
                    if pending is None:
                        with self.lock:
                            pending = self.queue.popleft() if self.queue else None
                        if pending:
                            if time.monotonic() >= pending.deadline:
                                pending.finish(False, 'command expired before transmission')
                                pending = None
                            else:
                                self.socket.sendall((json.dumps(pending.payload)+'\n').encode())
                    try:
                        data = self.socket.recv(4096)
                    except socket.timeout:
                        continue
                    if not data:
                        raise OSError('peer closed connection')
                    buffer += data
                    if len(buffer) > MAX_LINE_BUFFER:
                        raise OSError('JSON line buffer overflow')
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        if not line.strip():
                            continue
                        try:
                            message = json.loads(line)
                            if not isinstance(message, dict):
                                raise ValueError('expected JSON object')
                            if message.get('type') == 'response':
                                if pending and message.get('response_to') == pending.payload['command']:
                                    pending.finish(message.get('success') is True,
                                                   str(message.get('error_message') or ''))
                                    pending = None
                            else:
                                self.on_message(message)
                        except (ValueError, TypeError, KeyError, OverflowError) as exc:
                            self.on_error(str(exc))
            except OSError as exc:
                self.on_error(str(exc))
            finally:
                with self.lock:
                    self.connected = False
                    waiting = list(self.queue)
                    self.queue.clear()
                if pending:
                    waiting.append(pending)
                for command in waiting:
                    command.finish(False, 'DVL disconnected')
                if self.socket:
                    self.socket.close()
                    self.socket = None
                self.on_connection(False)
            self.stop.wait(delay)
            delay = min(delay*2., self.max_delay)
