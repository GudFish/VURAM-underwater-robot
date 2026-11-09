"""Protocol tests use ephemeral localhost sockets, never the configured DVL IP."""
import json
import socket
import threading
import time

from dvl_odometry.transport import JsonClient


def server(handler):
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(2)
    port = listener.getsockname()[1]
    def run():
        try:
            connection, _ = listener.accept()
            with connection:
                handler(connection)
        finally:
            listener.close()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return port, thread


def wait_connected(client):
    deadline = time.monotonic()+2.
    while not client.connected and time.monotonic() < deadline:
        time.sleep(.005)
    assert client.connected


def test_fragmented_json_bad_reports_and_serialized_commands():
    requests, reports, errors = [], [], []
    def handler(connection):
        source = connection.makefile('rb')
        for _ in range(2):
            request = json.loads(source.readline())
            requests.append(request['command'])
            connection.sendall(b'bad-json\n[]\n{"type":"vel')
            time.sleep(.01)
            connection.sendall(b'ocity","vx":1}\n')
            response = dict(type='response', response_to=request['command'], success=True)
            connection.sendall((json.dumps(response)+'\n').encode())
        time.sleep(.1)
    port, thread = server(handler)
    client = JsonClient('127.0.0.1', port, reports.append, lambda _: None, errors.append)
    client.start()
    try:
        wait_connected(client)
        first = client.submit({'command': 'reset_dead_reckoning'}, 1.)
        second = client.submit({'command': 'get_config'}, 1.)
        assert first.event.wait(2.) and first.ok
        assert second.event.wait(2.) and second.ok
        assert requests == ['reset_dead_reckoning', 'get_config']
        assert len(reports) == 2 and len(errors) >= 4
    finally:
        client.close()
        thread.join(timeout=2.)


def test_timeout_reconnects_instead_of_reusing_a_late_ack():
    requests = []
    def handler(connection):
        source = connection.makefile('rb')
        requests.append(json.loads(source.readline()))
        time.sleep(.3)
        try:
            connection.sendall(b'{"type":"response","response_to":"reset_dead_reckoning","success":true}\n')
        except OSError:
            pass
    port, thread = server(handler)
    client = JsonClient('127.0.0.1', port, lambda _: None, lambda _: None, lambda _: None)
    client.start()
    try:
        wait_connected(client)
        first = client.submit({'command': 'reset_dead_reckoning'}, .1)
        second = client.submit({'command': 'reset_dead_reckoning'}, 1.)
        assert first.event.wait(2.) and not first.ok
        assert second.event.wait(2.) and not second.ok
        assert len(requests) == 1  # second reset never misattributes the delayed first ACK
    finally:
        client.close()
        thread.join(timeout=2.)
