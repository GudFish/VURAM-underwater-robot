#!/usr/bin/env python3
"""Deterministic A50 TCP JSON simulator; no rospy dependency.

Commands: sim_dropout, sim_invalid, sim_pause_velocity, sim_pause_dr,
sim_malformed. Fault commands take parameters.seconds (default 2).
Movement is global, never stepped per connected client. Separate test clients
can issue faults without changing trajectory speed or closing the driver's socket.
"""
import argparse
import json
import socket
import threading
import time

import numpy as np
from dvl_odometry.simulation import trajectory, rpy_rotation, matrix_rpy
from dvl_odometry.core import FLIP


class DvlSimulator:
    def __init__(self, host='127.0.0.1', port=16171, velocity_rate=10., dr_rate=5.,
                 start_time=None, speed=0.3, acceleration=0., yaw_rate=0.1, roll=0., pitch=0.):
        if min(velocity_rate, dr_rate) <= 0:
            raise ValueError('report rates must be positive')
        self.host, self.port = host, port
        self.velocity_rate, self.dr_rate = velocity_rate, dr_rate
        self.start_time = start_time if start_time is not None else time.time()
        self.motion = dict(speed=speed, acceleration=acceleration, yaw_rate=yaw_rate, roll=roll, pitch=pitch)
        self.lock = threading.Lock()
        self.clients = set()
        self.invalid_until = self.pause_velocity_until = self.pause_dr_until = 0.
        self.reset_position = np.zeros(3)
        self.reset_rotation = np.eye(3)
        self.config = dict(speed_of_sound=1475., mounting_rotation_offset=0., acoustic_enabled=True,
                           dark_mode_enabled=False, range_mode='auto', periodic_cycling_enabled=True)

    def state(self):
        return trajectory(max(0., time.time()-self.start_time), **self.motion)

    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(8)
        print('DVL simulator listening on %s:%d' % (self.host, self.port), flush=True)
        while True:
            client, _ = server.accept()
            with self.lock:
                self.clients.add(client)
            threading.Thread(target=self.serve, args=(client,), daemon=True).start()

    def velocity_report(self):
        _, _, v, _, _ = self.state()
        v = FLIP.dot(v)
        now = time.time()
        return dict(type='velocity' if self.config['range_mode'] != 'wt' else 'velocity_water',
                    format='json_v3.3', time=1000./self.velocity_rate,
                    vx=v[0], vy=v[1], vz=v[2], fom=0.01, altitude=2., status=0,
                    covariance=np.diag([1e-4]*3).tolist(),
                    velocity_valid=now >= self.invalid_until and self.config['acoustic_enabled'],
                    transducers=[dict(id=i, velocity=0., distance=2., rssi=-30., nsd=-90., beam_valid=True)
                                 for i in range(4)],
                    time_of_validity=int(now*1e6), time_of_transmission=int(now*1e6))

    def dr_report(self):
        p, rpy, _, _, _ = self.state()
        with self.lock:
            p = FLIP.dot(self.reset_rotation.T.dot(p-self.reset_position))
            relative = self.reset_rotation.T.dot(rpy_rotation(*rpy))
            roll, pitch, yaw = matrix_rpy(relative)
            rpy_deg = np.degrees([roll, -pitch, -yaw])
        return dict(type='position_local', format='json_v3.3', ts=time.time(),
                    x=p[0], y=p[1], z=p[2], roll=rpy_deg[0], pitch=rpy_deg[1], yaw=rpy_deg[2],
                    std=0.01, status=0)

    @staticmethod
    def send(client, payload):
        client.sendall((json.dumps(payload)+'\n').encode())

    def command(self, client, message):
        name = message.get('command')
        params = message.get('parameters', {})
        result, ok, error = None, True, ''
        duration = max(0., float(params.get('seconds', 2.)))
        if name == 'sim_dropout':
            with self.lock:
                clients = list(self.clients)
            for target in clients:
                try:
                    target.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            return
        if name in ('sim_invalid', 'sim_pause_velocity', 'sim_pause_dr'):
            attr = {'sim_invalid': 'invalid_until', 'sim_pause_velocity': 'pause_velocity_until',
                    'sim_pause_dr': 'pause_dr_until'}[name]
            setattr(self, attr, time.time()+duration)
        elif name == 'sim_malformed':
            with self.lock:
                for target in list(self.clients):
                    try:
                        target.sendall(b'not-json\n[]\n{"type":"velocity","velocity_valid":true,"vx":"bad"}\n')
                    except OSError:
                        pass
        elif name == 'reset_dead_reckoning':
            time.sleep(0.05)
            p, rpy, _, _, _ = self.state()
            with self.lock:
                self.reset_position = p
                self.reset_rotation = rpy_rotation(*rpy)
        elif name == 'calibrate_gyro':
            time.sleep(2.)
        elif name == 'get_config':
            result = self.config.copy()
        elif name == 'set_config':
            self.config.update({key: value for key, value in params.items() if key in self.config})
        elif name == 'get_version_info':
            result = dict(product_name='DVL A50 simulator', version_short='2.7.2', is_ready=True)
        else:
            ok, error = False, 'unknown command'
        self.send(client, dict(type='response', format='json_v3.3', response_to=name,
                               success=ok, error_message=error, result=result))

    def serve(self, client):
        client.settimeout(0.005)
        buffer = b''
        next_velocity = next_dr = 0.
        try:
            while True:
                now = time.time()
                if now >= next_velocity:
                    next_velocity = now+1./self.velocity_rate
                    if now >= self.pause_velocity_until:
                        self.send(client, self.velocity_report())
                if now >= next_dr:
                    next_dr = now+1./self.dr_rate
                    if now >= self.pause_dr_until:
                        self.send(client, self.dr_report())
                try:
                    data = client.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break
                buffer += data
                if len(buffer) > 1 << 20:
                    break
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    try:
                        self.command(client, json.loads(line))
                    except (ValueError, TypeError, AttributeError):
                        continue
        except OSError:
            pass
        finally:
            with self.lock:
                self.clients.discard(client)
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=16171)
    parser.add_argument('--velocity-rate', type=float, default=10.)
    parser.add_argument('--dr-rate', type=float, default=5.)
    parser.add_argument('--start-time', type=float)
    parser.add_argument('--speed', type=float, default=0.3)
    parser.add_argument('--acceleration', type=float, default=0.)
    parser.add_argument('--yaw-rate', type=float, default=0.1)
    parser.add_argument('--roll', type=float, default=0.)
    parser.add_argument('--pitch', type=float, default=0.)
    args, _ = parser.parse_known_args()
    DvlSimulator(**vars(args)).run()


if __name__ == '__main__':
    main()
