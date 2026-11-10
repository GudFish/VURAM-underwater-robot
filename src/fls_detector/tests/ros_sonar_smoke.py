#!/usr/bin/env python3
"""End-to-end smoke test on a private ROS master; no vehicle/SITL required."""
import json
import math
import os
import signal
import socket
import subprocess
import tempfile
import time
import xmlrpc.client

import numpy as np


def main():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    uri = f'http://127.0.0.1:{port}'
    os.environ['ROS_MASTER_URI'] = uri
    processes = []
    with tempfile.TemporaryDirectory(prefix='bhq750d_ros_') as temporary:
        os.environ['ROS_HOME'] = temporary
        os.environ['ROS_LOG_DIR'] = temporary + '/logs'
        log = open(temporary + '/test.log', 'w+')
        try:
            def start(args):
                p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(p)
                return p
            start(['roscore', '-p', str(port)])
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    if xmlrpc.client.ServerProxy(uri).getPid('/sonar_test')[0] == 1:
                        break
                except OSError:
                    pass
                time.sleep(.1)
            else:
                raise AssertionError('private roscore failed to start')
            import rospy
            from geometry_msgs.msg import PoseStamped
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import Image
            from std_msgs.msg import String, Float64MultiArray, Float32MultiArray
            from visualization_msgs.msg import Marker, MarkerArray
            rospy.init_node('bhq750d_smoke', anonymous=True, disable_signals=True)
            latest, samples, candidates = {}, [], []
            subscribers = []
            for key in ('image', 'image_raw', 'instance_raw', 'direct_raw', 'wrt_raw', 'prt_raw'):
                subscribers.append(rospy.Subscriber('/forward_sonar/' + key, Image,
                                   lambda msg, key=key: latest.__setitem__(key, msg), queue_size=1))
            subscribers.append(rospy.Subscriber('/forward_sonar/metadata', String,
                               lambda msg: latest.__setitem__('metadata', json.loads(msg.data)), queue_size=1))
            subscribers.append(rospy.Subscriber('/forward_sonar/render_stats', Float64MultiArray,
                               lambda msg: samples.append(msg.data[1]), queue_size=100))
            subscribers.append(rospy.Subscriber('/forward_sonar/scene_candidates', Float32MultiArray,
                               lambda msg: candidates.append(list(msg.data)), queue_size=10))
            odom_pub = rospy.Publisher('/mavros/local_position/odom', Odometry, queue_size=1, latch=True)
            target_pub = rospy.Publisher('/dynamic_target/pose', PoseStamped, queue_size=1, latch=True)
            obstacle_pub = rospy.Publisher('/dynamic_obstacles/markers', MarkerArray, queue_size=1, latch=True)
            launch = start(['roslaunch', 'fls_detector', 'bhq750d_compare.launch', 'show_window:=false'])
            odom, target = Odometry(), PoseStamped()
            odom.header.frame_id = target.header.frame_id = 'map'
            odom.pose.pose.orientation.w = target.pose.orientation.w = 1.
            target.pose.position.x = 8.
            # Broadside aspect for the existing detector's 40-pixel region threshold.
            target.pose.orientation.z = target.pose.orientation.w = math.sqrt(.5)

            def pump(seconds=1.2):
                stop = time.monotonic() + seconds
                while time.monotonic() < stop:
                    assert launch.poll() is None, 'ROS launch exited early'
                    odom.header.stamp = target.header.stamp = rospy.Time.now()
                    odom_pub.publish(odom)
                    target_pub.publish(target)
                    time.sleep(.05)
            def labels():
                return np.frombuffer(latest['instance_raw'].data, dtype='<u2')
            pump(3.)
            for _ in range(10):
                if any(data and data[0] > 0 for data in candidates):
                    break
                pump(.5)
            assert {'image', 'image_raw', 'instance_raw', 'metadata'} <= latest.keys(), 'missing topics'
            assert latest['image'].width == 512 and latest['image'].height == 256
            assert latest['image_raw'].width == 256 and latest['image_raw'].height == 494
            assert latest['image'].encoding == latest['image_raw'].encoding == 'mono8'
            assert latest['metadata']['vertical_fov_deg'] == 20.
            assert len(latest['metadata']['beam_angles_deg']) == 256
            assert rospy.get_param('/forward_sonar_sim/noise_beam_bias_db') == []
            assert latest['metadata']['noise_profile_range_m'] == 10.
            native = latest['image_raw']
            raw = np.frombuffer(native.data, np.uint8).reshape(native.height, native.step)[:, :native.width]
            center = raw[40:450, 126:131].mean()
            flanks = np.concatenate([raw[40:450, 118:123], raw[40:450, 134:139]], axis=1).mean()
            assert abs(center-flanks) < 5., 'center stripe persists in measured background'
            assert 1 in labels(), 'target absent from direct instance mask'
            assert any(data and data[0] > 0 for data in candidates), 'existing detector produced no scene candidates'
            assert max(np.frombuffer(latest['image'].data, np.uint8)) > 80, 'display transfer hides target'
            # Block the entire target using a single foreground object.
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.ns, marker.id, marker.type = 'test', 10, Marker.CUBE
            marker.pose.orientation.w = 1.
            marker.pose.position.x = 4.
            marker.scale.x = marker.scale.y = marker.scale.z = 2.
            obstacle_pub.publish(MarkerArray(markers=[marker])); pump()
            assert 1 not in labels(), 'occluded target leaked through obstacle'
            marker.action = Marker.DELETE
            obstacle_pub.publish(MarkerArray(markers=[marker])); pump()
            assert 1 in labels(), 'marker DELETE did not restore target'
            marker.action = Marker.ADD
            marker.lifetime = rospy.Duration(1.)
            marker.header.stamp = rospy.Time.from_sec(1.)
            obstacle_pub.publish(MarkerArray(markers=[marker])); pump(.4)
            assert 1 not in labels(), 'marker lifetime must start at receipt, not the historical header stamp'
            pump(1.)
            assert 1 in labels(), 'expired marker still occludes target'
            target.pose.position.z = 3.
            pump(); assert 1 not in labels(), 'target outside vertical aperture still visible'
            pitch = -math.atan2(3., 8.)
            odom.pose.pose.orientation.y, odom.pose.pose.orientation.w = math.sin(pitch/2), math.cos(pitch/2)
            pump(); assert 1 in labels(), 'platform pitch did not restore elevated target'
            # Native radii are slant distances, not the former horizontal x coordinate.
            m = latest['instance_raw']; coordinates = np.where(labels().reshape(m.height, m.width) == 1)[0]
            measured = np.median(coordinates) * latest['metadata']['range_resolution_m']
            assert abs(measured - math.hypot(8., 3.)) < .8, (measured, 'slant projection mismatch')
            odom.pose.pose.orientation.y, odom.pose.pose.orientation.w = 0., 1.
            target.pose.position.z = 0.
            markers = []
            for i in range(16):
                m = Marker();m.header.frame_id = 'map';m.ns = 'performance';m.id = i
                m.type = (Marker.SPHERE, Marker.CUBE, Marker.CYLINDER)[i % 3]
                angle = math.radians(-55 + 110*i/15)
                m.pose.position.x, m.pose.position.y = 6*math.cos(angle), 6*math.sin(angle)
                m.pose.orientation.w = 1.
                m.scale.x = m.scale.y = m.scale.z = .7
                markers.append(m)
            obstacle_pub.publish(MarkerArray(markers=markers));samples.clear();pump(4.)
            assert len(samples) >= 25, 'renderer did not sustain a live stream'
            p95 = float(np.percentile(samples[3:], 95))
            assert p95 < 100., f'ROS frame P95={p95:.1f} ms exceeds 100 ms'
            print(f'PASS ROS: native/Cartesian topics, metadata, detector, occlusion/DELETE/lifetime, pitch/slant, 16 obstacles; total P95={p95:.2f} ms')
        except Exception:
            if 'latest' in locals() and 'image' in latest:
                import cv2
                msg = latest['image']
                array = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)
                cv2.imwrite('/tmp/bhq750d_ros_failure.png', array)
                print('debug:', {k: (len(v.data) if hasattr(v, 'data') else v) for k, v in latest.items() if k != 'metadata'},
                      'candidates:', candidates[-3:], 'max intensity:', array.max())
            log.flush();log.seek(0)
            print(log.read()[-12000:])
            raise
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            log.close()


if __name__ == '__main__':
    main()
