"""TorchScript VND policy loader + 20 Hz deployment loop (ArduSub/MAVROS).

The exported module (deploy.pt) has signature
forward(obs[21], history[20,18], valid) -> a[5] — a deterministic FiLM
policy conditioned on a GRU history encoder (latent z = 0 until `valid`
reaches the encoder context length). Obs and encoder normalization are
folded into the module; raw physical units go in. The collect reference mode
adds continuous real-robot figure8 episodes, HDF5 spooling and boundary-only
model updates. Legacy internal/external reference modes retain their loops.

动作契约（VND v4）：a[5] = 归一化机体力/力矩 [X,Y,Z,K,N]（无俯仰）。
经 ThrusterInterface.set_wrench_action 直接映射到 6DOF 通道 PWM
（X→forward、Y→lateral、Z→throttle、K→roll、N→yaw，符号按训练口径与
ArduSub 通道语义换算），发布到 /dof_cmd（复用 AUV_ctrl_6dof 的
set_6dof_cmd 链路，飞控 VECTORED 混控器完成推进器分配）。
"""
import hashlib
import json
import os
import time

import numpy as np
import rospy
from mavros_msgs.srv import ParamPull
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

from .reference_generator import ReferenceGenerator
from .spec import (build_obs, build_ref11, get_observation_encoding, load_spec,
                   obs_dim_from_layout)
from .thruster_interface import ThrusterInterface
from .traj_visualizer import TrajVisualizer
from .utils import world_to_body


def configure_inference_threads(num_threads):
    """Configure the CALLING thread before inference or model preparation.

    With the installed OpenMP build, configuring a loader thread does not change
    the control thread's thread budget. Both threads must call this explicitly.
    """
    import torch
    if isinstance(num_threads, bool) or int(num_threads) != num_threads or num_threads < 1:
        raise ValueError('inference_threads must be a positive integer')
    if torch.get_num_threads() != int(num_threads):
        torch.set_num_threads(int(num_threads))
    return torch


class PolicyInference(object):
    def __init__(self, model_dir, device='cpu', num_threads=2):
        # torch remains lazy for reference/observation-only users.
        self.torch = configure_inference_threads(num_threads)
        self.device = self.torch.device(device)
        if self.device.type == 'cuda' and not self.torch.cuda.is_available():
            rospy.logwarn('CUDA unavailable; using CPU for policy inference')
            self.device = self.torch.device('cpu')
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir))
        self.model = None
        self.version = None
        self.manifest = {}
        self.observation_encoding = None
        self.history_shape = (20, 18)
        self._load()

    @property
    def loaded(self):
        return self.model is not None

    def _load(self):
        pt_path = os.path.join(self.model_dir, 'deploy.pt')
        if not os.path.isfile(pt_path):
            rospy.logerr('no deploy.pt in %s', self.model_dir)
            return
        spec_json = os.path.join(self.model_dir, 'spec.json')
        with open(spec_json) as f:
            self.manifest = json.load(f)
        manifest = self.manifest
        self.observation_encoding = get_observation_encoding(manifest)
        if manifest['schema_version'] != 4:
            raise ValueError('expected a VND v4 deploy snapshot')
        if manifest.get('actor_future_spec'):
            raise ValueError('policy requires unsupported future-reference inputs')
        self.version = manifest['version']
        self.history_shape = tuple(manifest['history_shape'])
        if (manifest['obs_dim'] != 21 or
                manifest['action_dim'] != 5 or
                self.history_shape != (20, 18)):
            raise ValueError('expected obs[21], history[20,18] -> action[5]')
        if manifest['history_encoding'] != \
                'window_relative_position_v1':
            raise ValueError('unsupported history encoding')
        with open(pt_path, 'rb') as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != manifest['model_sha256']:
            raise ValueError('deploy snapshot digest mismatch')
        model = self.torch.jit.load(pt_path, map_location=self.device).eval()
        model_encoding = getattr(getattr(model, 'policy', None),
                                 'observation_encoding', None)
        if model_encoding != self.observation_encoding:
            raise ValueError('deploy manifest observation encoding differs from the scripted policy')
        self.model = model
        rospy.loginfo('loaded policy v%s (%s, device=%s) from %s',
                      self.version, self.observation_encoding, self.device,
                      self.model_dir)

    def validate_spec(self, spec):
        """Check timing/layout before any command can reach the vehicle.

        Both the ROS configuration and the snapshot must use body-relative inputs.
        """
        get_observation_encoding(spec)
        if spec['a_dim'] != 5 or obs_dim_from_layout(spec['obs_layout']) != 21:
            raise ValueError('ROS specification must have obs[21] and action[5]')
        dt = float(spec['dt'])
        if (not np.isfinite(dt) or dt <= 0 or
                not np.isclose(float(spec['control_rate']) * dt, 1.0)):
            raise ValueError('control_rate must equal 1 / dt')
        if not np.isclose(float(self.manifest['dt']), dt):
            raise ValueError('ROS control period differs from the deploy snapshot')

    def observe(self, state, reference, previous, layout):
        """Build body-relative observations in both reference loops."""
        return build_obs(state, reference, previous, layout)

    def warmup(self):
        """Exercise empty, partial and full history before a model is eligible.

        Several passes also let TorchScript finish profiling/optimization away
        from the control loop. Every prepared snapshot follows the same path.
        """
        obs = np.zeros(21, np.float32)
        history = np.zeros(self.history_shape, np.float32)
        for _ in range(3):
            for valid in range(self.history_shape[0] + 1):
                self.step(obs, history, valid)

    def step(self, obs, history, valid):
        """obs[21], history[20,18] (raw), valid int -> a[5]."""
        t = self.torch
        with t.no_grad():
            o = t.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)[None]
            h = t.as_tensor(np.asarray(history, dtype=np.float32), device=self.device)[None]
            v = t.tensor([int(valid)], dtype=t.int64, device=self.device)
            a = self.model(o, h, v)
        action = a.cpu().numpy()[0].astype(np.float64)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError('invalid deploy action: expected finite action[5]')
        return action


class PoseState(object):
    """MAVROS odometry snapshot: s[13] = [v_body6, quat_wxyz4, pos3].

    /mavros/local_position/odom 与 DVL odom 的 twist 均为机体系（FLU 轴，
    x 前 / y 左 / z 上），默认直接使用；~twist_in_world=true 时按四元数
    转到机体系（Gazebo pose_gt 场景）。
    """

    def __init__(self, odom_topic, twist_in_world=False, on_sample=None):
        self.twist_in_world = twist_in_world
        self.on_sample = on_sample
        self.pose_msg = None
        self._snapshot = None
        self.sub = rospy.Subscriber(odom_topic, Odometry, self._cb,
                                    queue_size=5)

    def _cb(self, msg):
        received = time.monotonic()
        if self.on_sample is not None:
            self.on_sample(msg.header.stamp.to_sec(), received)
        self._snapshot = (msg, received)
        self.pose_msg = msg

    def step(self):
        """Snapshot at the control tick; None until the first pose arrives."""
        snapshot = self._snapshot
        if snapshot is None:
            return None
        pm, received = snapshot
        pos = np.array([pm.pose.pose.position.x,
                        pm.pose.pose.position.y,
                        pm.pose.pose.position.z])
        q = pm.pose.pose.orientation
        q_wxyz = np.array([q.w, q.x, q.y, q.z])   # message xyzw -> state wxyz
        v = np.array([pm.twist.twist.linear.x,
                      pm.twist.twist.linear.y,
                      pm.twist.twist.linear.z])
        w = np.array([pm.twist.twist.angular.x,
                      pm.twist.twist.angular.y,
                      pm.twist.twist.angular.z])
        if self.twist_in_world:
            v = world_to_body(v, q_wxyz)
            w = world_to_body(w, q_wxyz)
        return {'state': np.concatenate([v, w, q_wxyz, pos]),
                'time': pm.header.stamp.to_sec(), 'received_monotonic': received}


def _history_tensor(history, context_len, history_dim):
    """历史行 [s13+a5] 打包为 [context_len, history_dim]，最新行在最后，
    不足部分零填充（与训练侧 demo_policy 一致）。"""
    h = np.zeros((context_len, history_dim), np.float32)
    if history:
        h[-len(history):] = history
    return h


def run_policy_node(model_dir, spec_path, ref_family='',
                    episode_len_s=60.0):
    """Deployment loop: waits for the thruster bridge ready signal and the
    MAVROS parameter-list download, then runs odom state + reference ->
    obs[21] + history -> a[5] wrench -> thruster PWM on /dof_cmd at the
    control rate. Reference (green) and actual (orange) trajectories are
    published for RViz.

    参考来源（~ref_source）：'internal' = 本节点随机生成轨迹（固定时长
    episode）；'external' = 订阅 ~ref_topic 上由外部节点（如 vuram_planning
    goal_planner）发布的参考 Odometry；'collect' = 水平 figure8 连续实机
    采集，每圈导出 H5 并采用已准备好的最新策略。
    """
    spec = load_spec(spec_path)
    odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
    twist_in_world = rospy.get_param('~twist_in_world', False)
    # Collection is dispatched before the legacy single-snapshot loader.
    ref_source = rospy.get_param('~ref_source', 'internal')
    if ref_source == 'collect':
        from .collection_node import run_collection_node
        run_collection_node(model_dir, spec, episode_len_s)
        return
    ref_topic = rospy.get_param('~ref_topic', '/planning/reference_odom')
    if ref_source not in ('internal', 'external'):
        rospy.logfatal('bad ~ref_source %r (expected internal|external|collect)',
                       ref_source)
        return
    model_dir = model_dir or os.path.join(os.path.dirname(__file__), 'models', 'policy_v000012')
    # ref_seed < 0: a fresh random trajectory on every run
    ref_seed = rospy.get_param('~ref_seed', -1)
    if ref_seed < 0:
        ref_seed = int(np.random.SeedSequence().entropy % (2 ** 31))
    # reference depth band (world ENU, m); defaults = MuJoCo training band,
    # override for shallow-water / SITL operation (e.g. -5 / 0)
    ref_z_min = float(rospy.get_param('~ref_z_min', -52.5))
    ref_z_max = float(rospy.get_param('~ref_z_max', -47.5))

    inference = PolicyInference(model_dir, device=rospy.get_param('~device', 'cpu'))
    if not inference.loaded:
        rospy.logfatal('no policy model in %s', model_dir)
        return
    inference.validate_spec(spec)

    # 等待推进器桥接节点（set_6dof_cmd）完成飞控参数配置并发布 latch 就绪
    # 信号后再开始生成参考轨迹与下发控制；SITL 首次参数表下载可达数十秒。
    # 独立运行本节点（无桥接）时置 ~wait_ready=false。
    wait_ready = rospy.get_param('~wait_ready', True)
    ready_topic = rospy.get_param('~ready_topic', '/dof_cmd/ready')
    if wait_ready:
        ready = {'flag': False}
        rospy.Subscriber(ready_topic, Bool,
                         lambda m: ready.update(flag=m.data))
        while not ready['flag'] and not rospy.is_shutdown():
            rospy.logwarn_throttle(5, 'thruster bridge not ready (%s); '
                                   'waiting ...', ready_topic)
            rospy.sleep(0.2)
        if not ready['flag']:
            rospy.loginfo('shutdown while waiting for thruster bridge ready '
                          '(%s); exiting', ready_topic)
            return
        rospy.loginfo('thruster bridge ready (%s)', ready_topic)

    # 等待 MAVROS 参数表下载完成（PR: parameters list received）后再开始生成
    # 参考轨迹：实测参数下载期间的参数洪流会挤占 RC override 转发链路，
    # 造成 ~30 s 的控制中断/乱飘。/mavros/param/pull 服务会阻塞至参数表
    # 下载完成（已下载时立即返回）。独立运行本节点时置 ~wait_param_list=false。
    wait_param_list = rospy.get_param('~wait_param_list', True)
    param_pull_service = rospy.get_param('~param_pull_service',
                                         '/mavros/param/pull')
    if wait_param_list:
        try:
            rospy.wait_for_service(param_pull_service, timeout=30.0)
            pull = rospy.ServiceProxy(param_pull_service, ParamPull,
                                      persistent=True)
            resp = pull(force_pull=False)
            if resp.success:
                rospy.loginfo('MAVROS parameter list received (%d params)',
                              resp.param_received)
            else:
                rospy.logwarn('MAVROS param pull failed; continuing without '
                              'parameter-list confirmation')
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn('MAVROS param pull unavailable (%s); continuing', e)

    thrusters = ThrusterInterface(
        rospy.get_param('~cmd_topic', '/dof_cmd'))
    rospy.on_shutdown(thrusters.shutdown)
    pose = PoseState(odom_topic, twist_in_world=twist_in_world)
    if ref_source == 'internal':
        work_env = {'x': [-8.0, 8.0], 'y': [-8.0, 8.0],
                    'z': [ref_z_min, ref_z_max]}
        ref_gen = ReferenceGenerator(work_env)
        rng = np.random.default_rng(ref_seed)
    viz = TrajVisualizer(rospy.get_param('~viz', False),
                         frame_id=rospy.get_param('~viz_frame_id', 'map'))

    while pose.pose_msg is None and not rospy.is_shutdown():
        rospy.sleep(0.05)
    if pose.pose_msg is None:
        # shut down before the first odometry message arrived
        rospy.logwarn('no odometry received on %s; exiting', odom_topic)
        thrusters.set_zero()
        return

    if ref_source == 'external':
        _run_external_ref_loop(inference, thrusters, pose, viz, spec,
                               ref_topic)
        return

    # anchor the reference trajectory at the vehicle's actual start pose
    # (the deploy loop does not teleport the vehicle toward the reference)
    snap = pose.step()
    if snap is None:
        thrusters.set_zero()
        return
    start_pos = snap['state'][10:13]
    family = ref_family or None
    ref_params = ref_gen.sample_episode(rng, start_pos=start_pos,
                                        family=family)
    viz.start_episode(ref_params, ref_gen, episode_len_s)

    dt = spec['dt']
    n_steps = int(round(episode_len_s / dt))
    a_prev = np.zeros(spec['a_dim'])                   # 5
    context_len, history_dim = inference.history_shape
    history = []
    t0 = None
    pos_errors = []
    step_i = 0
    rospy.loginfo('deploying policy v%s (ref_source=internal, ref_family=%s, '
                  'ref_seed=%d, episode_len_s=%.1f)',
                  inference.version, family, ref_seed, episode_len_s)

    rate = rospy.Rate(spec['control_rate'])
    while not rospy.is_shutdown() and step_i < n_steps:
        snap = pose.step()
        if snap is None:
            rate.sleep()
            continue
        if t0 is None:
            t0 = snap['time']
        t = snap['time'] - t0
        s = snap['state']
        pos, quat, vel, w_z = ref_gen.get_ref(ref_params, t)
        ref11 = build_ref11(pos, quat, vel, w_z)
        obs = inference.observe(s, ref11, a_prev, spec['obs_layout'])
        h = _history_tensor(history, context_len, history_dim)
        a = inference.step(obs, h, len(history))
        a = np.clip(a, -1.0, 1.0)
        thrusters.set_wrench_action(a)
        history.append(np.concatenate((s, a)).astype(np.float32))
        history = history[-context_len:]
        a_prev = a
        pos_errors.append(float(np.linalg.norm(ref11[0:3] - s[10:13])))
        if step_i % 10 == 0:
            viz.add_actual_point(s[10:13])
        step_i += 1
        rate.sleep()

    thrusters.set_zero()
    mean_err = float(np.mean(pos_errors)) if pos_errors else 0.0
    rospy.loginfo('deploy done: steps=%d mean_pos_err=%.3f m',
                  step_i, mean_err)


def _run_external_ref_loop(inference, thrusters, pose, viz, spec, ref_topic):
    """外部参考模式：订阅参考 Odometry（pos + quat + vel + w_z），
    无限运行至关机。无参考消息时回中并等待（配合 1 s 看门狗语义）。
    episode_len_s/ref_family/ref_seed 在本模式被忽略。

    参考位置/姿态与线速度使用同一世界系；此处由模型的观测编码完成
    机体系转换。当前训练包含 setpoint，支持规划器到达终点后的静止参考。
    """
    ref_msg = {'msg': None}
    rospy.Subscriber(ref_topic, Odometry, lambda m: ref_msg.update(msg=m),
                     queue_size=5)

    a_prev = np.zeros(spec['a_dim'])
    context_len, history_dim = inference.history_shape
    history = []
    pos_errors = []
    step_i = 0
    rospy.loginfo('deploying policy v%s (ref_source=external, ref_topic=%s, '
                  'episode_len_s ignored)',
                  inference.version, ref_topic)

    rate = rospy.Rate(spec['control_rate'])
    while not rospy.is_shutdown():
        snap = pose.step()
        if snap is None:
            rate.sleep()
            continue
        m = ref_msg['msg']
        if m is None:
            rospy.logwarn_throttle(5, 'no reference on %s yet; holding '
                                   'neutral', ref_topic)
            thrusters.set_zero()
            rate.sleep()
            continue
        q = m.pose.pose.orientation
        pos = np.array([m.pose.pose.position.x,
                        m.pose.pose.position.y,
                        m.pose.pose.position.z])
        quat = np.array([q.w, q.x, q.y, q.z])   # xyzw -> wxyz
        vel = np.array([m.twist.twist.linear.x,
                        m.twist.twist.linear.y,
                        m.twist.twist.linear.z])
        ref11 = build_ref11(pos, quat, vel,
                            float(m.twist.twist.angular.z))
        s = snap['state']
        obs = inference.observe(s, ref11, a_prev, spec['obs_layout'])
        h = _history_tensor(history, context_len, history_dim)
        a = inference.step(obs, h, len(history))
        a = np.clip(a, -1.0, 1.0)
        thrusters.set_wrench_action(a)
        history.append(np.concatenate((s, a)).astype(np.float32))
        history = history[-context_len:]
        a_prev = a
        pos_errors.append(float(np.linalg.norm(ref11[0:3] - s[10:13])))
        if step_i % 10 == 0:
            viz.add_actual_point(s[10:13])
        step_i += 1
        rate.sleep()

    thrusters.set_zero()
    mean_err = float(np.mean(pos_errors)) if pos_errors else 0.0
    rospy.loginfo('external ref loop done: steps=%d mean_pos_err=%.3f m',
                  step_i, mean_err)
