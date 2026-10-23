"""TorchScript VND policy loader + 20 Hz deployment loop (ArduSub/MAVROS).

The exported module (deploy.pt) has signature
forward(obs[21], history[20,18], valid) -> a[5] — a deterministic FiLM
policy conditioned on a GRU history encoder (latent z = 0 until `valid`
reaches the encoder context length). Obs and encoder normalization are
folded into the module; raw physical units go in. Deploy only: no
episodes, no domain randomization, no HDF5, no termination pruning, no
CSV logging.

动作契约（VND v4）：a[5] = 归一化机体力/力矩 [X,Y,Z,K,N]（无俯仰）。
经 ThrusterInterface.set_wrench_action 直接映射到 6DOF 通道 PWM
（X→forward、Y→lateral、Z→throttle、K→roll、N→yaw，符号按训练口径与
ArduSub 通道语义换算），发布到 /dof_cmd（复用 AUV_ctrl_6dof 的
set_6dof_cmd 链路，飞控 VECTORED 混控器完成推进器分配）。
"""
import json
import os

import numpy as np
import rospy
from mavros_msgs.srv import ParamPull
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

from .reference_generator import ReferenceGenerator
from .spec import build_obs, build_ref11, load_spec
from .thruster_interface import ThrusterInterface
from .traj_visualizer import TrajVisualizer
from .utils import world_to_body


class PolicyInference(object):
    def __init__(self, model_dir):
        # torch is imported lazily: startup stays fast and the missing-model
        # error path never needs it
        import torch
        self.torch = torch
        torch.set_num_threads(2)   # don't starve the SITL/MAVROS stack
        self.model_dir = model_dir
        self.model = None
        self.version = None
        self.history_shape = (20, 18)
        self._load()

    @property
    def loaded(self):
        return self.model is not None

    def _load(self):
        pt_path = None
        for name in ('deploy.pt', 'policy.pt'):
            candidate = os.path.join(self.model_dir, name)
            if os.path.exists(candidate):
                pt_path = candidate
                break
        if pt_path is None:
            rospy.logerr('no deploy.pt/policy.pt in %s', self.model_dir)
            return
        self.model = self.torch.jit.load(pt_path)
        self.model.eval()
        spec_json = os.path.join(self.model_dir, 'spec.json')
        if os.path.exists(spec_json):
            with open(spec_json) as f:
                manifest = json.load(f)
            self.version = manifest.get('version')
            self.history_shape = tuple(manifest.get('history_shape', [20, 18]))
        rospy.loginfo('loaded policy v%s from %s', self.version, self.model_dir)

    def step(self, obs, history, valid):
        """obs[21], history[20,18] (raw), valid int -> a[5]."""
        t = self.torch
        with t.no_grad():
            o = t.from_numpy(np.asarray(obs, dtype=np.float32))[None]
            h = t.from_numpy(np.asarray(history, dtype=np.float32))[None]
            v = t.tensor([int(valid)], dtype=t.int64)
            a = self.model(o, h, v)
        return a.numpy()[0].astype(np.float64)


class PoseState(object):
    """MAVROS odometry snapshot: s[13] = [v_body6, quat_wxyz4, pos3].

    /mavros/local_position/odom 与 DVL odom 的 twist 均为机体系（ENU 轴，
    x 前 / y 左 / z 上），默认直接使用；~twist_in_world=true 时按四元数
    转到机体系（Gazebo pose_gt 场景）。
    """

    def __init__(self, odom_topic, twist_in_world=False):
        self.twist_in_world = twist_in_world
        self.pose_msg = None
        self.sub = rospy.Subscriber(odom_topic, Odometry, self._cb,
                                    queue_size=5)

    def _cb(self, msg):
        self.pose_msg = msg

    def step(self):
        """Snapshot at the control tick; None until the first pose arrives."""
        if self.pose_msg is None:
            return None
        pm = self.pose_msg
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
                'time': pm.header.stamp.to_sec()}


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
    goal_planner）发布的参考 Odometry，无限运行至关机。
    """
    spec = load_spec(spec_path)
    odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
    twist_in_world = rospy.get_param('~twist_in_world', False)
    # 参考来源：internal | external（非法值直接退出）
    ref_source = rospy.get_param('~ref_source', 'internal')
    ref_topic = rospy.get_param('~ref_topic', '/planning/reference_odom')
    if ref_source not in ('internal', 'external'):
        rospy.logfatal('bad ~ref_source %r (expected internal|external)',
                       ref_source)
        return
    # ref_seed < 0: a fresh random trajectory on every run
    ref_seed = rospy.get_param('~ref_seed', -1)
    if ref_seed < 0:
        ref_seed = int(np.random.SeedSequence().entropy % (2 ** 31))
    # reference depth band (world ENU, m); defaults = MuJoCo training band,
    # override for shallow-water / SITL operation (e.g. -5 / 0)
    ref_z_min = float(rospy.get_param('~ref_z_min', -52.5))
    ref_z_max = float(rospy.get_param('~ref_z_max', -47.5))

    inference = PolicyInference(model_dir)
    if not inference.loaded:
        rospy.logfatal('no policy model in %s', model_dir)
        return

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
        obs = build_obs(s, ref11, a_prev, spec['obs_layout'])
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

    注意：训练策略的参考轨迹永远在运动，"静止参考"不在训练分布内——
    在重新训练出定点保持能力（训练侧加入 hold/hover 参考族）之前，
    策略对静止参考的表现会漂移。参考本身由规划器锚定保证静止不动。
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
        obs = build_obs(s, ref11, a_prev, spec['obs_layout'])
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
