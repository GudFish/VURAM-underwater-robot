"""TorchScript policy loader + 20 Hz deployment loop (ArduSub/MAVROS).

The exported module has signature forward(obs[22]) -> a[6] (the stochastic
deployment path, tanh(mean + std*eps)); obs normalization is folded into
the module, so raw physical units go in. Deploy only: no episodes, no
domain randomization, no HDF5, no termination pruning, no CSV logging.
"""
import json
import os

import numpy as np
import rospy
from mavros_msgs.srv import ParamPull
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

from .reference_generator import ReferenceGenerator
from .spec import build_obs, load_spec
from .thruster_interface import ThrusterInterface
from .traj_visualizer import TrajVisualizer
from .utils import quat_to_rpy, world_to_body


class PolicyInference(object):
    def __init__(self, model_dir):
        # torch is imported lazily: startup stays fast and the missing-model
        # error path never needs it
        import torch
        self.torch = torch
        torch.set_num_threads(2)   # don't starve the SITL/MAVROS stack
        # ~device: 'cpu'（默认）| 'cuda'。实测小模型 GPU 反而更慢
        # （kernel 启动 + 回传同步开销），故默认 CPU；显式指定 cuda
        # 且 CUDA 不可用时回退 CPU 并告警。
        device_arg = rospy.get_param('~device', 'cpu')
        if device_arg.startswith('cuda') and not torch.cuda.is_available():
            rospy.logwarn('device=%s requested but CUDA unavailable; '
                          'falling back to cpu', device_arg)
            device_arg = 'cpu'
        self.device = torch.device(device_arg)
        self.model_dir = model_dir
        self.model = None
        self.version = None
        self._load()

    @property
    def loaded(self):
        return self.model is not None

    def _load(self):
        pt_path = os.path.join(self.model_dir, 'policy.pt')
        if not os.path.exists(pt_path):
            rospy.logerr('no policy.pt in %s', self.model_dir)
            return
        self.model = self.torch.jit.load(pt_path)
        self.model.to(self.device)
        self.model.eval()
        spec_json = os.path.join(self.model_dir, 'spec.json')
        if os.path.exists(spec_json):
            with open(spec_json) as f:
                self.version = json.load(f).get('policy_version')
        rospy.loginfo('loaded policy version %s from %s (device: %s)',
                      self.version, self.model_dir, self.device)

    def step(self, obs):
        """obs[22] -> a[6] (stochastic deployment path)."""
        t = self.torch
        with t.no_grad():
            o = t.from_numpy(np.asarray(obs, dtype=np.float32))[None].to(self.device)
            a = self.model(o)
        return a.cpu().numpy()[0].astype(np.float64)


class PoseState(object):
    """MAVROS-odometry state snapshot: s[12] = [v_body6, rpy3, pos3].

    /mavros/local_position/odom reports twist in the BODY frame (ENU axes:
    x forward / y left / z up), so with twist_in_world=False (the ArduSub
    default, see config/obs_act.yaml) no world->body rotation is applied.
    Pose is world ENU, matching the MuJoCo training convention.
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
        rpy = quat_to_rpy([q.x, q.y, q.z, q.w])
        v = np.array([pm.twist.twist.linear.x,
                      pm.twist.twist.linear.y,
                      pm.twist.twist.linear.z])
        w = np.array([pm.twist.twist.angular.x,
                      pm.twist.twist.angular.y,
                      pm.twist.twist.angular.z])
        if self.twist_in_world:
            v = world_to_body(v, rpy)
            w = world_to_body(w, rpy)
        return {'state': np.concatenate([v, w, rpy, pos]),
                'time': pm.header.stamp.to_sec()}


def _run_external_ref_loop(inference, thrusters, pose, viz, spec, ref_topic):
    """外部参考模式：订阅参考 Odometry（pos + quat(yaw) + vel + w_z），
    无限运行至关机。无参考消息时回中并等待（配合 1 s 看门狗语义）。
    episode_len_s/ref_family/ref_seed 在本模式被忽略。
    """
    ref_msg = {'msg': None}
    rospy.Subscriber(ref_topic, Odometry, lambda m: ref_msg.update(msg=m),
                     queue_size=5)

    a_prev = np.zeros(spec['a_dim'])
    pos_errors = []
    step_i = 0
    rospy.loginfo('deploying policy version %s (ref_source=external, '
                  'ref_topic=%s, episode_len_s ignored)',
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
        rpy = quat_to_rpy([q.x, q.y, q.z, q.w])
        vel = np.array([m.twist.twist.linear.x,
                        m.twist.twist.linear.y,
                        m.twist.twist.linear.z])
        ref10 = np.concatenate([pos, rpy, vel,
                                [float(m.twist.twist.angular.z)]])
        obs = build_obs(snap['state'], ref10, a_prev, spec['obs_layout'])
        a = np.clip(inference.step(obs), -1.0, 1.0)
        thrusters.set_action_norm(a)
        a_prev = a
        pos_errors.append(float(np.linalg.norm(ref10[0:3] -
                                               snap['state'][9:12])))
        if step_i % 10 == 0:
            viz.add_actual_point(snap['state'][9:12])
        step_i += 1
        rate.sleep()

    thrusters.set_zero()
    mean_err = float(np.mean(pos_errors)) if pos_errors else 0.0
    rospy.loginfo('external ref loop done: steps=%d mean_pos_err=%.3f m',
                  step_i, mean_err)


def run_policy_node(model_dir, spec_path, ref_family='',
                    episode_len_s=60.0):
    """Deployment loop: waits for the thruster bridge ready signal and the
    MAVROS parameter-list download, then runs odom state + reference ->
    obs[22] -> a[6] -> thruster PWM on /dof_cmd at the control rate.
    Reference (green) and actual (orange) trajectories are published for
    RViz.

    参考来源（~ref_source）：'internal' = 本节点随机生成轨迹（固定时长
    episode）；'external' = 订阅 ~ref_topic 上由外部节点（如 vuram_planning
    goal_planner）发布的参考 Odometry，无限运行至关机。
    """
    spec = load_spec(spec_path)
    odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
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
        rospy.logfatal('no policy.pt in %s', model_dir)
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
        rospy.get_param('~cmd_topic', '/thruster_cmd'))
    pose = PoseState(odom_topic, twist_in_world=spec['twist_in_world'])
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
    start_pos = snap['state'][9:12]
    family = ref_family or None
    ref_params = ref_gen.sample_episode(rng, start_pos=start_pos,
                                        family=family)
    viz.start_episode(ref_params, ref_gen, episode_len_s)

    dt = spec['dt']
    n_steps = int(round(episode_len_s / dt))
    a_prev = np.zeros(spec['a_dim'])
    t0 = None
    pos_errors = []
    step_i = 0
    rospy.loginfo('deploying policy version %s (ref_family=%s, ref_seed=%d, '
                  'episode_len_s=%.1f)',
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
        pos, rpy, vel, w_z = ref_gen.get_ref(ref_params, t)
        ref10 = np.concatenate([pos, rpy, vel, [w_z]])
        obs = build_obs(s, ref10, a_prev, spec['obs_layout'])
        a = inference.step(obs)
        a = np.clip(a, -1.0, 1.0)
        thrusters.set_action_norm(a)
        a_prev = a
        pos_errors.append(float(np.linalg.norm(ref10[0:3] - s[9:12])))
        if step_i % 10 == 0:
            viz.add_actual_point(s[9:12])
        step_i += 1
        rate.sleep()

    thrusters.set_zero()
    mean_err = float(np.mean(pos_errors)) if pos_errors else 0.0
    rospy.loginfo('deploy done: steps=%d mean_pos_err=%.3f m',
                  step_i, mean_err)
