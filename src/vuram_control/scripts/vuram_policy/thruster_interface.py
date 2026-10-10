"""[-1,1]^6 normalized actions -> thruster PWM -> /dof_cmd（CUSTOM 1:1 帧）.

ArduSub 逐推进器直控链路（AUV_ctrl_custom / policy_deploy.launch）：
a in [-1,1] per thruster -> PWM us (1100..1900, 1500 neutral), published as
a 6-channel Float64MultiArray on /dof_cmd. The set_6dof_cmd node forwards
each channel as RC_CHANNELS_OVERRIDE; the FCU CUSTOM frame mixer maps
CH1..6 1:1 to MOTOR1..6（飞控侧 norm_input=(pwm-1500)/400 与 SITL 同口径，
acro_yaw_p=4.5 消除 yaw 缩放后全通道 1:1，保留 arm 门控与 failsafe）。

通道几何（重要）：策略输出 a[0..5] 对应 MuJoCo 训练模型的推进器 T0..T5
（~/mujoco_auv_env/envs/thrusters.py，与 UUV 端 actuators.xacro 一致）。
ArduSub 原版 VECTORED 几何（SIM_Submarine.cpp）与孪生模型是镜像（45°
矢量方向互为镜像、垂直推进器正 PWM 方向相反），恒等映射必然导致策略
"乱飞"。现 SITL 物理表已改为孪生几何（SIM_Submarine.cpp vectored_thrusters，
MOTOR1..6 = 孪生 T0..T5，与实车 Pixhawk MAIN OUT 1~6 接线一致），
恒等映射 a[i] -> CH(i+1) 在 SITL 与实车上均与训练一致。

映射模式（policy_deploy.launch 通过 ~tam_mix 选择）：
1. ~tam_mix=false（默认）：恒等映射 a[i] -> CH(i+1)，全部同向（孪生
   几何下正 PWM 与训练正方向一致，垂直推进器正 PWM = 向上）。接线
   不一致时用 ~servo_map / ~servo_sign 标定。
2. ~tam_mix=true（历史兼容）：若 SITL 物理表仍是原版镜像几何，用最小
   范数混合矩阵 M = pinv(TAM_sitl) @ TAM_twin 把 a 映射为 6 通道指令，
   在可达子空间上精确复现孪生力/力矩（重建误差 ~1e-15；实测
   a[0]=1 → 前+左、a[1]=1 → 前+右、a[4]=1 → 向上）。

The policy was trained against the MuJoCo thruster chain
(thruster_model.py): a -> deadband 0.08 -> T200 polynomial -> omega. The
same deadband is kept here so tiny commands map to neutral PWM; the
nonlinear T200 curve is NOT reproduced (ArduSub ESCs map PWM to RPM
directly, and SITL uses a quadratic out*|out| curve).

安全：CUSTOM 帧保留 arm 门控与 failsafe（未解锁无输出、override 过期
自动 disarm）；此外本节点看门狗超时回中 1500，set_6dof_cmd 提供
~enable 急停服务。
"""
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray

DEADBAND = 0.08          # MuJoCo thruster_model deadband
PWM_MIN, PWM_NEUTRAL, PWM_MAX = 1100.0, 1500.0, 1900.0
PWM_SPAN = (PWM_MAX - PWM_MIN) / 2.0   # 400 us full-scale excursion

# 恒等映射：a[i] -> servo(i+1)，全部同向（SITL 物理表已改为孪生几何：
# 正 PWM 与训练正方向一致，垂直推进器正 PWM = 向上）
IDENTITY_SERVO = (1, 2, 3, 4, 5, 6)
IDENTITY_SIGN = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

# 原版 ArduSub VECTORED 表下的最小范数混合映射（行=servo1..6, 列=a0..a5）
# M = pinv(TAM_sitl) @ TAM_twin（SITL 列已做 NED->ENU 转换：y/z 取反）；
# 孪生与 SITL 的可达力/力矩子空间相同（俯仰行为零），故 M 在可达子空间上
# 精确还原孪生力/力矩（重建误差 ~1e-15）。实测验证：a[0]=1 的混合指令
# 产生前+左推力与左转偏航（孪生 T0 口径）。
TAM_MIX = np.array([
    [-0.711, 0.211, 0.211, 0.289, 0.0, 0.0],
    [0.211, -0.711, 0.289, 0.211, 0.0, 0.0],
    [0.211, 0.289, -0.711, 0.211, 0.0, 0.0],
    [0.289, 0.211, 0.211, -0.711, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, -0.008, -0.992],
    [0.0, 0.0, 0.0, 0.0, -0.992, -0.008],
])


def action_to_pwm(a, servo_map=IDENTITY_SERVO, servo_sign=IDENTITY_SIGN,
                  tam_mix=None):
    """Normalized action a in [-1,1]^6 -> servo-ordered PWM us (1500 neutral).

    输出按 servo 通道号排序（index 0 = servo 1），可直接作为
    /dof_cmd 的 data 发布给 set_6dof_cmd。

    tam_mix 为 None 时用 servo_map/servo_sign 排列映射；否则用混合矩阵
    （原版 SITL 兼容模式），servo_map/servo_sign 被忽略。
    """
    a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
    a = np.where(np.abs(a) < DEADBAND, 0.0, a)
    if tam_mix is not None:
        out = np.asarray(tam_mix, dtype=np.float64) @ a
        return np.clip(PWM_NEUTRAL + PWM_SPAN * out, PWM_MIN, PWM_MAX)
    pwm = np.full(max(servo_map), PWM_NEUTRAL)
    for i, (servo, sign) in enumerate(zip(servo_map, servo_sign)):
        pwm[servo - 1] = PWM_NEUTRAL + PWM_SPAN * sign * a[i]
    return np.clip(pwm, PWM_MIN, PWM_MAX)


class ThrusterInterface(object):
    def __init__(self, cmd_topic='/thruster_cmd', n_thrusters=6,
                 watchdog_timeout=1.0):
        self.n_thrusters = n_thrusters
        self.pub = rospy.Publisher(cmd_topic, Float64MultiArray,
                                   queue_size=10)
        # 通道几何：默认恒等映射（SITL 物理表与实车接线均为孪生 T0..T5
        # 顺序）；原版镜像几何的 SITL 用 ~tam_mix:=true；实车接线差异用
        # ~servo_map/~servo_sign 覆盖。
        self.servo_map = tuple(int(v) for v in rospy.get_param(
            '~servo_map', IDENTITY_SERVO))
        self.servo_sign = tuple(float(v) for v in rospy.get_param(
            '~servo_sign', IDENTITY_SIGN))
        self.tam_mix = rospy.get_param('~tam_mix', False)
        self._watchdog_timeout = watchdog_timeout
        self._last_cmd = np.zeros(n_thrusters)
        self._last_cmd_time = rospy.Time.now()
        # Watchdog: if no command arrives for `watchdog_timeout` (node
        # stuck), return the thrusters to neutral instead of leaving them on.
        self._watchdog = rospy.Timer(rospy.Duration(0.2), self._watchdog_cb)

    @property
    def last_cmd(self):
        """Last published thruster command (servo-ordered PWM us)."""
        return self._last_cmd.copy()

    def set_action_norm(self, a):
        """Publish a normalized action a in [-1,1]^6 as thruster PWM."""
        cmd = action_to_pwm(a, self.servo_map, self.servo_sign,
                            TAM_MIX if self.tam_mix else None)
        self._last_cmd = cmd
        self._last_cmd_time = rospy.Time.now()
        self._publish(cmd)

    def set_zero(self):
        self._last_cmd = np.full(self.n_thrusters, PWM_NEUTRAL)
        self._last_cmd_time = rospy.Time.now()
        self._publish(self._last_cmd)

    def _publish(self, pwm):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in pwm]
        self.pub.publish(msg)

    def _watchdog_cb(self, _event):
        if (rospy.Time.now() - self._last_cmd_time) > \
                rospy.Duration(self._watchdog_timeout):
            self._publish(np.full(self.n_thrusters, PWM_NEUTRAL))

    def shutdown(self):
        self.set_zero()
        self._watchdog.shutdown()
