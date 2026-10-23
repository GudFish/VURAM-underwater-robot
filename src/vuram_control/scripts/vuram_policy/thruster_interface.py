"""Thruster command interface: 两种动作契约 -> 6 通道 PWM -> /dof_cmd.

ArduSub 直控链路（VECTORED 帧，AUV_ctrl_6dof / policy_deploy.launch）：
发布 6 通道 PWM µs (1100..1900, 1500 neutral) 到 /dof_cmd；set_6dof_cmd
节点转为 RC_CHANNELS_OVERRIDE；飞控 MANUAL 模式按 6DOF 通道语义
（CH1=pitch、CH2=roll、CH3=throttle、CH4=yaw、CH5=forward、CH6=lateral）
输入 VECTORED 混控器，由飞控完成推进器分配（norm_input=(pwm-1500)/400，
acro_yaw_p=4.5 消除 yaw 通道的额外缩放，保留 arm 门控与 failsafe）。

动作契约（两种，按所用策略版本选择）：
1. set_action_norm(a[6])：逐推进器归一化指令（v75 及之前的策略口径），
   经 servo_map/servo_sign/tam_mix 映射为 PWM。
2. set_wrench_action(a[5])：VND v4 策略口径，a[5] = 归一化机体力/力矩
   [X,Y,Z,K,N]（无俯仰），**直接映射到 6DOF 通道**（无逐推分配）：
       CH1(pitch)   = 1500（策略无俯仰轴）
       CH2(roll)    = 1500 + 400·a[3]   (K，右舷下沉为正，两约定一致)
       CH3(throttle)= 1500 + 400·a[2]   (Z，向上为正)
       CH4(yaw)     = 1500 − 400·a[4]   (N，训练正=左转、ArduSub 正=右转，取反)
       CH5(forward) = 1500 + 400·a[0]   (X)
       CH6(lateral) = 1500 − 400·a[1]   (Y，训练正=左、ArduSub 正=右，取反)
   增益说明：策略按 action_scales·a 的物理力/力矩口径训练，而 VECTORED
   混控器每单位通道输入实现的力/力矩由混控与推进器决定（近似为训练
   口径的 0.5~1 倍），闭环策略可自适应；如需标定用 ~wrench_gain 统一
   缩放。yaw 通道要求飞控 ACRO_YAW_P=4.5（set_6dof_cmd 已配）。

安全：arm 门控与 failsafe 保留（未解锁无输出、override 过期自动
disarm）；此外本节点看门狗超时回中 1500，set_6dof_cmd 提供
~enable 急停服务。
"""
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray

DEADBAND = 0.08          # MuJoCo thruster_model deadband
PWM_MIN, PWM_NEUTRAL, PWM_MAX = 1100.0, 1500.0, 1900.0
PWM_SPAN = (PWM_MAX - PWM_MIN) / 2.0   # 400 us full-scale excursion

# a[0..4] = [X, Y, Z, K, N] -> 1 起始 RC 通道号（6DOF 通道语义）
WRENCH_DOF_CHANNELS = (5, 6, 3, 2, 4)     # forward, lateral, throttle, roll, yaw
# 每轴符号：+1 同向；−1 取反（Y：训练正=左 vs ArduSub 正=右；N：左转 vs 右转）
WRENCH_DOF_SIGNS = (1.0, -1.0, 1.0, 1.0, -1.0)

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


def wrench_to_pwm(a, gain=1.0):
    """a[5] 归一化力/力矩 [X,Y,Z,K,N] -> 6DOF 通道 PWM µs（VECTORED 帧）。

    直接映射（无逐推分配）：a 的每个轴写到对应的 6DOF 通道，
    符号按训练口径与 ArduSub 通道语义换算（Y/N 取反），俯仰通道保持
    中立。gain 为整体标定缩放。
    """
    a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
    pwm = np.full(6, PWM_NEUTRAL)
    for i, (ch, sign) in enumerate(zip(WRENCH_DOF_CHANNELS,
                                       WRENCH_DOF_SIGNS)):
        pwm[ch - 1] = PWM_NEUTRAL + PWM_SPAN * gain * sign * a[i]
    return np.clip(pwm, PWM_MIN, PWM_MAX)


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
        # VND v4 力/力矩契约：直接 6DOF 通道映射的整体标定增益
        self.wrench_gain = float(rospy.get_param('~wrench_gain', 1.0))
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

    def set_wrench_action(self, a):
        """VND v4：a[5] 归一化力/力矩 [X,Y,Z,K,N] -> 6DOF 通道 PWM -> 发布。

        直接映射到 6DOF 通道（X→forward、Y→lateral(取反)、Z→throttle、
        K→roll、N→yaw(取反)，俯仰中立），由飞控 VECTORED 混控器完成
        推进器分配。整体标定用 ~wrench_gain。
        """
        pwm = wrench_to_pwm(a, self.wrench_gain)
        self._last_cmd = pwm
        self._last_cmd_time = rospy.Time.now()
        self._publish(pwm)

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
