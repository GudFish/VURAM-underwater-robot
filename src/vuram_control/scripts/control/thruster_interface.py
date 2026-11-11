"""Thruster command interface: body wrench[5] -> 6 通道 PWM -> /dof_cmd.

ArduSub 直控链路（VECTORED 帧，AUV_ctrl_6dof / policy_deploy.launch）：
发布 6 通道 PWM µs (1100..1900, 1500 neutral) 到 /dof_cmd；set_6dof_cmd
节点转为 RC_CHANNELS_OVERRIDE；飞控 MANUAL 模式按 6DOF 通道语义
（CH1=pitch、CH2=roll、CH3=throttle、CH4=yaw、CH5=forward、CH6=lateral）
输入 VECTORED 混控器，由飞控完成推进器分配（norm_input=(pwm-1500)/400，
acro_yaw_p=4.5 消除 yaw 通道的额外缩放，保留 arm 门控与 failsafe）。

动作契约：set_wrench_action(a[5])，a[5] = 归一化机体力/力矩
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

PWM_MIN, PWM_NEUTRAL, PWM_MAX = 1100.0, 1500.0, 1900.0
PWM_SPAN = (PWM_MAX - PWM_MIN) / 2.0   # 400 us full-scale excursion

# a[0..4] = [X, Y, Z, K, N] -> 1 起始 RC 通道号（6DOF 通道语义）
WRENCH_DOF_CHANNELS = (5, 6, 3, 2, 4)     # forward, lateral, throttle, roll, yaw
# 每轴符号：+1 同向；−1 取反（Y：训练正=左 vs ArduSub 正=右；N：左转 vs 右转）
WRENCH_DOF_SIGNS = (1.0, -1.0, 1.0, 1.0, -1.0)


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


class ThrusterInterface(object):
    def __init__(self, cmd_topic='/dof_cmd', n_thrusters=6,
                 watchdog_timeout=1.0):
        self.n_thrusters = n_thrusters
        self.pub = rospy.Publisher(cmd_topic, Float64MultiArray,
                                   queue_size=10)
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
