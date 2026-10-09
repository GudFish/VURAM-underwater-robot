#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
set_thruster_cmd.py

推进器直控节点：通过 MAV_CMD_DO_SET_SERVO (183) 直接写入每个推进器输出
通道的 PWM（µs），完全绕过 ArduSub 的混控器、飞行模式与解锁（arm）门控。
与 set_6dof_cmd.py（6 自由度 RC override，保留 arm 门控）互为替代方案，
二者勿同时运行。

原理
----
前提：飞控上这 6 个通道的 SERVOx_FUNCTION 必须为 0（禁用），否则 ArduSub
电机混控器（AP_Motors6DOF）每个控制周期都会覆盖写入值。
节点启动时默认自动通过 /mavros/param/set 完成该设置（~setup_fc_params 可
关闭）；参数持久保存在飞控中。恢复出厂控制：把 SERVO1~SERVO6_FUNCTION
改回 MOTOR1~MOTOR6（33~38）。

通道语义：CH1~CH6 对应 SERVO1~SERVO6 输出，即飞控 MOTOR1~MOTOR6 的物理
接线顺序（与 QGC 里 Motor1~6 编号一致），PWM 1100~1900，1500 停止。
输入不经过任何混控，上层控制器需自行完成推进器布置解算。

安全（务必了解）
----
本模式无飞控 arm 门控：只要节点发布非中立值，推进器就会转动，与飞控
是否解锁无关；飞控侧 failsafe（无操控输入自动 disarm 等）也无法停机。
安全仅依赖：脚本看门狗（~watchdog_timeout）、~enable 急停服务、
节点退出时输出中立值，以及推进器供电回路的物理急停开关（强烈建议）。

接口
----
订阅 ~cmd_topic        std_msgs/Float64MultiArray   6 个推进器 PWM (us, 1100~1900)
服务 ~enable           std_srvs/SetBool             使能/急停（禁用时持续发中立值）
发布 ~cmd_out_topic    std_msgs/Float64MultiArray   实际输出 PWM 回显（调试/记录）

用法
----
  rosrun sub_planning set_thruster_cmd.py
  # 测试 3 号推进器小幅正转
  rostopic pub -r 10 /thruster_cmd std_msgs/Float64MultiArray \
      "data: [1500.0, 1500.0, 1550.0, 1500.0, 1500.0, 1500.0]"

注意：若飞控走 115200 波特率串口，~send_rate 过高会挤占上行带宽
（6 条 COMMAND_LONG × 50 Hz ≈ 12 kB/s），可降至 30~40 Hz。
"""

import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import rospy
from mavros_msgs.msg import State, ParamValue
from mavros_msgs.srv import CommandLong, ParamGet, ParamSet
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, SetBoolResponse

# MAV_CMD_DO_SET_SERVO：直接设置伺服/电机输出通道 PWM（param1=通道号，param2=PWM）
MAV_CMD_DO_SET_SERVO = 183


class ThrusterCmdNode:
    """
    功能：订阅 6 个推进器的 PWM 指令（µs），以 DO_SET_SERVO 形式逐通道
          写入飞控输出，附带看门狗、急停服务与飞控参数管理。
    输入：~cmd_topic 上的 6 推进器 PWM 指令（µs，1500 停止），以及 MAVROS
          状态与参数/指令服务。
    输出：/mavros/cmd/command 上的 DO_SET_SERVO 指令流，~cmd_out_topic 回显。
    """

    def __init__(self):
        rospy.init_node('set_thruster_cmd', anonymous=True)

        # ---- ROS 参数 ----
        self.cmd_topic = rospy.get_param('~cmd_topic', '/thruster_cmd')
        self.cmd_out_topic = rospy.get_param('~cmd_out_topic', '/thruster_cmd/out')
        self.num_thrusters = int(rospy.get_param('~num_thrusters', 6))
        self.pwm_min = float(rospy.get_param('~pwm_min', 1100.0))
        self.pwm_max = float(rospy.get_param('~pwm_max', 1900.0))
        self.neutral_pwm = float(rospy.get_param('~neutral_pwm', 1500.0))
        self.send_rate = float(rospy.get_param('~send_rate', 50.0))
        self.watchdog_timeout = float(rospy.get_param('~watchdog_timeout', 1.0))
        # 飞控参数自动配置：SERVO1~SERVO{n}_FUNCTION=0（本方案必需，参数持久化，
        # 首次成功设置后可置 false 跳过）
        self.setup_fc_params = rospy.get_param('~setup_fc_params', True)
        # 等待 MAVROS 参数表下载完成的总超时（ArduSub 参数约上千个，下载需数秒到数十秒）
        self.param_wait_timeout = float(rospy.get_param('~param_wait_timeout', 180.0))

        # MAVROS 话题/服务名
        self.state_topic = rospy.get_param('~state_topic', '/mavros/state')
        self.cmd_service = rospy.get_param('~cmd_service', '/mavros/cmd/command')
        self.param_get_service = rospy.get_param('~param_get_service', '/mavros/param/get')
        self.param_set_service = rospy.get_param('~param_set_service', '/mavros/param/set')

        if not 1 <= self.num_thrusters <= 8:
            rospy.logfatal("num_thrusters=%d 超出 ArduSub 伺服通道范围 [1, 8]",
                           self.num_thrusters)
            sys.exit(1)

        # ---- 状态变量 ----
        self.current_state = State()
        self.enabled = True
        self.target_cmd = [self.neutral_pwm] * self.num_thrusters
        self.last_cmd_time = rospy.Time.now()
        self.cmd_received = False

        # ---- 订阅 / 发布 / 服务 ----
        self.state_sub = rospy.Subscriber(self.state_topic, State, self.state_cb)
        self.cmd_sub = rospy.Subscriber(self.cmd_topic, Float64MultiArray, self.cmd_cb)
        self.cmd_out_pub = rospy.Publisher(self.cmd_out_topic, Float64MultiArray, queue_size=1)
        self.enable_srv = rospy.Service('~enable', SetBool, self.enable_cb)

        # ---- 等待 MAVROS 连接与所需服务就绪 ----
        self._wait_for_mavros()

        self.cmd_srv = rospy.ServiceProxy(self.cmd_service, CommandLong, persistent=True)
        self.param_get_srv = rospy.ServiceProxy(self.param_get_service, ParamGet, persistent=True)
        self.param_set_srv = rospy.ServiceProxy(self.param_set_service, ParamSet, persistent=True)

        # ---- 启动前的飞控配置（禁用 SERVOx_FUNCTION，本方案必需） ----
        self._setup_fc_params()

        # ---- 控制定时器与退出钩子 ----
        period = 1.0 / max(self.send_rate, 1e-3)
        self.timer = rospy.Timer(rospy.Duration(period), self.control_loop)
        rospy.on_shutdown(self._publish_neutral)

        rospy.loginfo("Thruster direct-PWM node ready: %d thrusters at %.1f Hz, "
                      "PWM range [%.0f, %.0f], neutral %.0f",
                      self.num_thrusters, self.send_rate,
                      self.pwm_min, self.pwm_max, self.neutral_pwm)
        rospy.logwarn("DIRECT PWM MODE: no arm gating and no FC-side failsafe! "
                      "Safety relies on this node's watchdog, the ~enable service, "
                      "and a physical power cut-off.")

    # ------------------------------------------------------------------
    # 启动配置
    # ------------------------------------------------------------------

    def _wait_for_mavros(self):
        """
        功能：阻塞等待飞控连接与 MAVROS 参数/指令服务就绪。
        输入：无显式参数；依赖 current_state 与 MAVROS 服务状态。
        输出：无显式返回；连接与服务全部就绪后返回。
        """
        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown():
            if self.current_state.connected:
                try:
                    rospy.wait_for_service(self.cmd_service, timeout=1.0)
                    rospy.wait_for_service(self.param_get_service, timeout=1.0)
                    rospy.wait_for_service(self.param_set_service, timeout=1.0)
                    rospy.loginfo("MAVROS connected, services ready")
                    return
                except rospy.ROSException:
                    rospy.loginfo_throttle(5, "Waiting for MAVROS services...")
            else:
                rospy.loginfo_throttle(5, "Waiting for FC connection...")
            rate.sleep()

    def _param_get_int(self, param_id):
        """
        功能：读取整型参数值；参数不可见或读取失败返回 None（不打印，避免轮询刷屏）。
        输入：参数名 param_id。
        输出：整型值或 None。
        """
        try:
            resp = self.param_get_srv(param_id=param_id)
            if resp.success:
                # 旧版 mavros_msgs 的 ParamGet 响应只有 value (ParamValue)，
                # 整型值在 value.integer 中
                return int(resp.value.integer)
        except (rospy.ServiceException, AttributeError):
            pass
        return None

    def _set_param(self, param_id, value):
        """
        功能：调用 /mavros/param/set 设置整型参数，带少量重试。
        输入：参数名 param_id 与整型值 value。
        输出：设置成功返回 True，失败返回 False。
        """
        for _ in range(3):
            try:
                resp = self.param_set_srv(param_id=param_id,
                                          value=ParamValue(integer=value))
                if resp.success:
                    return True
            except rospy.ServiceException:
                pass
            rospy.sleep(1.0)
        return False

    def _wait_and_set_params(self, targets):
        """
        功能：等待 MAVROS 下载完飞控参数表（用 param/get 逐个探测），然后设置参数。
              MAVROS 启动后需先下载完整参数表（ArduSub 约上千个参数），下载完成前
              param/set 会被拒绝（Unknown parameter / receiving not complete）。
              若同时开着 QGC，QGC 可能独占参数流导致 MAVROS 一直收不完，
              需关闭 QGC 后重试。
        输入：targets 为 {参数名: 目标整型值} 字典。
        输出：全部设置成功返回 True，超时返回 False（节点继续运行但推进器不响应）。
        """
        ids = list(targets)
        rospy.loginfo("Waiting for MAVROS parameter list, then setting %s ...",
                      ", ".join("%s=%d" % (k, v) for k, v in targets.items()))
        start = rospy.Time.now()

        def _elapsed():
            return (rospy.Time.now() - start).to_sec()

        # 阶段一：只探测第一个参数（金丝雀），等参数表下载完成。
        # 避免对每个参数轮询触发 MAVROS 刷屏（Unknown parameter to get）
        canary = ids[0]
        while not rospy.is_shutdown() and _elapsed() < self.param_wait_timeout:
            if self._param_get_int(canary) is not None:
                break
            rospy.sleep(2.0)

        # 阶段二：逐个确认/设置所有目标参数
        done = set()
        shown_progress = -1
        while not rospy.is_shutdown() and _elapsed() < self.param_wait_timeout:
            for pid in ids:
                if pid in done:
                    continue
                current = self._param_get_int(pid)
                # 参数不可见 → 参数表仍在下载，等下一轮再探测
                if current is None:
                    continue
                # 参数已可见且值已正确 → 无需再设置
                if current == targets[pid]:
                    done.add(pid)
                    continue
                # 参数已可见但值不正确 → 设置
                if self._set_param(pid, targets[pid]):
                    done.add(pid)
                    rospy.loginfo("Set %s = %d", pid, targets[pid])
            if len(done) == len(ids):
                rospy.loginfo("All %d parameters ready", len(done))
                return True
            if len(done) != shown_progress:
                rospy.loginfo("Parameter setup progress: %d/%d", len(done), len(ids))
                shown_progress = len(done)
            rospy.sleep(1.0)
        rospy.logerr(
            "Failed to set %s within %.0f s. Close QGC (it can monopolize the "
            "parameter stream) and restart this node to retry.",
            ", ".join(ids), self.param_wait_timeout)
        return False

    def _setup_fc_params(self):
        """
        功能：将 SERVO1~SERVO{n}_FUNCTION 置 0（禁用），避免飞控混控器覆盖输出。
              若 ~setup_fc_params=false 则跳过（参数已持久化在飞控中时使用）。
        输入：无显式参数；依赖 setup_fc_params 开关与 num_thrusters。
        输出：无显式返回；打印设置结果。
        """
        if not self.setup_fc_params:
            rospy.loginfo("FC parameter auto-setup skipped (~setup_fc_params=false); "
                          "assuming SERVO1~%d_FUNCTION=0 is already configured "
                          "in the FC", self.num_thrusters)
            return
        targets = {"SERVO%d_FUNCTION" % i: 0 for i in range(1, self.num_thrusters + 1)}
        self._wait_and_set_params(targets)

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------

    def state_cb(self, msg):
        """缓存飞控连接状态。"""
        self.current_state = msg

    def cmd_cb(self, msg):
        """
        功能：接收 6 推进器 PWM 指令（µs），更新目标输出并刷新看门狗时间戳。
        输入：Float64MultiArray，长度为推进器数（不足补中立值，超出截断），
              值域 [pwm_min, pwm_max]，1500 为停止。
        输出：无显式返回；更新 target_cmd、last_cmd_time、cmd_received。
        """
        if len(msg.data) < self.num_thrusters:
            rospy.logwarn_throttle(5, "cmd has %d values, expected %d; "
                                  "padding with neutral",
                                  len(msg.data), self.num_thrusters)
        cmd = list(msg.data[:self.num_thrusters])
        cmd += [self.neutral_pwm] * (self.num_thrusters - len(cmd))
        self.target_cmd = [self._clamp(v) for v in cmd]
        self.last_cmd_time = rospy.Time.now()
        self.cmd_received = True

    def enable_cb(self, req):
        """
        功能：使能/禁用节点输出；禁用即急停，持续输出中立值。
        输入：SetBool 请求，data=True 使能 / False 禁用。
        输出：SetBoolResponse，恒为 success=True。
        """
        self.enabled = req.data
        if self.enabled:
            rospy.loginfo("Output enabled")
        else:
            self._publish_neutral()
            rospy.logwarn("Output DISABLED: holding neutral on all thrusters")
        return SetBoolResponse(success=True, message="")

    # ------------------------------------------------------------------
    # 控制循环与底层发送
    # ------------------------------------------------------------------

    def control_loop(self, event):
        """
        功能：按 ~send_rate 周期向飞控刷新各推进器 PWM；处理看门狗与急停。
        输入：定时器事件；依赖 target_cmd、enabled 与飞控连接状态。
        输出：无显式返回；逐通道调用 DO_SET_SERVO 并发布 ~cmd_out_topic 回显。
        """
        if not self.current_state.connected:
            rospy.logwarn_throttle(5, "FC not connected; PWM not sent")
            return

        # 看门狗：超时未收到新指令 → 全部输出中立值
        if self.cmd_received and \
                (rospy.Time.now() - self.last_cmd_time).to_sec() > self.watchdog_timeout:
            rospy.logwarn_throttle(5, "Command timeout (%.1f s): holding neutral",
                                   self.watchdog_timeout)
            self.target_cmd = [self.neutral_pwm] * self.num_thrusters
            self.cmd_received = False

        pwm_out = self.target_cmd if self.enabled else \
            [self.neutral_pwm] * self.num_thrusters

        for i, pwm in enumerate(pwm_out, start=1):
            self._send_set_servo(i, pwm)

        out_msg = Float64MultiArray()
        out_msg.data = pwm_out
        self.cmd_out_pub.publish(out_msg)

    def _send_set_servo(self, servo_num, pwm):
        """
        功能：通过 MAV_CMD_DO_SET_SERVO 写入单个输出通道的 PWM。
        输入：servo_num 为 1 起始的通道号，pwm 为已钳位的 PWM 值。
        输出：无显式返回；失败时按节流打印告警。
        """
        try:
            resp = self.cmd_srv(broadcast=False,
                                confirmation=0,
                                command=MAV_CMD_DO_SET_SERVO,
                                param1=float(servo_num),
                                param2=float(pwm),
                                param3=0.0, param4=0.0, param5=0.0,
                                param6=0.0, param7=0.0)
            if not resp.success:
                rospy.logwarn_throttle(5, "DO_SET_SERVO (servo %d) rejected", servo_num)
        except rospy.ServiceException:
            rospy.logwarn_throttle(5, "mavros cmd service unavailable; link down?")

    def _publish_neutral(self):
        """所有推进器输出中立值（急停服务与节点退出时调用）。"""
        if not hasattr(self, 'cmd_srv'):
            return
        for i in range(1, self.num_thrusters + 1):
            self._send_set_servo(i, self.neutral_pwm)
        rospy.logwarn("All thrusters set to neutral (%d us)", int(self.neutral_pwm))

    def _clamp(self, pwm):
        """将 PWM 钳位到 [pwm_min, pwm_max]。"""
        return max(self.pwm_min, min(self.pwm_max, pwm))

    def run(self):
        """进入事件循环，保持节点运行。"""
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ThrusterCmdNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
