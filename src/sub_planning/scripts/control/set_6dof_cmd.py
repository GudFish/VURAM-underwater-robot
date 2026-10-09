#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
set_6dof_cmd.py

推进器控制节点（RC override 方式）：接收 6 个自由度（DOF）的 RC 通道 PWM
指令（µs，1100~1900，1500 中立，不做归一化），通过 MAVLink
RC_CHANNELS_OVERRIDE 原样发送给飞控，由 ArduSub 的 MANUAL 模式混控器
解算到各推进器输出。本方案保留飞控的解锁（arm）门控：未解锁时飞控不会
驱动推力器。

通道语义（本机 ArduSub 4.2 固件 radio.cpp 定义，非常规 1~4 顺序）：
  CH1 = pitch    CH2 = roll    CH3 = throttle（垂向）
  CH4 = yaw      CH5 = forward CH6 = lateral

指令语义：输入值即飞控 RC 通道的原始 PWM（µs），1500 为中立。
飞控内部按 norm = (pwm - 1500) / 400 归一化（RCx_MIN/MAX=1100/1900，
无死区、无 expo）。例外：CH4 (yaw) 在飞控内额外乘以 ACRO_YAW_P/4.5 ≈ 0.75
（当前飞控 ACRO_YAW_P=3.375），满量程指令实际只有 0.75 的 yaw 输出。

关键前提（节点启动时默认自动设置，可用相应参数关闭）：
  1. SYSID_MYGCS = 1：飞控只接受"控制 GCS"发来的 override。MAVROS 的
     sysid 是 1，而本飞控默认 SYSID_MYGCS=255，不设置则指令被静默丢弃。
     注意：设置后 QGC 若要保持控制权，需把 QGC 的 MAVLink system ID
     改为 1（Application Settings -> Comm Links）。
  2. SERVO1~SERVO6_FUNCTION = 33~38（MOTOR1~MOTOR6）：若此前按
     DO_SET_SERVO 直控方案改成了 0（禁用），混控器不工作，需恢复。
  3. 飞控处于 MANUAL 模式：其他模式（GUIDED/ALT_HOLD 等）不消费
     RC override 输入。节点启动时自动切换。

安全（飞控侧，无需本节点存活）：
  - 未解锁：推力器被飞控锁死，无任何输出。
  - 节点死掉/链路中断：override 在 RC_OVERRIDE_TIME（3 s）后过期，
    且 FS_PILOT_INPUT=2 的"无操控输入 3 s 自动 disarm"随即触发——
    飞控自动解锁停机，不会保持最后指令。若 FS_PILOT_INPUT=0（禁用），
    此安全网失效，启动时会打印告警。

接口
----
订阅 ~cmd_topic        std_msgs/Float64MultiArray   6 通道 PWM (us, 1100~1900, 1500 中立)
服务 ~enable           std_srvs/SetBool             使能/急停（禁用时持续发中立值）
发布 ~cmd_out_topic    std_msgs/Float64MultiArray   实际输出 PWM 回显（调试/记录）

安全
----
- 看门狗：超过 ~watchdog_timeout 未收到新指令 → 全部输出中立值 ~neutral_pwm
- 节点退出（Ctrl-C / kill）时输出中立值，随后由飞控 failsafe 接管
- 输入一律钳位到 [~pwm_min, ~pwm_max]

用法
----
  rosrun sub_planning set_6dof_cmd.py
  # 解锁后测试：CH5 前进方向小幅输出
  rostopic pub -r 10 /dof_cmd std_msgs/Float64MultiArray \
      "data: [1500.0, 1500.0, 1500.0, 1500.0, 1550.0, 1500.0]"
"""

import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import rospy
from mavros_msgs.msg import State, ParamValue, OverrideRCIn
from mavros_msgs.srv import CommandBool, ParamGet, ParamSet, SetMode
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, SetBoolResponse

# ArduSub 4.2 radio.cpp：RC1~RC6 与 6 自由度命令的对应关系
CHANNEL_MEANING = {1: 'pitch', 2: 'roll', 3: 'throttle', 4: 'yaw', 5: 'forward', 6: 'lateral'}

# SERVOx_FUNCTION 取值：MOTOR1~MOTOR6（与飞控出厂参数一致）
MOTOR_FUNCTIONS = [33, 34, 35, 36, 37, 38, 39, 40]

# RC_CHANNELS_OVERRIDE 中"忽略该通道"的取值（MAVLink 规范）
RC_CHANNEL_IGNORE = 65535


class SixDofCmdNode:
    """
    功能：订阅 6 通道 RC PWM 指令（µs），以 RC_CHANNELS_OVERRIDE 形式
          发给飞控，附带看门狗、急停服务与飞控参数/模式/解锁状态管理。
    输入：~cmd_topic 上的 6 通道 PWM 指令（µs，1500 中立），以及 MAVROS
          状态与参数服务。
    输出：/mavros/rc/override 指令流，~cmd_out_topic 回显。
    """

    def __init__(self):
        rospy.init_node('set_6dof_cmd', anonymous=True)

        # ---- ROS 参数 ----
        self.cmd_topic = rospy.get_param('~cmd_topic', '/dof_cmd')
        self.cmd_out_topic = rospy.get_param('~cmd_out_topic', '/dof_cmd/out')
        self.num_channels = int(rospy.get_param('~num_channels', 6))
        # RC 通道 PWM 参数（与飞控 RCx_TRIM/MIN/MAX 一致）
        self.neutral_pwm = float(rospy.get_param('~neutral_pwm', 1500.0))
        self.pwm_min = float(rospy.get_param('~pwm_min', 1100.0))
        self.pwm_max = float(rospy.get_param('~pwm_max', 1900.0))
        self.send_rate = float(rospy.get_param('~send_rate', 50.0))
        self.watchdog_timeout = float(rospy.get_param('~watchdog_timeout', 1.0))
        self.mode = rospy.get_param('~mode', 'MANUAL')
        self.arm_vehicle = rospy.get_param('~arm_vehicle', False)
        # 飞控参数自动配置总开关：参数持久化在飞控中，首次成功设置后
        # 可置 false 跳过（如已在 QGC 中手动改好 SYSID_MYGCS 与 SERVOx_FUNCTION）
        self.setup_fc_params = rospy.get_param('~setup_fc_params', True)
        # 细分开关（仅 setup_fc_params=true 时生效）
        self.set_gcs_sysid = rospy.get_param('~set_gcs_sysid', True)
        self.gcs_sysid = int(rospy.get_param('~gcs_sysid', 1))  # MAVROS 自身 sysid
        self.restore_servo_functions = rospy.get_param('~restore_servo_functions', True)
        # 等待 MAVROS 参数表下载完成的总超时（ArduSub 参数约上千个，下载需数秒到数十秒）
        self.param_wait_timeout = float(rospy.get_param('~param_wait_timeout', 180.0))

        # MAVROS 话题/服务名
        self.state_topic = rospy.get_param('~state_topic', '/mavros/state')
        self.override_topic = rospy.get_param('~override_topic', '/mavros/rc/override')
        self.param_get_service = rospy.get_param('~param_get_service', '/mavros/param/get')
        self.param_set_service = rospy.get_param('~param_set_service', '/mavros/param/set')
        self.arming_service = rospy.get_param('~arming_service', '/mavros/cmd/arming')
        self.set_mode_service = rospy.get_param('~set_mode_service', '/mavros/set_mode')

        if not 1 <= self.num_channels <= 6:
            rospy.logfatal("num_channels=%d 超出 ArduSub 6 自由度通道范围 [1, 6]",
                           self.num_channels)
            sys.exit(1)

        # ---- 状态变量 ----
        self.current_state = State()
        self.enabled = True
        self.target_cmd = [self.neutral_pwm] * self.num_channels
        self.last_cmd_time = rospy.Time.now()
        self.cmd_received = False

        # ---- 订阅 / 发布 / 服务 ----
        self.state_sub = rospy.Subscriber(self.state_topic, State, self.state_cb)
        self.cmd_sub = rospy.Subscriber(self.cmd_topic, Float64MultiArray, self.cmd_cb)
        self.cmd_out_pub = rospy.Publisher(self.cmd_out_topic, Float64MultiArray, queue_size=1)
        self.override_pub = rospy.Publisher(self.override_topic, OverrideRCIn, queue_size=1)
        self.enable_srv = rospy.Service('~enable', SetBool, self.enable_cb)

        # ---- 等待 MAVROS 连接与所需服务就绪 ----
        self._wait_for_mavros()

        self.param_get_srv = rospy.ServiceProxy(self.param_get_service, ParamGet, persistent=True)
        self.param_set_srv = rospy.ServiceProxy(self.param_set_service, ParamSet, persistent=True)
        self.arming_srv = rospy.ServiceProxy(self.arming_service, CommandBool, persistent=True)
        self.set_mode_srv = rospy.ServiceProxy(self.set_mode_service, SetMode, persistent=True)

        # ---- 启动前的飞控配置 ----
        self._setup_fc_params()
        self._check_fc_failsafes()
        self._set_mode()
        if self.arm_vehicle:
            self._arm()
        if not self.current_state.armed:
            rospy.logwarn("Vehicle is not armed. ArduSub will ignore RC commands "
                          "until armed (arm via RC switch or /mavros/cmd/arming).")

        # ---- 控制定时器与退出钩子 ----
        period = 1.0 / max(self.send_rate, 1e-3)
        self.timer = rospy.Timer(rospy.Duration(period), self.control_loop)
        rospy.on_shutdown(self._publish_neutral)

        rospy.loginfo("Thruster command node ready: %d channels at %.1f Hz, "
                      "PWM input [%.0f, %.0f] us, neutral %.0f us, mode %s",
                      self.num_channels, self.send_rate,
                      self.pwm_min, self.pwm_max, self.neutral_pwm, self.mode)
        rospy.loginfo("Channel mapping: %s",
                      ", ".join("CH%d=%s" % (i, CHANNEL_MEANING[i])
                                for i in range(1, self.num_channels + 1)))

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
                    rospy.wait_for_service(self.param_get_service, timeout=1.0)
                    rospy.wait_for_service(self.param_set_service, timeout=1.0)
                    rospy.wait_for_service(self.arming_service, timeout=1.0)
                    rospy.wait_for_service(self.set_mode_service, timeout=1.0)
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
        输出：全部设置成功返回 True，超时返回 False（节点继续运行但可能不受控）。
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
        功能：自动配置飞控参数——SYSID_MYGCS 与 SERVOx_FUNCTION 恢复。
        输入：无显式参数；依赖 setup_fc_params、gcs_sysid、restore_servo_functions 开关。
        输出：无显式返回；打印各参数的设置结果。
        """
        if not self.setup_fc_params:
            rospy.loginfo("FC parameter auto-setup skipped (~setup_fc_params=false); "
                          "assuming SYSID_MYGCS=%d and SERVO1~%d_FUNCTION=33~38 "
                          "are already configured in the FC",
                          self.gcs_sysid, self.num_channels)
            return
        targets = {}
        if self.set_gcs_sysid:
            # 飞控只接受"控制 GCS"（SYSID_MYGCS）发来的 RC override，
            # MAVROS 的 sysid 是 1，不设则指令被静默丢弃
            targets["SYSID_MYGCS"] = self.gcs_sysid
        if self.restore_servo_functions:
            # 若此前用 DO_SET_SERVO 直控方案把 SERVOx_FUNCTION 设成了 0，
            # 恢复为 MOTOR1~MOTOR6，混控器才能工作
            for i in range(1, self.num_channels + 1):
                targets["SERVO%d_FUNCTION" % i] = MOTOR_FUNCTIONS[i - 1]
        if targets:
            self._wait_and_set_params(targets)

    def _check_fc_failsafes(self):
        """
        功能：读取并核对飞控侧安全网参数（override 过期时间与无操控输入 failsafe）。
        输入：无显式参数；依赖 param_get 服务。
        输出：无显式返回；打印安全网状态，失效时告警。
        """
        fs_pilot = self._param_get_int("FS_PILOT_INPUT")
        rc_timeout = self._param_get_int("RC_OVERRIDE_TIME")
        if fs_pilot is None or rc_timeout is None:
            rospy.logwarn("Cannot read FS_PILOT_INPUT/RC_OVERRIDE_TIME; "
                          "FC-side failsafe state unknown")
            return
        rospy.loginfo("FC-side safety net: RC_OVERRIDE_TIME=%d s, FS_PILOT_INPUT=%d",
                      rc_timeout, fs_pilot)
        if fs_pilot == 0:
            rospy.logwarn("FS_PILOT_INPUT=0 (disabled): if this node dies, the FC "
                          "will NOT auto-disarm. Consider setting FS_PILOT_INPUT=2.")

    def _set_mode(self):
        """
        功能：切换飞控到 ~mode 模式（默认 MANUAL，RC override 仅在此模式生效）。
        输入：无显式参数；依赖 set_mode 服务。
        输出：无显式返回；打印切换结果。
        """
        try:
            resp = self.set_mode_srv(base_mode=0, custom_mode=self.mode)
            if resp.mode_sent:
                rospy.loginfo("Mode set to %s", self.mode)
            else:
                rospy.logwarn("Failed to set mode %s", self.mode)
        except rospy.ServiceException as e:
            rospy.logwarn("set_mode failed: %s", e)

    def _arm(self):
        """
        功能：通过 /mavros/cmd/arming 解锁（可选，默认关闭，建议手动解锁）。
        输入：无显式参数；依赖 arming 服务。
        输出：无显式返回；打印解锁结果。
        """
        try:
            resp = self.arming_srv(value=True)
            if resp.success:
                rospy.loginfo("Vehicle armed")
            else:
                rospy.logwarn("Arming rejected by autopilot")
        except rospy.ServiceException as e:
            rospy.logwarn("arming service failed: %s", e)

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------

    def state_cb(self, msg):
        """缓存飞控连接/模式/解锁状态。"""
        self.current_state = msg

    def cmd_cb(self, msg):
        """
        功能：接收 6 通道 RC PWM 指令（µs），更新目标输出并刷新看门狗时间戳。
        输入：Float64MultiArray，长度为通道数（不足补中立值，超出截断），
              值域 [pwm_min, pwm_max]，1500 为中立。
        输出：无显式返回；更新 target_cmd、last_cmd_time、cmd_received。
        """
        if len(msg.data) < self.num_channels:
            rospy.logwarn_throttle(5, "cmd has %d values, expected %d; "
                                  "padding with neutral",
                                  len(msg.data), self.num_channels)
        cmd = list(msg.data[:self.num_channels])
        cmd += [self.neutral_pwm] * (self.num_channels - len(cmd))
        self.target_cmd = [self._clamp(v) for v in cmd]
        self.last_cmd_time = rospy.Time.now()
        self.cmd_received = True

    def enable_cb(self, req):
        """
        功能：使能/禁用节点输出；禁用即急停，持续输出零指令（不停发，以维持
              飞控 FS_PILOT_INPUT 计时器，避免误触发 disarm）。
        输入：SetBool 请求，data=True 使能 / False 禁用。
        输出：SetBoolResponse，恒为 success=True。
        """
        self.enabled = req.data
        if self.enabled:
            rospy.loginfo("Output enabled")
        else:
            self._publish_neutral()
            rospy.logwarn("Output DISABLED: holding neutral on all channels")
        return SetBoolResponse(success=True, message="")

    # ------------------------------------------------------------------
    # 控制循环与底层发送
    # ------------------------------------------------------------------

    def control_loop(self, event):
        """
        功能：按 ~send_rate 周期向飞控发布 RC override；处理看门狗与急停。
        输入：定时器事件；依赖 target_cmd、enabled 与飞控连接状态。
        输出：无显式返回；发布 /mavros/rc/override 与 ~cmd_out_topic 回显。
        """
        if not self.current_state.connected:
            rospy.logwarn_throttle(5, "FC not connected; override not sent")
            return
        if self.current_state.mode != self.mode:
            rospy.logwarn_throttle(5, "FC mode is %s, not %s; RC override is ignored "
                                   "in this mode", self.current_state.mode, self.mode)
        if not self.current_state.armed:
            rospy.logwarn_throttle(5, "Vehicle not armed; FC will ignore RC commands "
                                   "until armed")

        # 看门狗：超时未收到新指令 → 全部输出中立值
        if self.cmd_received and \
                (rospy.Time.now() - self.last_cmd_time).to_sec() > self.watchdog_timeout:
            rospy.logwarn_throttle(5, "Command timeout (%.1f s): holding neutral",
                                   self.watchdog_timeout)
            self.target_cmd = [self.neutral_pwm] * self.num_channels
            self.cmd_received = False

        cmd_out = self.target_cmd if self.enabled else \
            [self.neutral_pwm] * self.num_channels
        self._publish_override(cmd_out)

        out_msg = Float64MultiArray()
        out_msg.data = cmd_out
        self.cmd_out_pub.publish(out_msg)

    def _publish_override(self, pwm_list):
        """
        功能：将通道 PWM（µs）打包为 RC_CHANNELS_OVERRIDE 并发布。
        输入：pwm_list 为已钳位的通道 PWM 列表。
        输出：无显式返回；未使用的通道填 65535（忽略，不影响 RC 遥控器/云台等）。
        """
        msg = OverrideRCIn()
        channels = list(pwm_list) + \
            [RC_CHANNEL_IGNORE] * (len(msg.channels) - len(pwm_list))
        msg.channels = [int(round(v)) for v in channels]
        self.override_pub.publish(msg)

    def _publish_neutral(self):
        """所有受控通道输出中立值（急停服务与节点退出时调用）。"""
        if not hasattr(self, 'override_pub'):
            return
        self._publish_override([self.neutral_pwm] * self.num_channels)
        rospy.logwarn("All channels set to neutral (%d us)", int(self.neutral_pwm))

    def _clamp(self, pwm):
        """将通道 PWM 钳位到 [pwm_min, pwm_max]。"""
        return max(self.pwm_min, min(self.pwm_max, pwm))

    def run(self):
        """进入事件循环，保持节点运行。"""
        rospy.spin()


if __name__ == '__main__':
    try:
        node = SixDofCmdNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
