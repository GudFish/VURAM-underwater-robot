#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：在仿真启动后自动将飞控切换到 GUIDED 模式并解锁（ARM）。
      等待场景（目标与障碍物）生成就绪后，再延迟 arm_delay 秒，
      然后依次调用 MAVROS 服务设置 GUIDED 并 ARM。
      这样确保每次测试在相同的场景初始状态下开始，保证对比公正。

用法：
  rosrun vuram_control auto_arm_guided.py _arm_delay:=5.0 _scene_timeout:=60.0
"""
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, SetMode


def main():
    rospy.init_node('auto_arm_guided', anonymous=True)

    arm_delay = rospy.get_param('~arm_delay', 5.0)
    retry_timeout = rospy.get_param('~retry_timeout', 30.0)
    scene_timeout = rospy.get_param('~scene_timeout', 60.0)
    scene_ready_topic = rospy.get_param('~scene_ready_topic', '/dynamic_target/pose')

    # ====================================================================
    #  阶段 1: 等待场景就绪（目标与障碍物已生成）
    # ====================================================================
    scene_ready = {'flag': False}

    def scene_cb(_msg):
        if not scene_ready['flag']:
            rospy.loginfo('[AutoArm] scene ready — first target pose received')
        scene_ready['flag'] = True

    sub = rospy.Subscriber(scene_ready_topic, PoseStamped, scene_cb, queue_size=1)

    rospy.loginfo('[AutoArm] waiting for scene to be ready (topic: %s, timeout: %.1fs)...',
                  scene_ready_topic, scene_timeout)
    start_wait = rospy.Time.now()
    while not rospy.is_shutdown() and not scene_ready['flag']:
        elapsed = (rospy.Time.now() - start_wait).to_sec()
        if elapsed > scene_timeout:
            rospy.logerr('[AutoArm] scene not ready after %.1fs, aborting', scene_timeout)
            return
        rospy.sleep(0.5)

    if rospy.is_shutdown():
        return

    sub.unregister()

    # ====================================================================
    #  阶段 2: 场景就绪后再等待 arm_delay 秒（让场景稳定）
    # ====================================================================
    rospy.loginfo('[AutoArm] scene ready, waiting additional %.1fs before arming...', arm_delay)
    rospy.sleep(arm_delay)

    if rospy.is_shutdown():
        return

    # ====================================================================
    #  阶段 3: 等待 MAVROS 服务就绪
    # ====================================================================
    rospy.loginfo('[AutoArm] waiting for MAVROS services...')
    try:
        rospy.wait_for_service('/mavros/set_mode', timeout=10.0)
        rospy.wait_for_service('/mavros/cmd/arming', timeout=10.0)
    except rospy.ROSException as e:
        rospy.logerr('[AutoArm] MAVROS service not available: %s', str(e))
        return

    set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
    arming = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)

    # ====================================================================
    #  阶段 4: 设置 GUIDED 模式
    # ====================================================================
    deadline = rospy.Time.now().to_sec() + retry_timeout
    guided_ok = False
    while not rospy.is_shutdown() and rospy.Time.now().to_sec() < deadline:
        try:
            resp = set_mode(custom_mode='GUIDED')
            if resp.mode_sent:
                rospy.loginfo('[AutoArm] GUIDED mode set successfully')
                guided_ok = True
                break
            rospy.logwarn('[AutoArm] set_mode returned mode_sent=False, retrying...')
        except rospy.ServiceException as e:
            rospy.logwarn('[AutoArm] set_mode service call failed: %s', str(e))
        rospy.sleep(2.0)

    if not guided_ok:
        rospy.logerr('[AutoArm] failed to set GUIDED mode within %.1fs', retry_timeout)
        return

    rospy.sleep(1.0)  # 等待模式切换生效

    # ====================================================================
    #  阶段 5: 解锁 ARM
    # ====================================================================
    deadline = rospy.Time.now().to_sec() + retry_timeout
    arm_ok = False
    while not rospy.is_shutdown() and rospy.Time.now().to_sec() < deadline:
        try:
            resp = arming(value=True)
            if resp.success:
                rospy.loginfo('[AutoArm] vehicle ARMED successfully')
                arm_ok = True
                break
            rospy.logwarn('[AutoArm] arming returned success=False, retrying...')
        except rospy.ServiceException as e:
            rospy.logwarn('[AutoArm] arming service call failed: %s', str(e))
        rospy.sleep(2.0)

    if not arm_ok:
        rospy.logerr('[AutoArm] failed to ARM within %.1fs', retry_timeout)
        return

    rospy.loginfo('[AutoArm] auto-arm sequence complete. node exiting.')
    rospy.signal_shutdown('auto_arm_complete')


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
