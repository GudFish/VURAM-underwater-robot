#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：演示如何调用 Fossen 三自由度动力学模型完成简单的离散步进仿真。
输入：通过函数参数或上层调用提供动力学、噪声和状态等模型输入。
输出：返回模型计算结果，供规划器、估计器或演示脚本直接调用。
"""
import rospy
from models.fossen3 import Fossen3DOFModel


def main():
    """
    功能：运行一个简化的模型演示流程并输出结果日志。
    输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
    输出：无显式返回；通过更新内部状态或维护对象成员完成功能。
    """
    rospy.init_node('fossen_model_demo', anonymous=True)
    params = rospy.get_param('~dynamics', {})
    model = Fossen3DOFModel(params)

    eta = [0.0, 0.0, 0.0]
    nu = [0.0, 0.0, 0.0]
    tau = [30.0, 0.0, 5.0]
    dt = 0.1

    for _ in range(20):
        eta, nu = model.step(eta, nu, tau, dt)

    rospy.loginfo("[FossenDemo] eta=(%.3f, %.3f, %.3f), nu=(%.3f, %.3f, %.3f)",
                  eta[0], eta[1], eta[2], nu[0], nu[1], nu[2])


if __name__ == '__main__':
    main()
