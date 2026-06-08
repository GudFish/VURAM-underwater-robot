#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sub_planning.models.fossen3 import Fossen3DOFModel


def main():
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
