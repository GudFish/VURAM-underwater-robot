#!/usr/bin/env python3
"""Run the PD controller with the shared figure8, safety and episode lifecycle."""
import sys
from pathlib import Path

import rospkg
import rospy

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from control.observation import load_spec
from control.pd_controller import PDController
from control.tracking_node import StaticControllerSource, run_tracking_node


def run_pd_node(spec, episode_len_s):
    def source_factory(contract, model_root):
        controller = PDController.from_ros_params(contract.allocator)
        controller.validate_spec(spec)
        return StaticControllerSource(controller)

    return run_tracking_node(spec, episode_len_s, source_factory)


def main():
    rospy.init_node('pd_controller', anonymous=False)
    package = Path(rospkg.RosPack().get_path('vuram_control'))
    spec = load_spec(rospy.get_param('~spec', str(package / 'config/obs_act.yaml')))
    run_pd_node(spec, float(rospy.get_param('~episode_len_s', 20.)))


if __name__ == '__main__':
    main()
