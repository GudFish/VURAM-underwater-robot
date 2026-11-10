#!/usr/bin/env python3
"""Vuram RL policy deployment node on the ArduSub/MAVROS stack.

Loads the MuJoCo-trained TorchScript policy from --model_dir (VND v4:
deploy.pt, obs[21] + history[20,18] + valid -> a[5] normalized wrench),
runs the 20 Hz closed loop with internal/external references, or continuous
horizontal figure8 collection (~ref_source=collect, policy_deploy.launch):
odom -> obs[21] -> a[5] wrench -> thruster PWM -> /dof_cmd
(the set_6dof_cmd node forwards each channel via RC override, VECTORED
frame). Reference (green) and actual (orange) trajectories are published
for RViz.

用法：
  roslaunch vuram_control policy_deploy.launch
  rosrun vuram_control policy_inference_node.py \
      --model_dir $(rospack find vuram_control)/scripts/vuram_policy/models/policy_v000012 \
      --spec $(rospack find vuram_control)/config/obs_act.yaml
"""
import argparse
import os
import sys
from pathlib import Path

import rospkg
import rospy

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from vuram_policy.policy_inference import run_policy_node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    pkg_path = rospkg.RosPack().get_path('vuram_control')
    parser.add_argument('--model_dir',
                        default='',
                        help='snapshot directory; empty uses latest.json in collect '
                             'mode or the bundled policy_v000012 in legacy modes')
    parser.add_argument('--spec',
                        default=os.path.join(pkg_path, 'config',
                                             'obs_act.yaml'),
                        help='obs_act.yaml')
    parser.add_argument('--ref_family', default='',
                        help="reference family: line | lissajous | "
                             "waypoints_dp ('' = random)")
    parser.add_argument('--episode_len_s', type=float, default=60.0,
                        help='internal duration / collect lap period [s]')
    # rospy.myargv() strips the __name:= / __log:= remapping args that
    # roslaunch appends to node args (parse_args would reject them -> exit 2)
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('policy_inference', anonymous=False)
    run_policy_node(args.model_dir, args.spec,
                    ref_family=args.ref_family,
                    episode_len_s=args.episode_len_s)


if __name__ == '__main__':
    main()
