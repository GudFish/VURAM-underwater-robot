#!/usr/bin/env python3
"""Vuram RL policy deployment node on the ArduSub/MAVROS stack.

Loads the MuJoCo-trained TorchScript policy from --model_dir (VND v4:
deploy.pt, obs[21] + history[20,18] + valid -> a[5] normalized wrench),
samples a reference trajectory (line | lissajous | waypoints_dp) anchored
at the vehicle's current pose, and runs the 20 Hz closed loop:
odom -> obs[21] -> a[5] wrench -> thruster PWM -> /dof_cmd
(the set_6dof_cmd node forwards each channel via RC override, CUSTOM 1:1
frame). Reference (green) and actual (orange) trajectories are published
for RViz.

用法：
  roslaunch vuram_control policy_deploy.launch
  rosrun vuram_control policy_inference_node.py \
      --model_dir $(rospack find vuram_control)/scripts/vuram_policy/models/best \
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

# 模型目录随包内脚本位置（scripts/vuram_policy/models/），与 launch 的
# model_dir 默认值保持一致
MODELS_DIR = Path(__file__).resolve().parent / 'models'

from vuram_policy.policy_inference import run_policy_node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    pkg_path = rospkg.RosPack().get_path('vuram_control')
    parser.add_argument('--model_dir',
                        default=str(MODELS_DIR / 'policy_v15'),
                        help='directory containing deploy.pt (+ spec.json)')
    parser.add_argument('--spec',
                        default=os.path.join(pkg_path, 'config',
                                             'obs_act.yaml'),
                        help='obs_act.yaml')
    parser.add_argument('--ref_family', default='',
                        help="reference family: line | lissajous | "
                             "waypoints_dp ('' = random)")
    parser.add_argument('--episode_len_s', type=float, default=60.0,
                        help='deploy duration [s] at the 20 Hz control rate')
    # rospy.myargv() strips the __name:= / __log:= remapping args that
    # roslaunch appends to node args (parse_args would reject them -> exit 2)
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('policy_inference', anonymous=False)
    run_policy_node(args.model_dir, args.spec,
                    ref_family=args.ref_family,
                    episode_len_s=args.episode_len_s)


if __name__ == '__main__':
    main()
