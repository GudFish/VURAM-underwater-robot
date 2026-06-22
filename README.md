# sub_planning

`sub_planning` is a ROS `catkin` package for underwater vehicle planning, target tracking, and visualization. The repository contains vehicle control nodes, MPC-based trackers, target/sensor simulators, and RViz visualization tools built around MAVROS and ArduSub-style topics.

## Features

- Body-velocity and optional position control helpers for the vehicle
- MPC trackers for dynamic-target tracking and belief-aware target tracking
- Simulated target generation, acoustic TDOA sensing, and forward sonar sensing
- RViz helpers for boundaries, target belief, TF, and sonar visualization

## Repository Layout

```text
ardupilot_ws/
├── install_sub_planning_deps.sh
└── src/sub_planning/
    ├── config/
    ├── launch/
    ├── scripts/
    ├── src/sub_planning/
    ├── package.xml
    └── requirements.txt
```

## Environment

- Ubuntu with ROS 1 `catkin`
- Recommended ROS distribution: `noetic`
- Python 3
- `rosdep`, `catkin_make`, and `pip`

The launch files assume a MAVROS-based vehicle stack and, by default, include:

- `mavros`
- `rviz`
- `robot_state_publisher`
- `tf` / `tf2_ros`

## Installation

Clone the repository into a catkin workspace, then run the provided install script:

```bash
cd /path/to/ardupilot_ws
chmod +x install_sub_planning_deps.sh
./install_sub_planning_deps.sh
catkin_make
source devel/setup.bash
```

If your ROS distribution is not `noetic`, set it before running the script:

```bash
export ROS_DISTRO=noetic
./install_sub_planning_deps.sh
```

What the script does:

- sources `/opt/ros/$ROS_DISTRO/setup.bash`
- installs ROS dependencies via `rosdep`
- installs Python dependencies from `src/sub_planning/requirements.txt`

What you do after the script finishes:

- build the workspace with `catkin_make`
- source `devel/setup.bash`

## Python Dependencies

The project keeps non-ROS Python dependencies in:

- [src/sub_planning/requirements.txt](/home/lhy/ardupilot_ws/src/sub_planning/requirements.txt)

Current Python dependencies:

- `casadi`
- `numpy`
- `pymap3d`
- `scipy`
- `shapely`

## ROS Dependencies

ROS dependencies are declared in:

- [src/sub_planning/package.xml](/home/lhy/ardupilot_ws/src/sub_planning/package.xml)

This package currently depends on ROS components such as:

- `geometry_msgs`
- `nav_msgs`
- `mavros`
- `mavros_msgs`
- `sensor_msgs`
- `visualization_msgs`
- `rospy`
- `tf`
- `tf2_ros`
- `rviz`
- `robot_state_publisher`

## Common Usage

After building the workspace:

```bash
cd /path/to/ardupilot_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
```

### 1. Vehicle control stack

```bash
roslaunch sub_planning common_vehicle_control.launch
```

This includes the MAVROS bridge launch from `apm.launch`, the boundary loader, control nodes, and RViz support.

### 2. 2D MPC planner

```bash
roslaunch sub_planning mpc_2d.launch
```

### 3. Forward sonar simulation

```bash
roslaunch sub_planning forward_sonar_sim.launch
```

To show the sonar image window:

```bash
roslaunch sub_planning forward_sonar_sim.launch show_window:=true
```

## Notes

- `apm.launch` currently uses `udp://127.0.0.1:14551@14555` as the MAVROS FCU URL.
- Several launch files expect MAVROS topics such as `/mavros/local_position/odom` and `/mavros/state`.
- If you only want dependency installation, run `./install_sub_planning_deps.sh` and compile later when you are ready.

## Troubleshooting

- If `rosdep` fails, initialize it first on the target machine and retry.
- If `catkin_make` is not found, install ROS catkin build tools for your distribution.
- If OpenCV windows do not appear, check that the target machine has a desktop environment and OpenCV GUI support.
- If MAVROS cannot connect, update the FCU URL in [src/sub_planning/launch/apm.launch](/home/lhy/ardupilot_ws/src/sub_planning/launch/apm.launch:1) to match your simulator or hardware.
