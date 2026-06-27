#!/usr/bin/env bash

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="${WORKSPACE_DIR}/src/sub_planning"
REQUIREMENTS_FILE="${PACKAGE_DIR}/requirements.txt"
ROS_DISTRO="${ROS_DISTRO:-noetic}"

if [[ ! -f "${PACKAGE_DIR}/package.xml" ]]; then
  echo "Cannot find sub_planning package under ${PACKAGE_DIR}" >&2
  exit 1
fi

if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
else
  echo "ROS environment /opt/ros/${ROS_DISTRO}/setup.bash was not found." >&2
  echo "Set ROS_DISTRO first, for example: export ROS_DISTRO=noetic" >&2
  exit 1
fi

if ! command -v catkin_make >/dev/null 2>&1; then
  echo "catkin_make is not available. This is fine for dependency installation only." >&2
fi

ROS_APT_PACKAGES=(
  ros-${ROS_DISTRO}-geometry-msgs
  ros-${ROS_DISTRO}-nav-msgs
  ros-${ROS_DISTRO}-mavros-msgs
  ros-${ROS_DISTRO}-sensor-msgs
  ros-${ROS_DISTRO}-rospy
  ros-${ROS_DISTRO}-std-msgs
  ros-${ROS_DISTRO}-visualization-msgs
  ros-${ROS_DISTRO}-mavros
  ros-${ROS_DISTRO}-robot-state-publisher
  ros-${ROS_DISTRO}-tf
  ros-${ROS_DISTRO}-tf2-ros
  ros-${ROS_DISTRO}-rviz
)

SYSTEM_APT_PACKAGES=(
  python3-opencv
  python3-rospkg
)

echo "Installing ROS dependencies with apt..."
sudo apt update
sudo apt install -y "${ROS_APT_PACKAGES[@]}" "${SYSTEM_APT_PACKAGES[@]}"

echo "Installing Python dependencies from ${REQUIREMENTS_FILE}..."
python3 -m pip install -r "${REQUIREMENTS_FILE}"

echo
echo "Dependency installation complete."
echo "Next step:"
echo "  catkin_make --directory \"${WORKSPACE_DIR}\""
echo "  source \"${WORKSPACE_DIR}/devel/setup.bash\""
