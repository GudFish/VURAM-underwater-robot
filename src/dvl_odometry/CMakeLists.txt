cmake_minimum_required(VERSION 3.0.2)
project(dvl_odometry)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  tf
  tf2_ros
  rospy
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs sensor_msgs std_msgs std_srvs tf tf2_ros rospy
)

catkin_python_setup()

catkin_install_python(PROGRAMS
  scripts/dvl_odometry_node.py
  scripts/demos/dvl_simulator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
