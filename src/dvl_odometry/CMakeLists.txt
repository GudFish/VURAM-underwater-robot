cmake_minimum_required(VERSION 3.0.2)
project(dvl_odometry)

find_package(catkin REQUIRED COMPONENTS
  diagnostic_msgs
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
  CATKIN_DEPENDS diagnostic_msgs geometry_msgs nav_msgs sensor_msgs std_msgs std_srvs tf tf2_ros rospy
)

catkin_python_setup()

catkin_install_python(PROGRAMS
  scripts/dvl_velocity_node.py
  scripts/dvl_fusion_node.py
  scripts/mavros_dvl_fusion_node.py
  scripts/demos/mavros_imu_simulator.py
  scripts/dvl_odometry_node.py
  scripts/demos/dvl_simulator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(FILES README.md DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
