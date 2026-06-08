# Install script for directory: /home/lhy/ardupilot_ws/src/sub_planning

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/lhy/ardupilot_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/sub_planning.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sub_planning/cmake" TYPE FILE FILES
    "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/sub_planningConfig.cmake"
    "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/sub_planningConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sub_planning" TYPE FILE FILES "/home/lhy/ardupilot_ws/src/sub_planning/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/set_body_vel_yaw.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/depth_hold_velocity_mux.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/mpc_2d_planner.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/mpc_dynamic_target_tracker.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/mpc_belief_target_tracker.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/acoustic_range_sensor_sim.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/acoustic_tdoa_sensor_sim.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/tdoa_bearing_estimator.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/target_belief_viz.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/target_state_estimator_ekf.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/target_bearing_estimator_ekf.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/target_bearing_window_ls.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/fossen_model_demo.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/set_local_position.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/tf_baselink_map_node.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/boundary_loader.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sub_planning" TYPE PROGRAM FILES "/home/lhy/ardupilot_ws/build/sub_planning/catkin_generated/installspace/dynamic_target_generator.py")
endif()

