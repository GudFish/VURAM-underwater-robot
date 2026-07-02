cmake_minimum_required(VERSION 3.0.2)
project(sonar_driver)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
)

## System dependencies
find_package(OpenCV REQUIRED)

## Qt5 — SDK 头文件依赖 Qt5 Widgets/Network/OpenGL
find_package(Qt5 REQUIRED COMPONENTS
  Core
  Widgets
  Network
  Gui
)

## 启用 Qt MOC (Meta-Object Compiler)，处理 Q_OBJECT 宏
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES sonar_driver
#  CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## 头文件路径
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Qt5Core_INCLUDE_DIRS}
  ${Qt5Widgets_INCLUDE_DIRS}
  ${Qt5Network_INCLUDE_DIRS}
  ${Qt5Gui_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/sdk_linux_aarch/include
)

## SDK 库路径
link_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/sdk_linux_aarch/lib
)

## 可执行文件
add_executable(sonar_image_pub
  src/sonar_image_pub.cpp
  src/sonar_image_pub.h
)

## 链接库
target_link_libraries(sonar_image_pub
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Qt5::Core
  Qt5::Widgets
  Qt5::Network
  Qt5::Gui
  ForwardlookingSonarSDK
)
