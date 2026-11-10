cmake_minimum_required(VERSION 3.10)
project(fls_detector)
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs nav_msgs sensor_msgs std_msgs visualization_msgs rospy roscpp roslib tf2 tf2_ros)
find_package(OpenMP)
catkin_python_setup()
catkin_package(INCLUDE_DIRS include LIBRARIES sonar_renderer
  CATKIN_DEPENDS geometry_msgs nav_msgs sensor_msgs std_msgs visualization_msgs rospy roscpp roslib tf2 tf2_ros)
add_library(sonar_renderer src/sonar_renderer.cpp)
target_compile_features(sonar_renderer PUBLIC cxx_std_14)
target_include_directories(sonar_renderer PUBLIC include)
target_compile_options(sonar_renderer PRIVATE -O3 -Wall -Wextra)
if(OpenMP_CXX_FOUND)
  target_link_libraries(sonar_renderer PUBLIC OpenMP::OpenMP_CXX)
endif()
add_executable(forward_sonar_raytrace_node src/forward_sonar_raytrace_node.cpp)
target_include_directories(forward_sonar_raytrace_node PRIVATE ${catkin_INCLUDE_DIRS})
target_link_libraries(forward_sonar_raytrace_node sonar_renderer ${catkin_LIBRARIES})
target_compile_options(forward_sonar_raytrace_node PRIVATE -O3 -Wall -Wextra)
if(CATKIN_ENABLE_TESTING)
  add_executable(test_sonar_renderer tests/test_sonar_renderer.cpp)
  target_link_libraries(test_sonar_renderer sonar_renderer)
  add_test(NAME sonar_renderer COMMAND test_sonar_renderer
    ${CMAKE_CURRENT_SOURCE_DIR}/../vuram_description/meshes/AUV.STL)
  add_test(NAME sonar_tools COMMAND ${PYTHON_EXECUTABLE} -m pytest -q
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sonar_tools.py)
endif()
add_executable(benchmark_sonar_renderer tests/benchmark_sonar_renderer.cpp)
target_link_libraries(benchmark_sonar_renderer sonar_renderer)
catkin_install_python(PROGRAMS
  scripts/sensors/forward_sonar_target_detector.py
  scripts/sensors/forward_sonar_sim.py
  scripts/sensors/batch_sonar_image_detector.py
  scripts/sensors/sonar_separation_heatmap.py
  scripts/visualization/forward_sonar_image_viewer.py
  scripts/visualization/forward_sonar_range_viz.py
  scripts/calibrate_sonar_noise.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(TARGETS sonar_renderer forward_sonar_raytrace_node benchmark_sonar_renderer
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
install(DIRECTORY launch config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
