cmake_minimum_required(VERSION 3.0.2)
project(kiss_slam_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  tf
)

catkin_python_setup()

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

install(PROGRAMS
  src/kiss_slam_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
