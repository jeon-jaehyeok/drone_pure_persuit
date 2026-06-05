cmake_minimum_required(VERSION 3.0.2)
project(pure_pursuit_drone)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  mavros_msgs
  std_msgs
)

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/drone_pure_pursuit.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
