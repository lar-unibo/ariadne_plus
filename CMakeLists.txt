cmake_minimum_required(VERSION 2.8.3)
project(ariadne_plus)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED
  COMPONENTS
  message_generation
  geometry_msgs
  std_msgs
  sensor_msgs
  actionlib_msgs
)

add_message_files(DIRECTORY msg
  FILES
  spline_tck.msg
)

add_action_files(DIRECTORY msg
  FILES
  spline_tck.action 
)

add_service_files(DIRECTORY srv
  FILES
  getSplines.srv 
)

generate_messages(DEPENDENCIES
  std_msgs
  actionlib_msgs
  geometry_msgs
  sensor_msgs
)


catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    std_msgs
    message_generation
    message_runtime
)