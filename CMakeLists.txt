cmake_minimum_required(VERSION 2.8.3)
project(objdetection)

find_package(catkin REQUIRED COMPONENTS
    rospy
    std_msgs
    sensor_msgs
    message_generation)

catkin_python_setup()


################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(
    FILES
    Detection.msg
    Segmentation.msg
    Object.msg
)

generate_messages(
    DEPENDENCIES
    std_msgs sensor_msgs
)


###################################
## catkin specific configuration ##
###################################

catkin_package(CATKIN_DEPENDS rospy std_msgs sensor_msgs message_runtime)


###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


#############
## Install ##
#############

install(PROGRAMS
  nodes/detection_node
  nodes/segmentation_node
  #node/visualization_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  msg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
