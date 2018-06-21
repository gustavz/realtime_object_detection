# ROS Package for github/gustav/realtime_object_detection

## Getting started:
- clone this repo into your catkin workspace [catkin_ws/src/]
- build your workspace: `catkin build`
- `source devel/setup.bash`
- create copy of `config.sample.yml` named `config.sample` and change parameter according to your needs
- start ros: `roscore`
- start camera node: `rosrun usb_cam usb_cam_node` or `roslaunch realsense2_camera rs_camera.launch`
- start detection_node: `rosrun objdetection detection_node`
- witness greatness!

## Using Scripts:
- edit VIDEO_INPUT variable to an openCV readable path: like `0` for `/dev/video0`
- run the py_scripts inside the directory like: python `run_objectdetection.py`
- to run the sh_scripts first change parameters inside `config_tools.sh` and the `source` it to export environment variables. After that run the scripts in the same terminal like: `source summarize_graph.sh`

[Link Base Repository](https://github.com/GustavZ/realtime_object_detection)
