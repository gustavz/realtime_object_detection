# Tensorflow-Object-Detection
My Version of Googles Object Detection API. Plug and Play Video Object Detection. <br />
It displays reached fps every three seconds and also the final mean value after ending the Visualizer.
<br />

## Getting Started:  
- Optional: change *INPUT PARAMS* at the beginning of *object_detection.py* (default input is *video0* with size 480x640)
- If you are not interested in Visualization and want to increase FPS: set *visualize* to False. If you do this make sure to chose a proper *max_frames* value (default is 300)
- run *object_detection.py* Script  <br />
- Enjoy!
<br />

## My Setup:
- Ubuntu 16.04
- Python 2.7
- Tensorflow 1.4
- OpenCV 3.3.1
 <br />

## Current Performance on SSD Mobilenet (with|without visualzation):
- Dell Laptop with i7 and GeForce GTX 1050: 22fps | 25fps
- Nvidia Jetson Tx2: 5fps
 <br />

## Known Issues:
- if the script won't compile correctly: just re-run it two or three timese. Seems random.
- if you somehow closed or interrupted the script/terminal before it finished successfully it might be necessary to restart python/terminal/system
- ...
