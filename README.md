# realtime_object_detection
My Version of [Tensorflows Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
<br />
<br />
## About the Project
The Idea was to create a realtime capable object detection pipeline on various machines. <br />
Plug and play, ready to use without deep previous knowledge.<br /> <br />
The following work has been done based on the original API:
- Capturing frames of a Camera-Input using OpenCV in seperate thread to increase performance
- Calculate, print and optionally visualize current-local and total FPS
- Allows Models to grow GPU memory allocation. *(ssd_mobilenet_v11_coco needs 350 MB)*
- Added Option for detection without visualization to increase performance
- Added optional automated model download from [model-zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) if necessary
- Added `config.yml` for quick&easy parameter parsing
- Exported new frozen Model based on `ssd_mobilenet_v1_coco` with altered `batch_non_max_suppression.score_threshold` to increase perfomance
- Added various scripts in `/stuff` to make use of tf's API
- Added `split_model` Option to split frozen graph into a GPU and CPU session. <br />
Works only for `ssd_mobilenet` models but results in significant performance increase. 
- **Results: Overall Performance Increase of up to 300%** depending on the config and the running system
<br />

## Getting Started:  
- Optional: Change Parameters in `config.yml` to load other models or to modify input params.
- For example: If you are not interested in visualization: set `visualize` to `False`. <br />
- if you want to import the pre-trained frozen Model `.pb` file to Tensorboard to visualize the Graph, <br />
run `stuff/frozenmodel_to_tensorboard.py` and follow the command line instructions <br />
(opt: change `MODEL_NAME`  inside if necessary)
- run `image_detection.py` for single test image detection
- run `object_detection.py` for realtime object detection
- Enjoy!
<br />

## My Setup:
- Ubuntu 16.04
- Python 2.7
- Tensorflow 1.4
- OpenCV 3.3.1
 <br />

## Current max Performance on `ssd_mobilenet` (with|without visualization):
- Dell XPS 15 with i7 @ 2.80GHZ x8 and GeForce GTX 1050 4GB:  **42fps | 76fps**
- Nvidia Jetson Tx2 with Tegra 8GB:                           **5fps | 14 fps**
 <br />
