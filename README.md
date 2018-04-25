# realtime_object_detection
Realtime object detection based on [Tensorflow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) with an extreme Focus on Performance. <br />
Specialized for `ssd_mobilenet` models.

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
- Added `split_model` Option to split frozen graph into a GPU and CPU session (thanks to [wkelongws](https://github.com/wkelongws)). <br />
Works only for `ssd_mobilenet` models but results in significant performance increase. 
- Added mutlithreading for the split sessions (thanks to [naisy](https://github.com/naisy))
- **Results: Overall up to x10 Performance Increase** depending on the config and the running system

## Getting Started:  
- create a copy of `config.sample.yml` called `config.yml`
- Optional: Change Parameters in `config.yml` to load other models or to modify input params.
- For example: If you are not interested in visualization: set `visualize` to `False`. <br />
- run `image_detection.py` for single test image detection
- run `object_detection.py` for realtime object detection
- Enjoy!

## Under Development (help appreciated):
- **KCF Tracking**: run `./build_kcf.sh` inside directory, set `use_tracker` to `True` inside config, run `object_detection_kcf_test.py` (currently only works more or less stable without `split_model`)
- **Mask Detection**: run `object_detection_mask_test.py` (currently only works for `mask r-cnn` models and `TF 1.5`, so also no `split_model`)
<br />

## My Setup:
- Ubuntu 16.04
- Python 2.7
- Tensorflow 1.4 
([this repo](https://github.com/peterlee0127/tensorflow-nvJetson) provides pre-build tf wheel files for jetson tx2)
- OpenCV 3.3.1
> Note: This project currently does not run with tensorflow v1.7.0

## Current max Performance on `ssd_mobilenet` (with|without visualization):
- Dell XPS 15 with i7 @ 2.80GHZ x8 and GeForce GTX 1050 4GB:  **78fps | 105fps**
- Nvidia Jetson Tx2 with Tegra 8GB:                           **30fps | 33 fps**

## To Do:
If you like the project, got improvement or constructive critisism, please feel free to open an Issue. <br />
I am always happy to get feedback or help to be able to further improve the project. <br />
Future implementation plans are: <br />
- [ ] Add KCF Tracking to improve fps especially on the jetson
- [ ] Mask-SSD: Modify SSD to be able to predict a segmentation mask in parallel to the bounding box
- [ ] Split Model and Threading for R-CNN Models
 
## Related Work:
- [test_models](https://github.com/GustavZ/test_models): A repo for models i am currently working on for benchmark tests
- [deeptraining_hands](https://github.com/GustavZ/deeptraining_hands): A repo for setting up the [ego](http://vision.soic.indiana.edu/projects/egohands/)- and [oxford](http://www.robots.ox.ac.uk/~vgg/data/hands/) hands-datasets.<br />
It also contains several scripts to convert various annotation formats to be able to train Networks on different deep learning frameworks <br />
currently supports `.xml`, `.mat`, `.csv`, `.record`, `.txt` annotations
- [yolo_for_tf_od_api](https://github.com/GustavZ/yolo_for_tf_od_api): A repo to be able to include Yolo V2 in tf's object detection api
