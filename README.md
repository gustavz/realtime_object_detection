# realtime_object_detection
Realtime Object Detection based on Tensorflow's [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [DeepLab Project](https://github.com/tensorflow/models/tree/master/research/deeplab) <br />
> Release Note: use v1.0 for the original repo that was focused on high performance inference of `ssd_mobilenet` <br />
(*x10 Performance Increase on Nvidia Jetson TX2*)

<img src="test_images/od_demo.gif" width="33.3%">

> Release Note: use Master or v2.0 to be additionally able to run and test Mask-Detection Models, KCF-Tracking and DeepLab Models (*merge of this [project](https://github.com/GustavZ/realtime_segmenation)*)

<img src="test_images/dl_demo.gif" width="33.3%">

## About the Project
The Idea was to create a scaleable realtime-capable object detection pipeline that runs on various systems. <br />
Plug and play, ready to use without deep previous knowledge.<br /> <br />
The project includes following work:
- optionally download tensorflow pretrained models
- do Inference with OpenCV, either through video input or on selected test_images. <br />
supported Models are all `research/object_detection` as well as `research/deeplab models`
- enjoy this project's own `ssd_mobilenet` speed hack, which splits the model in a mutlithreaded cpu and gpu session. <br />
Results in up to x10 performance increase depending on the running system <br />
⇒ which makes it (one of) the fastest inference piplines out there
- do benchmark tests on sets of images and get statistical information like mean and median fps, std dev and much more

## Getting Started:  
- create a copy of `config.sample.yml` called `config.yml`
- optional: Change Parameters in `config.yml` to load other models or to modify configurations.<br />
For example: If you are not interested in visualization: set `visualize` to `False`, <br />
or if you want to switch off the speed hack set `split_model` to `False`, <br />
or to use KCF-Tracking set `use_tracker` to `true` (currently only works for pure object detection models without `split_model`)
- for realtime inference using video stream run: `run_objectdetection.py` or `run_deeplab.py`
- for benchmark tests on sample images run: `test_objectdetection.py`or `test_deeplab.py` <br />
(put them as `.jpg`  into `test_images/`)
- Enjoy deeplearning!

## My Setup:
Use the following setup for best and verified performance
- Ubuntu 16.04
- Python 2.7
- Tensorflow 1.4
([this repo](https://github.com/peterlee0127/tensorflow-nvJetson) provides pre-build tf wheel files for jetson tx2)
- OpenCV 3.3.1
> Note: tensorflow v1.7.0 seems to have massive performance issues (try to use other versions)

## Current max Performance on `ssd_mobilenet` (with|without visualization):
- Dell XPS 15 with i7 @ 2.80GHZ x8 and GeForce GTX 1050 4GB:  **78fps | 105fps**
- Nvidia Jetson Tx2 with Tegra 8GB:                           **30fps | 33 fps**

## To Do:
If you like the project, got improvement or constructive critisism, please feel free to open an Issue. <br />
I am always happy to get feedback or help to be able to further improve the project. <br />
Future implementation plans are: <br />
- [X] Add KCF Tracking to improve fps especially on the jetson
- [ ] ~~Mask-SSD: Modify SSD to be able to predict a segmentation mask in parallel to the bounding box~~
- [ ] Train a `mask_rcnn Model` with Mobilenet V1/V2 as backbone and deploy it on the Jetson
- [ ] Split Model and Threading for R-CNN Models

## Related Work:
- [test_models](https://github.com/GustavZ/test_models): A repo for models i am currently working on for benchmark tests
- [deeptraining_hands](https://github.com/GustavZ/deeptraining_hands): A repo for setting up the [ego](http://vision.soic.indiana.edu/projects/egohands/)- and [oxford](http://www.robots.ox.ac.uk/~vgg/data/hands/) hands-datasets.<br />
It also contains several scripts to convert various annotation formats to be able to train Networks on different deep learning frameworks <br />
currently supports `.xml`, `.mat`, `.csv`, `.record`, `.txt` annotations
- [yolo_for_tf_od_api](https://github.com/GustavZ/yolo_for_tf_od_api): A repo to be able to include Yolo V2 in tf's object detection api
- [realtime_segmenation](https://github.com/GustavZ/realtime_segmenation): This repo was merged into v2.0
- [Mobile_Mask_RCNN](https://github.com/GustavZ/Mobile_Mask_RCNN): a Keras Model for training Mask R-CNN for mobile deployment
- [tf_training](https://github.com/GustavZ/tf_training): Train Mobile Mask R-CNN Models on AWS Cloud
- [tf_models](https://github.com/GustavZ/tf_models): My `tensorflow/models` fork which includes `yolov2` and `mask_rcnn_mobilenet_v1_coco`
