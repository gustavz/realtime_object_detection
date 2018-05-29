#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: www.github.com/GustavZ
"""
import numpy as np

class Config(object):
    """
    Inference Configuration class
    Replaces 'config.sample.yml' of v1.0
    """
    ### Inference Config
    VIDEO_INPUT = 0                 # Input Must be OpenCV readable
    VISUALIZE = True                # Disable for performance increase


    ### Testing
    IMAGE_PATH = 'test_images'      # path for test.py test_images
    LIMIT_IMAGES = None             # if set to None, all images are used
    CPU_ONLY = False                # CPU Placement for speed test
    WRITE_TIMELINE = False           # write json timeline file (slows infrence)


    ### Object_Detection
    WIDTH = 600                     # OpenCV only supports 4:3 formats others will be converted
    HEIGHT = 600                    # 600x600 leads to 640x480
    MAX_FRAMES = 5000               # only used if visualize==False
    FPS_INTERVAL = 5                # Interval [s] to print fps of the last interval in console
    DET_INTERVAL = 500              # intervall [frames] to print detections to console
    DET_TH = 0.5                    # detection threshold for det_intervall
    ## speed hack
    SPLIT_MODEL = True              # Splits Model into a GPU and CPU session (currently only works for ssd_mobilenets)
    SSD_SHAPE = 300                 # used for the split model algorithm (currently only supports ssd networks trained on 300x300 and 600x600 input)
    ## Tracking
    USE_TRACKER = False             # Use a Tracker (currently only works properly WITHOUT split_model)
    TRACKER_FRAMES = 20             # Number of tracked frames between detections
    NUM_TRACKERS = 5                # Max number of objects to track
    ## Model
    OD_MODEL_NAME = 'ssd_mobilenet_v11_coco'
    OD_MODEL_PATH = 'models/ssd_mobilenet_v11_coco/frozen_inference_graph.pb'
    LABEL_PATH = 'rod/data/mscoco_label_map.pbtxt'
    NUM_CLASSES = 90


    ### DeepLab
    ALPHA = 0.3                     # mask overlay factor
    BBOX = True                     # compute boundingbox in postprocessing
    MINAREA = 500                   # min Pixel Area to apply bounding boxes (avoid noise)
    ## Model
    DL_MODEL_NAME = 'deeplabv3_mnv2_pascal_train_aug_2018_01_29'
    DL_MODEL_PATH = 'models/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb'
    LABEL_NAMES = np.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'])


    def __init__(self):
        ## TimeLine File naming
        if self.CPU_ONLY:
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.DEVICE = '_CPU'
        else:
            self.DEVICE = ''
