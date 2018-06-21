#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: www.github.com/GustavZ
"""

import numpy as np
import yaml
import os
import sys

try:
    import rospkg
    rospack = rospkg.RosPack()
    rospack_me = rospack.get_path('objdetection')
except:
    rospack_me = os.path.abspath("../")

## LOAD CONFIG PARAMS ##
if (os.path.isfile(rospack_me + '/config/config.yml')):
    with open(rospack_me + "/config/config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
else:
    with open(rospack_me + "/config/config.sample.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

class Config(object):
    """
    Inference Configuration class
    loads Params from 'config.sample.yml'
    """
    ### Inference Config
    ROS_INPUT = cfg['ROS_INPUT']            # ROS Input Image topic
    CV_INPUT = cfg['CV_INPUT']              # OpenCV Readable Input for py-scripts
    VISUALIZE = cfg['VISUALIZE']            # Disable for performance increase
    CPU_ONLY = cfg['CPU_ONLY']              # CPU Placement for speed test
    USE_OPTIMIZED = cfg['USE_OPTIMIZED']    # whether to use the optimized model (only possible if transform with script)


    ### Testing
    IMAGE_PATH = cfg['IMAGE_PATH']          # path for test.py test_images
    LIMIT_IMAGES = cfg['LIMIT_IMAGES']      # if set to None, all images are used
    WRITE_TIMELINE = cfg['WRITE_TIMELINE']  # write json timeline file (slows infrence)
    SEQ_MODELS = cfg['SEQ_MODELS']        # List of Models to sequentially test (Default all Models)


    ### Object_Detection
    WIDTH = cfg['WIDTH']                    # OpenCV only supports 4:3 formats others will be converted
    HEIGHT = cfg['HEIGHT']                  # 600x600 leads to 640x480
    MAX_FRAMES = cfg['MAX_FRAMES']          # only used if visualize==False
    FPS_INTERVAL = cfg['FPS_INTERVAL']      # Interval [s] to print fps of the last interval in console
    PRINT_INTERVAL = cfg['PRINT_INTERVAL']  # intervall [frames] to print detections to console
    PRINT_TH = cfg['PRINT_TH']              # detection threshold for det_intervall
    ## speed hack
    SPLIT_MODEL = cfg['SPLIT_MODEL']        # Splits Model into a GPU and CPU session (currently only works for ssd_mobilenets)
    SSD_SHAPE = cfg['SSD_SHAPE']            # used for the split model algorithm (currently only supports ssd networks trained on 300x300 and 600x600 input)
    ## Tracking
    USE_TRACKER = cfg['USE_TRACKER']        # Use a Tracker (currently only works properly WITHOUT split_model)
    TRACKER_FRAMES = cfg['TRACKER_FRAMES']  # Number of tracked frames between detections
    NUM_TRACKERS = cfg['NUM_TRACKERS']      # Max number of objects to track
    ## Model
    OD_MODEL_NAME = cfg['OD_MODEL_NAME']    # Only used for downloading the correct Model Version
    OD_MODEL_PATH = cfg['OD_MODEL_PATH']
    LABEL_PATH = cfg['LABEL_PATH']
    NUM_CLASSES =  cfg['NUM_CLASSES']

    ### DeepLab
    ALPHA = cfg['ALPHA']                    # mask overlay factor
    BBOX = cfg['BBOX']                      # compute boundingbox in postprocessing
    MINAREA = cfg['MINAREA']                # min Pixel Area to apply bounding boxes (avoid noise)
    ## Model
    DL_MODEL_NAME = cfg['DL_MODEL_NAME']    # Only used for downloading the correct Model Version
    DL_MODEL_PATH = cfg['DL_MODEL_PATH']
    LABEL_NAMES = np.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'])


    def __init__(self):
        ## Set correct ROS Package Paths
        self.LABEL_PATH = rospack_me + '/src/realtime_object_detection/data/' + self.LABEL_PATH
        self.OD_MODEL_PATH = rospack_me + '/src/realtime_object_detection/models/' + self.OD_MODEL_PATH
        self.DL_MODEL_PATH = rospack_me + '/src/realtime_object_detection/models/' + self.DL_MODEL_PATH
        ## CPU Placement and Timeline naming
        if self.CPU_ONLY:
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self._DEV = '_CPU'
        else:
            self._DEV = ''
        ## Loading Standard or Optimized Model
        if self.USE_OPTIMIZED:
            self.OD_MODEL_PATH = self.OD_MODEL_PATH.format("optimized_inference_graph.pb")
            self.DL_MODEL_PATH = self.DL_MODEL_PATH.format("optimized_inference_graph.pb")
            self._OPT = '_opt'
        else:
            self.OD_MODEL_PATH = self.OD_MODEL_PATH.format("frozen_inference_graph.pb")
            self.DL_MODEL_PATH = self.DL_MODEL_PATH.format("frozen_inference_graph.pb")
            self._OPT = ''


    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
