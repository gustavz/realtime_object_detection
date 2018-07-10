#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: www.github.com/GustavZ
"""

import numpy as np
import yaml
import os
import sys

## LOAD CONFIG PARAMS ##
if (os.path.isfile('config.yml')):
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
else:
    with open("config.sample.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

class Config(object):
    """
    Inference Configuration class
    loads Params from 'config.sample.yml'
    """
    ### Inference Config
    VIDEO_INPUT = cfg['VIDEO_INPUT']        # Input Must be OpenCV readable
    ROS_INPUT = cfg['ROS_INPUT']            # ROS Image Topic
    VISUALIZE = cfg['VISUALIZE']            # Disable for performance increase
    VIS_FPS = cfg['VIS_FPS']                # Draw current FPS in the top left Image corner
    CPU_ONLY = cfg['CPU_ONLY']              # CPU Placement for speed test
    USE_OPTIMIZED = cfg['USE_OPTIMIZED']    # whether to use the optimized model (only possible if transform with script)
    DISCO_MODE = cfg['DISCO_MODE']          # Secret Disco Visualization Mode
    DOWNLOAD_MODEL = cfg['DOWNLOAD_MODEL']  # Only for Models available at the TF model_zoo


    ### Testing
    IMAGE_PATH = cfg['IMAGE_PATH']          # path for test.py test_images
    LIMIT_IMAGES = cfg['LIMIT_IMAGES']      # if set to None, all images are used
    WRITE_TIMELINE = cfg['WRITE_TIMELINE']  # write json timeline file (slows infrence)
    SAVE_RESULT = cfg['SAVE_RESULT']        # save detection results to disk
    RESULT_PATH = cfg['RESULT_PATH']        # path to save detection results
    SEQ_MODELS = cfg['SEQ_MODELS']          # List of Models to sequentially test (Default all Models)


    ### Object_Detection
    WIDTH = cfg['WIDTH']                    # OpenCV only supports 4:3 formats others will be converted
    HEIGHT = cfg['HEIGHT']                  # 600x600 leads to 640x480
    MAX_FRAMES = cfg['MAX_FRAMES']          # only used if visualize==False
    FPS_INTERVAL = cfg['FPS_INTERVAL']      # Interval [s] to print fps of the last interval in console
    PRINT_INTERVAL = cfg['PRINT_INTERVAL']  # intervall [frames] to print detections to console
    PRINT_TH = cfg['PRINT_TH']              # detection threshold for det_intervall
    ## speed hack
    SPLIT_MODEL = cfg['SPLIT_MODEL']        # Splits Model into a GPU and CPU session (currently only works for ssd_mobilenets)
    MULTI_THREADING = cfg['MULTI_THREADING']# Additional Split Model Speed up through multi threading
    SSD_SHAPE = cfg['SSD_SHAPE']            # used for the split model algorithm (currently only supports ssd networks trained on 300x300 and 600x600 input)
    SPLIT_NODES = cfg['SPLIT_NODES']       # hardcoded split points for ssd_mobilenet_v1
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


    def __init__(self,model_type):
        assert model_type in ['od','dl'], "only deeplab or object_detection models"
        #  model type
        self.MODEL_TYPE = model_type
        if self.MODEL_TYPE is 'od':
            self.MODEL_PATH = self.OD_MODEL_PATH
            self.MODEL_NAME = self.OD_MODEL_NAME
        elif self.MODEL_TYPE is 'dl':
            self.MODEL_PATH = self.DL_MODEL_PATH
            self.MODEL_NAME = self.DL_MODEL_NAME
        ## CPU Placement
        if self.CPU_ONLY:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self._DEV = '_CPU'
        else:
            self._DEV = ''
        ## Loading Standard or Optimized Model
        if self.USE_OPTIMIZED:
            self.MODEL_PATH = self.MODEL_PATH.format("optimized_inference_graph.pb")
            self._OPT = '_opt'
        else:
            self.MODEL_PATH = self.MODEL_PATH.format("frozen_inference_graph.pb")
            self._OPT = ''

        self.DISPLAY_NAME = self.MODEL_NAME+self._DEV+self._OPT

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
