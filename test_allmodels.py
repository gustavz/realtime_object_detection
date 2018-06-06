#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:45:23 2018

@author: www.github.com/GustavZ
"""

import os
import sys
import numpy as np
from test_objectdetection import detection
from test_deeplab import segmentation
from rod.config import Config
from rod.model import Model
from time import sleep

def create_test_config(model, type='OD', optimized=False):
        class TestConfig(Config):
            LIMIT_IMAGES = 100
            SPLIT_MODEL = False
            WRITE_TIMELINE = True
            if type is 'DL':
                DL_MODEL_PATH=model+'/{}'
            else:
                OD_MODEL_PATH=model+'/{}'
            if optimized:
                USE_OPTIMIZED=True
        return TestConfig()

# Read sequentail Models or Gather all Models from models/
CONFIG = Config()
if CONFIG.SEQ_MODELS:
    models = CONFIG.SEQ_MODELS
else:
    MODELS_DIR = os.path.join(os.getcwd(),'models')
    for root, dirs, files in os.walk(MODELS_DIR):
        if root.count(os.sep) - MODELS_DIR.count(os.sep) == 0:
            for idx,model in enumerate(dirs):
                models=[]
                models.append(dirs)
                models = np.squeeze(models)
                models.sort()
print("> start testing following sequention of models: \n{}".format(models))
for mod in models:
    print("> testing model: {}".format(mod))
    MODEL_DIR =  os.path.join(os.getcwd(),'models',mod)
    optimized=False
    # Check if there is an optimized graph
    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            if 'optimized' in file:
                optimized=True
                print '> found: optimized graph'
    # Test Model
    if 'deeplab' in mod:
        config = create_test_config(MODEL_DIR,'DL')
        model = Model('dl', config.DL_MODEL_NAME, config.DL_MODEL_PATH).prepare_dl_model()
        segmentation(model,config)
    else:
        config = create_test_config(MODEL_DIR)
        model = Model('od', config.OD_MODEL_NAME, config.OD_MODEL_PATH, config.LABEL_PATH,
                    config.NUM_CLASSES, config.SPLIT_MODEL, config.SSD_SHAPE).prepare_od_model()
        detection(model,config)
    # Again for the optimized graph
    if optimized:
        if 'deeplab' in mod:
            config = create_test_config(MODEL_DIR,'DL',optimized)
            model = Model('dl', config.DL_MODEL_NAME, config.DL_MODEL_PATH).prepare_dl_model()
            segmentation(model,config)
        else:
            config = create_test_config(MODEL_DIR,'OD',optimized)
            model = Model('od', config.OD_MODEL_NAME, config.OD_MODEL_PATH, config.LABEL_PATH,
                        config.NUM_CLASSES, config.SPLIT_MODEL, config.SSD_SHAPE).prepare_od_model()
            detection(model,config)
    #sleep(30)
