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

# Root directory of the project
ROOT_DIR = os.path.abspath("../src/realtime_object_detection")
sys.path.append(ROOT_DIR)
MODELS_DIR = os.path.join(ROOT_DIR,'models')

def create_test_config(model, type='OD', optimized=False, single_class=False):
        class TestConfig(Config):
            SPLIT_MODEL = False
            WRITE_TIMELINE = True
            if type is 'DL':
                DL_MODEL_NAME=model
                DL_MODEL_PATH='models/'+model+'/{}'
            else:
                OD_MODEL_NAME=model
                OD_MODEL_PATH='models/'+model+'/{}'
            if optimized:
                USE_OPTIMIZED=True
            else:
                USE_OPTIMIZED=False
            if single_class:
                NUM_CLASSES=1
            else:
                NUM_CLASSES=90

        return TestConfig()

# Read sequentail Models or Gather all Models from models/
config = Config()
if config.SEQ_MODELS:
    models = config.SEQ_MODELS
else:
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
    # conditionals
    optimized=False
    single_class=False
    # Test Model
    if 'hands' in mod or 'person' in mod:
        single_class=True
    if 'deeplab' in mod:
        config = create_test_config(mod,'DL',optimized,single_class)
        print("TEST")
        model = Model('dl', config.DL_MODEL_NAME, config.DL_MODEL_PATH).prepare_dl_model()
        segmentation(model,config)
    else:
        config = create_test_config(mod,'OD',optimized,single_class)
        model = Model('od', config.OD_MODEL_NAME, config.OD_MODEL_PATH, config.LABEL_PATH,
                    config.NUM_CLASSES, config.SPLIT_MODEL, config.SSD_SHAPE).prepare_od_model()
        detection(model,config)

    # Check if there is an optimized graph
    model_dir =  os.path.join(os.getcwd(),'models',mod)
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if 'optimized' in file:
                optimized=True
                print '> found: optimized graph'

    # Again for the optimized graph
    if optimized:
        if 'deeplab' in mod:
            config = create_test_config(mod,'DL',optimized,single_class)
            model = Model('dl', config.DL_MODEL_NAME, config.DL_MODEL_PATH).prepare_dl_model()
            segmentation(model,config)
        else:
            config = create_test_config(mod,'OD',optimized,single_class)
            model = Model('od', config.OD_MODEL_NAME, config.OD_MODEL_PATH, config.LABEL_PATH,
                        config.NUM_CLASSES, config.SPLIT_MODEL, config.SSD_SHAPE).prepare_od_model()
            detection(model,config)
