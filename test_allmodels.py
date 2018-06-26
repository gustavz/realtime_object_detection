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

ROOT_DIR = os.getcwd()
MODELS_DIR = os.path.join(ROOT_DIR,'models')

def create_test_config(type,model_name, optimized=False, single_class=False):
        class TestConfig(Config):
            MODEL_PATH='models/'+model_name+'/{}'
            def __init__(self):
                super(TestConfig, self).__init__(type)
                self.SPLIT_MODEL = False
                self.WRITE_TIMELINE = True
                self.MODEL_NAME=model_name
                if optimized:
                    self.USE_OPTIMIZED=True
                else:
                    self.USE_OPTIMIZED=False
                if single_class:
                    self.NUM_CLASSES=1
                else:
                    self.NUM_CLASSES=90

        return TestConfig()

# Read sequentail Models or Gather all Models from models/
config = Config('od')
if config.SEQ_MODELS:
    model_names = config.SEQ_MODELS
else:
    for root, dirs, files in os.walk(MODELS_DIR):
        if root.count(os.sep) - MODELS_DIR.count(os.sep) == 0:
            for idx,model in enumerate(dirs):
                model_names=[]
                models_names.append(dirs)
                model_names = np.squeeze(model_names)
                model_names.sort()

print("> start testing following sequention of models: \n{}".format(model_names))
for model_name in model_names:
    print("> testing model: {}".format(model_name))
    # conditionals
    optimized=False
    single_class=False
    # Test Model
    if 'hands' in model_name or 'person' in model_name:
        single_class=True
    if 'deeplab' in model_name:
        config = create_test_config('dl',model_name,optimized,single_class)
        model = Model(config).prepare_model()
        segmentation(model)
    else:
        config = create_test_config('od',model_name,optimized,single_class)
        model = Model(config).prepare_model()
        detection(model)

    # Check if there is an optimized graph
    model_dir =  os.path.join(os.getcwd(),'models',model_name)
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if 'optimized' in file:
                optimized=True
                print '> found: optimized graph'

    # Again for the optimized graph
    if optimized:
        if 'deeplab' in model_name:
            config = create_test_config('dl',model_name,optimized,single_class)
            model = Model(config).prepare_model()
            segmentation(model)
        else:
            config = create_test_config('dl',model_name,optimized,single_class)
            model = Model(config).prepare_od_model()
            detection(model)
