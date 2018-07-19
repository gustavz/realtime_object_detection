#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:45:23 2018

@author: www.github.com/GustavZ
"""

import os
import sys
import numpy as np

from rod.config import Config
from rod.helper import get_model_list, check_if_optimized_model
from rod.model import ObjectDetectionModel, DeepLabModel

ROOT_DIR = os.getcwd()
#MODELS_DIR = os.path.join(ROOT_DIR,'models')
MODELS_DIR = '/home/gustav/workspace/eetfm_automation/nmsspeed_test'
INPUT_TYPE = 'image'

def create_test_config(type,model_name, optimized=False, single_class=False):
        class TestConfig(Config):
            OD_MODEL_PATH=MODELS_DIR+'/'+model_name+'/{}'
            DL_MODEL_PATH=MODELS_DIR+'/'+model_name+'/{}'
            OD_MODEL_NAME=model_name
            DL_MODEL_NAME=model_name
            VISUALIZE=False
            SPLIT_MODEL = False
            WRITE_TIMELINE = True
            LIMIT_IMAGES = 11
            if optimized:
                USE_OPTIMIZED=True
            else:
                USE_OPTIMIZED=False
            if single_class:
                NUM_CLASSES=1
            else:
                NUM_CLASSES=90
            def __init__(self):
                super(TestConfig, self).__init__(type)
        return TestConfig()

# Read sequentail Models or Gather all Models from models/
config = Config('od')
if config.SEQ_MODELS:
    model_names = config.SEQ_MODELS
else:
    model_names = get_model_list(MODELS_DIR)

# Sequential testing
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
        model = DeepLabModel(config).prepare_model(INPUT_TYPE)
    else:
        config = create_test_config('od',model_name,optimized,single_class)
        model = ObjectDetectionModel(config).prepare_model(INPUT_TYPE)

    # Check if there is an optimized graph
    model_dir =  os.path.join(os.getcwd(),'models',model_name)
    optimized = check_if_optimized_model(model_dir)

    # Again for the optimized graph
    if optimized:
        if 'deeplab' in model_name:
            config = create_test_config('dl',model_name,optimized,single_class)
            model = DeepLabModel(config).prepare_model(INPUT_TYPE)
        else:
            config = create_test_config('od',model_name,optimized,single_class)
            model = ObjectDetectionModel(config).prepare_model(INPUT_TYPE)

    model.run()
