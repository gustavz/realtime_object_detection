#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:58:42 2018

@author: www.github.com/GustavZ
"""

import tensorflow as tf
import yaml
import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../src/realtime_object_detection")
sys.path.append(ROOT_DIR)

from config import Config


MODEL = 'OD' # Change to 'OD' or 'DL'
NODE_NAMES = ['Momentum','Optimizer','BatchNorm', 'Loss']
NODE_OPS = ['Placeholder','Identity','CheckNumerics','BatchNorm']

## Don't Change ##
config = Config()
if MODEL == 'OD':
    MODEL_PATH = config.OD_MODEL_PATH
elif MODEL == 'DL':
    MODEL_PATH = config.DL_MODEL_PATH

print("> exploring Model: {}".format(MODEL_PATH))

gf = tf.GraphDef()
gf.ParseFromString(open(MODEL_PATH,'rb').read())

print('>Looking for node Names:')
for NAME in NODE_NAMES:
    print(NAME)
    print([n.name + '=>' +  n.op for n in gf.node if NAME in n.name ])

print('>Looking for node Ops:')
for OP in NODE_OPS:
    print(OP)
    print([n.name + '=>' +  n.op for n in gf.node if OP in n.op])
