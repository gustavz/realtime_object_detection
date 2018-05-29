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
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from rod.config import Config

MODEL = 'OD' # Change to 'OD' or 'DL'
NODE_OPS = ['Placeholder','Identity','CheckNumerics']

## Don't Change ##
config = Config()
if MODEL == 'OD':
    MODEL_PATH = '../'+config.OD_MODEL_PATH
elif MODEL == 'DL':
    MODEL_PATH = '../'+config.DL_MODEL_PATH

gf = tf.GraphDef()
gf.ParseFromString(open(MODEL_PATH,'rb').read())

print([n.name + '=>' +  n.op for n in gf.node if n.op in (NODE_OPS)])
