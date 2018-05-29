#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:45:23 2018

@author: www.github.com/GustavZ
"""

import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from rod.config import Config

## Change to 'OD' or 'DL'
MODEL = 'OD'

## Don't Change ##
config = Config()
if MODEL == 'OD':
    MODEL_NAME = config.OD_MODEL_NAME
    MODEL_PATH = '../'+config.OD_MODEL_PATH
elif MODEL == 'DL':
    MODEL_NAME = config.DL_MODEL_NAME
    MODEL_PATH = '../'+config.DL_MODEL_PATH

LOG_DIR='../models/{}/log/'.format(MODEL_NAME)

with tf.Session() as sess:
    with gfile.FastGFile(MODEL_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
train_writer = tf.summary.FileWriter(LOG_DIR)
train_writer.add_graph(sess.graph)

print("Model Imported. Run the command line:\n" \
          "--> tensorboard --logdir={}".format(LOG_DIR))
print( "Then open http://0.0.0.0:6006/ into your web browser")
