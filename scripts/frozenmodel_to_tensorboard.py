#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:45:23 2018

@author: www.github.com/GustavZ
"""

import tensorflow as tf
from tensorflow.python.platform import gfile
import yaml

## LOAD CONFIG PARAMS ##
if (os.path.isfile('config.yml')):
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
else:
    with open("config.sample.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
MODEL_NAME = cfg['od_model_name']

## Actual Script ##
MODEL_FILE ='../models/{}/frozen_inference_graph.pb'.format(MODEL_NAME)
LOG_DIR='../models/{}/log/'.format(MODEL_NAME)

with tf.Session() as sess:
    with gfile.FastGFile(MODEL_FILE, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
train_writer = tf.summary.FileWriter(LOG_DIR)
train_writer.add_graph(sess.graph)

print("Model Imported. Run the command line:\n" \
          "--> tensorboard --logdir={}".format(LOG_DIR))
print( "Then open http://0.0.0.0:6006/ into your web browser")
