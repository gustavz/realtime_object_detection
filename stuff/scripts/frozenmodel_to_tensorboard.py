#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:45:23 2018

@author: gustav
"""
import tensorflow as tf
from tensorflow.python.platform import gfile

MODEL_NAME = 'ssd_mobilenet_v11_coco'
MODEL_FILE ='../models/' + MODEL_NAME +'/frozen_inference_graph.pb'
LOG_DIR='../models/ssd_mobilenet_v11_coco/log/'

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
