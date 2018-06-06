#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:45:23 2018

@author: www.github.com/GustavZ
"""

from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary

import os
import sys
import numpy as np
ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR,'models')

# Gather all Model Names in models/
for root, dirs, files in os.walk(MODEL_DIR):
    if root.count(os.sep) - MODEL_DIR.count(os.sep) == 0:
        for idx,model in enumerate(dirs):
            models=[]
            models.append(dirs)
            models = np.squeeze(models)

# Create Tensorboard readable tfevent files in models/{}/log
for model in models:
    print("> creating tfevent of model: {}".format(model))
    MODEL_NAME=model
    MODEL_PATH=ROOT_DIR+'/models/{}/frozen_inference_graph.pb'.format(MODEL_NAME)
    LOG_DIR=ROOT_DIR+'/models/{}/log/'.format(MODEL_NAME)

    with session.Session(graph=ops.Graph()) as sess:
        with gfile.FastGFile(MODEL_PATH, "rb") as f:
          graph_def = graph_pb2.GraphDef()
          graph_def.ParseFromString(f.read())
          importer.import_graph_def(graph_def)
        pb_visual_writer = summary.FileWriter(LOG_DIR)
        pb_visual_writer.add_graph(sess.graph)
    print("> Model Imported. Visualize by running: "
    "tensorboard --logdir={}".format(LOG_DIR))
