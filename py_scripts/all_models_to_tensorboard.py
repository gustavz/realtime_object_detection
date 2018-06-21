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

# Root directory of the project
ROOT_DIR = os.path.abspath("../src/realtime_object_detection")
sys.path.append(ROOT_DIR)

def create_tfevent_from_pb(model,optimized=False):
    print("> creating tfevent of model: {}".format(model))

    if optimized:
        model_path=ROOT_DIR+'/models/{}/optimized_inference_graph.pb'.format(model)
        log_dir=ROOT_DIR+'/models/{}/log_opt/'.format(model)
    else:
        model_path=ROOT_DIR+'/models/{}/frozen_inference_graph.pb'.format(model)
        log_dir=ROOT_DIR+'/models/{}/log/'.format(model)

    with session.Session(graph=ops.Graph()) as sess:
        with gfile.FastGFile(model_path, "rb") as f:
          graph_def = graph_pb2.GraphDef()
          graph_def.ParseFromString(f.read())
          importer.import_graph_def(graph_def)
        pb_visual_writer = summary.FileWriter(log_dir)
        pb_visual_writer.add_graph(sess.graph)
    print("> Model {} Imported. \nVisualize by running: \
    tensorboard --logdir={}".format(model_path, log_dir))

# Gather all Model Names in models/
MODELS_DIR = os.path.join(ROOT_DIR,'models')
for root, dirs, files in os.walk(MODELS_DIR):
    if root.count(os.sep) - MODELS_DIR.count(os.sep) == 0:
        for idx,model in enumerate(dirs):
            models=[]
            models.append(dirs)
            models = np.squeeze(models)
            models.sort()

# Create Tensorboard readable tfevent files in models/{}/log
for model in models:
    optimized=False
    create_tfevent_from_pb(model,optimized)
    # Check if there is an optimized graph
    model_dir =  os.path.join(MODELS_DIR,model)
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if 'optimized' in file:
                optimized=True
                print '> found: optimized graph'
    create_tfevent_from_pb(model,optimized)
