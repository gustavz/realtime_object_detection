#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:58:42 2018

@author: gustav
"""

import tensorflow as tf

NODE_OPS = ['Placeholder','Identity']
MODEL_FILE = '../models/ssd_mobilenet_v11_coco/frozen_inference_graph.pb'

gf = tf.GraphDef()
gf.ParseFromString(open(MODEL_FILE,'rb').read())

print([n.name + '=>' +  n.op for n in gf.node if n.op in (NODE_OPS)])
