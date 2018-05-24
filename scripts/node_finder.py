#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:58:42 2018

@author: www.github.com/GustavZ
"""

import tensorflow as tf
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
NODE_OPS = ['Placeholder','Identity']
MODEL_FILE = '../models/{}/frozen_inference_graph.pb'.format(MODEL_NAME)

gf = tf.GraphDef()
gf.ParseFromString(open(MODEL_FILE,'rb').read())

print([n.name + '=>' +  n.op for n in gf.node if n.op in (NODE_OPS)])
