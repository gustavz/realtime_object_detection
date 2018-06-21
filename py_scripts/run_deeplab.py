#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:17:38 2018

@author: github.com/GustavZ
"""
import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from skimage import measure

# Root directory of the project
ROOT_DIR = os.path.abspath("../src/realtime_object_detection")
sys.path.append(ROOT_DIR)

from vis_utils import draw_single_box_on_image, visualize_deeplab
from helper import FPS, WebcamVideoStream
from model import Model
from config import Config


def segmentation(model,config):
    detection_graph = model.detection_graph
    # fixed input sizes as model needs resize either way
    vs = WebcamVideoStream(config.CV_INPUT,640,480).start()
    resize_ratio = 1.0 * 513 / max(vs.real_width,vs.real_height)
    target_size = (int(resize_ratio * vs.real_width), int(resize_ratio * vs.real_height)) #(513, 384)
    tf_config = model.tf_config
    fps = FPS(config.FPS_INTERVAL).start()
    print("> Starting Segmentaion")
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph,config=tf_config) as sess:
            while vs.isActive():
                frame = vs.resized(target_size)
                batch_seg_map = sess.run('SemanticPredictions:0',
                                        feed_dict={'ImageTensor:0':
                                        [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)]})
                seg_map = batch_seg_map[0]
                #boxes = []
                #labels = []
                map_labeled = measure.label(seg_map, connectivity=1)
                for region in measure.regionprops(map_labeled):
                    if region.area > config.MINAREA:
                        box = region.bbox
                        label = config.LABEL_NAMES[seg_map[tuple(region.coords[0])]]
                        #boxes.append(box)
                        #labels.append(label)
                        if config.VISUALIZE:
                            draw_single_box_on_image(frame,box,label)

                vis = visualize_deeplab(frame,seg_map,fps._glob_numFrames,config.MAX_FRAMES,fps.fps_local(),
                                        config.PRINT_INTERVAL,config.PRINT_TH,config.OD_MODEL_NAME+config._DEV+config._OPT,config.VISUALIZE)
                if not vis:
                    break
                fps.update()
    fps.stop()
    vs.stop()


if __name__ == '__main__':
    config = Config()
    model = Model('dl', config.DL_MODEL_NAME, config.DL_MODEL_PATH).prepare_dl_model()
    segmentation(model,config)
