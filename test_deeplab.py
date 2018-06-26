#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:17:38 2018

@author: github.com/GustavZ
"""
import os
import numpy as np
import tensorflow as tf
import cv2
from skimage import measure

from rod.helper import TimeLiner, Timer, load_images
from rod.visualizer import Visualizer
from rod.config import Config
from rod.model import Model


def segmentation(model):
    images = load_images(model.config.IMAGE_PATH,model.config.LIMIT_IMAGES)
    # Tf Session + Timeliner
    tf_config = model.tf_config
    detection_graph = model.detection_graph
    if model.config.WRITE_TIMELINE:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        timeliner = TimeLiner()
    else:
        options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        run_metadata = False
    timer = Timer()
    visualizer = Visualizer(model.config.TYPE)
    print("> Starting Segmentaion")
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=tf_config) as sess:
            for image in images:
                # input
                frame = cv2.imread(image)
                height, width, channels = frame.shape
                resize_ratio = 1.0 * 513 / max(width,height)
                target_size = (int(resize_ratio * width), int(resize_ratio * height))
                frame = cv2.resize(frame, target_size)
                timer.tic()
                batch_seg_map = sess.run('SemanticPredictions:0',
                				feed_dict={'ImageTensor:0': [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)]},
                				options=options, run_metadata=run_metadata)
                timer.toc()
                if model.config.WRITE_TIMELINE:
                    timeliner.write_timeline(run_metadata.step_stats,
                                            '{}/timeline_{}.json'.format(
                                            model.config.RESULT_PATH,model.config.DISPLAY_NAME))
                seg_map = batch_seg_map[0]
                boxes = []
                labels = []
                ids = []
                map_labeled = measure.label(seg_map, connectivity=1)
                for region in measure.regionprops(map_labeled):
                    if region.area > model.config.MINAREA:
                        box = region.bbox
                        id = seg_map[tuple(region.coords[0])]
                        label = model.config.LABEL_NAMES[id]
                        boxes.append(box)
                        labels.append(label)
                        ids.append(id)
                visualizer.visualize_deeplab(frame,boxes,labels,ids,seg_map,timer.get_frame(),timer.get_fps())
                if model.config.SAVE_RESULT:
                    cv2.imwrite('{}/{}_{}.jpg'.format(model.config.RESULT_PATH,timer.get_frame(),model.config.DISPLAY_NAME),frame)

	visualizer.stop()
    timer.stop()


if __name__ == '__main__':
    model_type = 'dl'
    config = Config()
    model = Model(config).prepare_model()
    segmentation(model)
