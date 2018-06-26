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

from tensorflow.python.client import timeline
from rod.helper import TimeLiner, Timer, load_images
from rod.vis_utils import draw_single_box_on_image, visualize_deeplab
from rod.model import Model
from rod.config import Config


def segmentation(model,config):
    images = load_images(config.IMAGE_PATH,config.LIMIT_IMAGES)
    # Tf Session + Timeliner
    tf_config = model.tf_config
    detection_graph = model.detection_graph
    if config.WRITE_TIMELINE:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        timeliner = TimeLiner()
    else:
        options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        run_metadata = False
    timer = Timer().start()
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
                if config.WRITE_TIMELINE:
                    timeliner.write_timeline(run_metadata.step_stats,
                                            '{}/timeline_{}.json'.format(
                                            config.RESULT_PATH,config.DL_DISPLAY_NAME))
                seg_map = batch_seg_map[0]
                #boxes = []
                #labels = []
                #ids = []
                map_labeled = measure.label(seg_map, connectivity=1)
                for region in measure.regionprops(map_labeled):
                    if region.area > config.MINAREA:
                        box = region.bbox
                        id = seg_map[tuple(region.coords[0])]
                        label = config.LABEL_NAMES[id]
                        #boxes.append(box)
                        #labels.append(label)
                        #ids.append(id)
                        if config.VISUALIZE:
                            draw_single_box_on_image(frame,box,label,id,config.DISCO_MODE)

                vis = visualize_deeplab(frame,seg_map,timer.get_frame(),config.MAX_FRAMES,timer.get_fps(),
                                        config.PRINT_INTERVAL,config.PRINT_TH,config.DL_DISPLAY_NAME,
                                        config.VISUALIZE,config.VIS_FPS,config.DISCO_MODE,config.ALPHA)
                if not vis:
                    break
                if config.SAVE_RESULT:
                    cv2.imwrite('{}/{}_{}.jpg'.format(config.RESULT_PATH,timer.get_frame(),config.DL_DISPLAY_NAME),frame)

	cv2.destroyAllWindows()
    timer.stop()


if __name__ == '__main__':
    config = Config()
    model = Model('dl', config.DL_MODEL_NAME, config.DL_MODEL_PATH).prepare_dl_model()
    segmentation(model,config)
