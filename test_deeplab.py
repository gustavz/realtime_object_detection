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
from rod.helper import TimeLiner, Timer, create_colormap, vis_text, load_images
from rod.model import Model
from rod.config import Config


def segmentation(model,config):
    images = load_images(config.IMAGE_PATH,config.LIMIT_IMAGES)
    # Tf Session + Timeliner
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth=True
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
                                            'test_results/timeline_{}{}{}.json'.format(
                                            config.OD_MODEL_NAME,config._DEV,config._OPT))
                # visualization
                if config.VISUALIZE:
                    seg_map = batch_seg_map[0]
                    seg_image = create_colormap(seg_map).astype(np.uint8)
                    cv2.addWeighted(seg_image,config.ALPHA,frame,1-config.ALPHA,0,frame)
                    vis_text(frame,"fps: {}".format(timer.get_fps()),(10,30))
                    # boxes (ymin, xmin, ymax, xmax)
                    if config.BBOX:
                        map_labeled = measure.label(seg_map, connectivity=1)
                        for region in measure.regionprops(map_labeled):
                            if region.area > config.MINAREA:
                                box = region.bbox
                                p1 = (box[1], box[0])
                                p2 = (box[3], box[2])
                                cv2.rectangle(frame, p1, p2, (77,255,9), 2)
                                vis_text(frame,config.LABEL_NAMES[seg_map[tuple(region.coords[0])]],(p1[0],p1[1]-10))
                    cv2.imshow(config.DL_MODEL_NAME+config._OPT,frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
	cv2.destroyAllWindows()
    timer.stop()


if __name__ == '__main__':
    config = Config()
    model = Model('dl', config.DL_MODEL_NAME, config.DL_MODEL_PATH).prepare_dl_model()
    segmentation(model,config)
