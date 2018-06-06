#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:51:59 2017

@author: GustavZ
"""

import numpy as np
import os
import tensorflow as tf
import cv2
import datetime
from rod.helper import Timer, WebcamVideoStream, SessionWorker, vis_detection, TimeLiner, load_images
from rod.model import Model
from rod.config import Config
from rod.utils import ops as utils_ops


def detection(model,config):
    # Tf Session
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth=True
    detection_graph = model.detection_graph
    category_index = model.category_index
    print("> Building Graph")
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph,config=tf_config) as sess:
            # start Videostream
            # Define Input and Ouput tensors
            tensor_dict = model.get_tensordict(['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks'])
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Mask Transformations
            if 'detection_masks' in tensor_dict:
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, config.HEIGHT, config.WIDTH)
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            if config.SPLIT_MODEL:
                score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
                expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
                # Threading
                score = model.score
                expand = model.expand

            # Timeliner
            if config.WRITE_TIMELINE:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                timeliner = TimeLiner()
            else:
                options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                run_metadata = False

            images = load_images(config.IMAGE_PATH,config.LIMIT_IMAGES)
            timer = Timer().start()
            print('> Starting Detection')
            for image in images:
                if config.SPLIT_MODEL:
                    # split model in seperate gpu and cpu session threads
                    masks = None # No Mask Detection possible yet
                    frame = cv2.resize(cv2.imread(image),(config.WIDTH,config.HEIGHT))
                    frame_expanded = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0)
                    timer.tic()
                    # GPU Session
                    score, expand = sess.run([score_out, expand_out],
                            feed_dict={image_tensor: frame_expanded},
                            options=options, run_metadata=run_metadata)
                    timer.tictic()
                    if config.WRITE_TIMELINE:
                        timeliner.write_timeline(run_metadata.step_stats,
                                                'test_results/timeline_{}{}{}{}.json'.format(
                                                config.OD_MODEL_NAME,'_SM1',config._DEV,config._OPT))
                    timer.tic()
                    # CPU Session
                    boxes, scores, classes, num = sess.run(
                            [tensor_dict['detection_boxes'], tensor_dict['detection_scores'], tensor_dict['detection_classes'], tensor_dict['num_detections']],
                            feed_dict={score_in:score, expand_in: expand},
                            options=options, run_metadata=run_metadata)
                    timer.toc()
                    if config.WRITE_TIMELINE:
                        timeliner.write_timeline(run_metadata.step_stats,
                                                'test_results/timeline_{}{}{}{}.json'.format(
                                                config.OD_MODEL_NAME,'_SM2',config._DEV,config._OPT))
                else:
                    # default session
                    frame = cv2.resize(cv2.imread(image),(config.WIDTH,config.HEIGHT))
                    frame_expanded = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0)
                    timer.tic()
                    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: frame_expanded},
                            options=options, run_metadata=run_metadata)
                    timer.toc()
                    if config.WRITE_TIMELINE:
                        timeliner.write_timeline(run_metadata.step_stats,
                                                'test_results/timeline_{}{}{}.json'.format(
                                                config.OD_MODEL_NAME,config._DEV,config._OPT))
                    num = output_dict['num_detections'][0]
                    classes = output_dict['detection_classes'][0]
                    boxes = output_dict['detection_boxes'][0]
                    scores = output_dict['detection_scores'][0]
                    if 'detection_masks' in output_dict:
                        masks = output_dict['detection_masks'][0]
                    else:
                        masks = None

                # reformat detection
                num = int(num)
                boxes = np.squeeze(boxes)
                classes = np.squeeze(classes).astype(np.uint8)
                scores = np.squeeze(scores)

                # Visualization
                vis = vis_detection(frame, boxes, classes, scores, masks, category_index, timer.get_fps(),
                                    config.VISUALIZE, config.DET_INTERVAL, config.DET_TH, config.MAX_FRAMES, None, config.OD_MODEL_NAME+config._OPT)
                if not vis:
                    break

    cv2.destroyAllWindows()
    timer.stop()


if __name__ == '__main__':
    config = Config()
    config.display()
    model = Model('od', config.OD_MODEL_NAME, config.OD_MODEL_PATH, config.LABEL_PATH,
                config.NUM_CLASSES, config.SPLIT_MODEL, config.SSD_SHAPE).prepare_od_model()
    detection(model,config)
