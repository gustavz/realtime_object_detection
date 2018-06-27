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

from rod.helper import Timer, WebcamVideoStream, SessionWorker, TimeLiner, load_images
from rod.model import Model
from rod.config import Config
from rod.visualizer import Visualizer
from rod.tf_utils import reframe_box_masks_to_image_masks


def detection(model):
    # Tf Session
    tf_config = model.tf_config
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
                detection_masks_reframed = reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, model.HEIGHT, model.WIDTH)
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            if model.config.SPLIT_MODEL:
                score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
                expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
                # Threading
                #score = model.score
                #expand = model.expand

            # Timeliner
            if model.config.WRITE_TIMELINE:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                timeliner = TimeLiner()
            else:
                options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                run_metadata = False
            masks = None
            images = load_images(model.config.IMAGE_PATH,model.config.LIMIT_IMAGES)
            timer = Timer()
            visualizer = Visualizer(model.config)
            print('> Starting Detection')
            for image in images:
                if model.config.SPLIT_MODEL:
                    # split model in seperate gpu and cpu session threads
                    frame = cv2.resize(cv2.imread(image),(model.config.WIDTH,model.config.HEIGHT))
                    frame_expanded = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0)
                    timer.tic()
                    # GPU Session
                    score, expand = sess.run([score_out, expand_out],
                            feed_dict={image_tensor: frame_expanded},
                            options=options, run_metadata=run_metadata)
                    timer.tictic()
                    if model.config.WRITE_TIMELINE:
                        timeliner.write_timeline(run_metadata.step_stats,
                                                '{}/timeline_{}_SM1.json'.format(
                                                model.config.RESULT_PATH,model.config.DISPLAY_NAME))
                    timer.tic()
                    # CPU Session
                    boxes, scores, classes, num = sess.run(
                            [tensor_dict['detection_boxes'], tensor_dict['detection_scores'], tensor_dict['detection_classes'], tensor_dict['num_detections']],
                            feed_dict={score_in:score, expand_in: expand},
                            options=options, run_metadata=run_metadata)
                    timer.toc()
                    if model.config.WRITE_TIMELINE:
                        timeliner.write_timeline(run_metadata.step_stats,
                                                '{}/timeline_{}_SM2.json'.format(
                                                model.config.RESULT_PATH,model.config.DISPLAY_NAME))
                else:
                    # default session
                    frame = cv2.resize(cv2.imread(image),(model.config.WIDTH,model.config.HEIGHT))
                    frame_expanded = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0)
                    timer.tic()
                    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: frame_expanded},
                            options=options, run_metadata=run_metadata)
                    timer.toc()
                    if model.config.WRITE_TIMELINE:
                        timeliner.write_timeline(run_metadata.step_stats,
                                                '{}/timeline_{}.json'.format(
                                                model.config.RESULT_PATH,model.config.DISPLAY_NAME))
                    num = output_dict['num_detections'][0]
                    classes = output_dict['detection_classes'][0]
                    boxes = output_dict['detection_boxes'][0]
                    scores = output_dict['detection_scores'][0]
                    if 'detection_masks' in output_dict:
                        masks = output_dict['detection_masks'][0]

                # reformat detection
                num = int(num)
                boxes = np.squeeze(boxes)
                classes = np.squeeze(classes).astype(np.uint8)
                scores = np.squeeze(scores)

                # Visualization
                visualizer.visualize_objectdetection(frame,boxes,classes,scores,masks,category_index,
                                                    timer.get_frame(),timer.get_fps())
                if model.config.SAVE_RESULT:
                    cv2.imwrite('{}/{}_{}.jpg'.format(model.config.RESULT_PATH,timer.get_frame(),model.config.DISPLAY_NAME),frame)
    visualizer.stop()
    timer.stop()


def main():
    model_type = 'od'
    config = Config(model_type)
    model = Model(config).prepare_model()
    detection(model)

if __name__ == '__main__':
    main()
