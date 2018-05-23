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
import yaml
import datetime
from stuff.helper import Model, Timer, WebcamVideoStream, SessionWorker, vis_detection, TimeLiner, load_images
from object_detection.utils import ops as utils_ops
from tensorflow.python.client import timeline


## LOAD CONFIG PARAMS ##
if (os.path.isfile('config.yml')):
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
else:
    with open("config.sample.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
VIDEO_INPUT     = cfg['video_input']
VISUALIZE       = cfg['visualize']
MAX_FRAMES      = cfg['max_frames']
WIDTH           = cfg['width']
HEIGHT          = cfg['height']
DET_INTERVAL    = cfg['det_interval']
DET_TH          = cfg['det_th']
IMAGE_PATH      = cfg['image_path']
LIMIT_IMAGES    = cfg['limit_images']
CPU_ONLY        = cfg['cpu_only']
MODEL_NAME      = cfg['od_model_name']
MODEL_PATH      = cfg['od_model_path']
LABEL_PATH      = cfg['label_path']
NUM_CLASSES     = cfg['num_classes']
SPLIT_MODEL     = cfg['split_model']
SSD_SHAPE       = cfg['ssd_shape']
WRITE_TIMELINE  = cfg['write_timeline']

if CPU_ONLY:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	DEVICE = '_CPU'
else:
	DEVICE = '_GPU'

def detection(model):
    # Tf Session + Timeliner
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    detection_graph = model.detection_graph
    category_index = model.category_index
    if WRITE_TIMELINE:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        many_runs_timeline = TimeLiner()
    else:
        options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        run_metadata = False
    print("> Building Graph")
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph,config=config) as sess:
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
                        detection_masks, detection_boxes, HEIGHT, WIDTH)
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            if SPLIT_MODEL:
                score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
                expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
                # Threading
                score = model.score
                expand = model.expand

            images = load_images(IMAGE_PATH,LIMIT_IMAGES)
            timer = Timer().start()
            print('> Starting Detection')
            for image in images:
                if SPLIT_MODEL:
                    # split model in seperate gpu and cpu session threads
                    masks = None # No Mask Detection possible yet
                    frame = cv2.resize(cv2.imread(image),(WIDTH,HEIGHT))
                    frame_expanded = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0)
                    timer.tic()
                    score, expand = sess.run([score_out, expand_out],
                            feed_dict={image_tensor: frame_expanded},
                            options=options, run_metadata=run_metadata)
                    boxes, scores, classes, num = sess.run(
                            [tensor_dict['detection_boxes'], tensor_dict['detection_scores'], tensor_dict['detection_classes'], tensor_dict['num_detections']],
                            feed_dict={score_in:score, expand_in: expand},
                            options=options, run_metadata=run_metadata)
                    timer.toc()
                else:
                    # default session
                    frame = cv2.resize(cv2.imread(image),(WIDTH,HEIGHT))
                    frame_expanded = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0)
                    timer.tic()
                    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: frame_expanded},
                            options=options, run_metadata=run_metadata)
                    timer.toc()
                    num = output_dict['num_detections'][0]
                    classes = output_dict['detection_classes'][0]
                    boxes = output_dict['detection_boxes'][0]
                    scores = output_dict['detection_scores'][0]
                    if 'detection_masks' in output_dict:
                        masks = output_dict['detection_masks'][0]
                    else:
                        masks = None
                # Timeliner
                if WRITE_TIMELINE:
    		        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    		        chrome_trace = fetched_timeline.generate_chrome_trace_format()
    		        many_runs_timeline.update_timeline(chrome_trace)
    		        with open('{}_timeline{}.json'.format(MODEL_NAME,DEVICE), 'w') as f:
    		        	f.write(chrome_trace)
                # reformat detection
                num = int(num)
                boxes = np.squeeze(boxes)
                classes = np.squeeze(classes).astype(np.uint8)
                scores = np.squeeze(scores)

                # Visualization
                vis = vis_detection(frame, VISUALIZE, boxes, classes, scores, masks, category_index, DET_INTERVAL, DET_TH, MAX_FRAMES)
                if not vis:
                    break

    cv2.destroyAllWindows()
    timer.stop()


if __name__ == '__main__':
    model = Model('od', MODEL_NAME, MODEL_PATH, LABEL_PATH,
            NUM_CLASSES, SPLIT_MODEL, SSD_SHAPE).prepare_od_model()
    detection(model)
