#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017

@author: www.github.com/GustavZ
"""
import numpy as np
import tensorflow as tf
import yaml
import os
from stuff.helper import Model, FPS, WebcamVideoStream, SessionWorker, conv_detect2track, conv_track2detect, vis_detection
from object_detection.utils import ops as utils_ops


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
FPS_INTERVAL    = cfg['fps_interval']
DET_INTERVAL    = cfg['det_interval']
DET_TH          = cfg['det_th']
MODEL_NAME      = cfg['od_model_name']
MODEL_PATH      = cfg['od_model_path']
LABEL_PATH      = cfg['label_path']
NUM_CLASSES     = cfg['num_classes']
SPLIT_MODEL     = cfg['split_model']
SSD_SHAPE       = cfg['ssd_shape']
USE_TRACKER     = cfg['use_tracker']
TRACKER_FRAMES  = cfg['tracker_frames']
NUM_TRACKERS    = cfg['num_trackers']


def detection(model):
    # Tracker
    if USE_TRACKER:
        import sys
        sys.path.append(os.getcwd()+'/stuff/kcf')
        import KCF
        tracker = KCF.kcftracker(False, True, False, False)
        tracker_counter = 0
        track = False

    print("> Building Graph")
    # tf Session Config
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    detection_graph = model.detection_graph
    category_index = model.category_index
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph,config=config) as sess:
            # start Videostream
            vs = WebcamVideoStream(VIDEO_INPUT,WIDTH,HEIGHT).start()
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
                        detection_masks, detection_boxes, vs.real_height, vs.real_width)
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
                gpu_worker = SessionWorker("GPU",detection_graph,config)
                cpu_worker = SessionWorker("CPU",detection_graph,config)
                gpu_opts = [score_out, expand_out]
                cpu_opts = [tensor_dict['detection_boxes'], tensor_dict['detection_scores'], tensor_dict['detection_classes'], tensor_dict['num_detections']]
                gpu_counter = 0
                cpu_counter = 0

            fps = FPS(FPS_INTERVAL).start()
            print('> Starting Detection')
            while vs.isActive():
                # Detection
                if not (USE_TRACKER and track):
                    if SPLIT_MODEL:
                        # split model in seperate gpu and cpu session threads
                        masks = None # No Mask Detection possible yet
                        if gpu_worker.is_sess_empty():
                            # read video frame, expand dimensions and convert to rgb
                            frame = vs.read()
                            # put new queue
                            gpu_feeds = {image_tensor: vs.expanded()}
                            if VISUALIZE:
                                gpu_extras = frame # for visualization frame
                            else:
                                gpu_extras = None
                            gpu_worker.put_sess_queue(gpu_opts,gpu_feeds,gpu_extras)
                        g = gpu_worker.get_result_queue()
                        if g is None:
                            # gpu thread has no output queue. ok skip, let's check cpu thread.
                            gpu_counter += 1
                        else:
                            # gpu thread has output queue.
                            gpu_counter = 0
                            score,expand,frame = g["results"][0],g["results"][1],g["extras"]

                            if cpu_worker.is_sess_empty():
                                # When cpu thread has no next queue, put new queue.
                                # else, drop gpu queue.
                                cpu_feeds = {score_in: score, expand_in: expand}
                                cpu_extras = frame
                                cpu_worker.put_sess_queue(cpu_opts,cpu_feeds,cpu_extras)
                        c = cpu_worker.get_result_queue()
                        if c is None:
                            # cpu thread has no output queue. ok, nothing to do. continue
                            cpu_counter += 1
                            continue # If CPU RESULT has not been set yet, no fps update
                        else:
                            cpu_counter = 0
                            boxes, scores, classes, num, frame = c["results"][0],c["results"][1],c["results"][2],c["results"][3],c["extras"]
                    else:
                        # default session
                        frame = vs.read()
                        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: vs.expanded()})
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
                    vis = vis_detection(frame, VISUALIZE, boxes, classes, scores, masks, category_index, DET_INTERVAL, DET_TH, MAX_FRAMES, fps)
                    if not vis:
                        break

                    # Activate Tracker
                    if USE_TRACKER and num <= NUM_TRACKERS:
                        tracker_frame = frame
                        track = True
                        first_track = True

                # Tracking
                else:
                    frame = vs.read()
                    if first_track:
                        trackers = []
                        tracker_boxes = boxes
                        for box in boxes[~np.all(boxes == 0, axis=1)]:
                                tracker.init(conv_detect2track(box,vs.real_width, vs.real_height), tracker_frame)
                                trackers.append(tracker)
                        first_track = False

                    for idx,tracker in enumerate(trackers):
                        tracker_box = tracker.update(frame)
                        tracker_boxes[idx,:] = conv_track2detect(tracker_box, vs.real_width, vs.real_height)
                    vis = vis_detection(frame, VISUALIZE, tracker_boxes, classes, scores, masks, category_index, DET_INTERVAL, DET_TH, MAX_FRAMES, fps)
                    if not vis:
                        break

                    tracker_counter += 1
                    #tracker_frame = frame
                    if tracker_counter >= TRACKER_FRAMES:
                        track = False
                        tracker_counter = 0

                fps.update()

    # End everything
    vs.stop()
    fps.stop()
    if SPLIT_MODEL:
        gpu_worker.stop()
        cpu_worker.stop()


if __name__ == '__main__':
    model = Model('od', MODEL_NAME, MODEL_PATH, LABEL_PATH, NUM_CLASSES,
            SPLIT_MODEL, SSD_SHAPE).prepare_od_model()
    detection(model)
