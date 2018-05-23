#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:17:38 2018

@author: github.com/GustavZ
"""
import os
import tarfile
from six.moves import urllib
import numpy as np
import tensorflow as tf
import yaml
import cv2
from stuff.helper import TimeLiner, Timer, Model, create_colormap, vis_text, load_images
from skimage import measure
from tensorflow.python.client import timeline

## LOAD CONFIG PARAMS ##
if (os.path.isfile('config.yml')):
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
else:
    with open("config.sample.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
VIDEO_INPUT		= cfg['video_input']
ALPHA			= cfg['alpha']
BBOX            = cfg['bbox']
MINAREA         = cfg['minArea']
MODEL_NAME		= cfg['dl_model_name']
MODEL_PATH		= cfg['dl_model_path']
IMAGE_PATH		= cfg['image_path']
LIMIT_IMAGES    = cfg['limit_images']
CPU_ONLY 		= cfg['cpu_only']
WRITE_TIMELINE  = cfg['write_timeline']
VISUALIZE       = cfg['visualize']

if CPU_ONLY:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	DEVICE = '_CPU'
else:
	DEVICE = '_GPU'


# Hardcoded COCO_VOC Labels
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'])

def segmentation(model):
    images = load_images(IMAGE_PATH,LIMIT_IMAGES)
    # Tf Session + Timeliner
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    detection_graph = model.detection_graph
    if WRITE_TIMELINE:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        many_runs_timeline = TimeLiner()
    else:
        options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        run_metadata = False
    timer = Timer().start()
    print("> Starting Segmentaion")
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
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
                if WRITE_TIMELINE:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    many_runs_timeline.update_timeline(chrome_trace)
                    with open('test_timeline{}.json'.format(DEVICE), 'w') as f:
                    	f.write(chrome_trace)
                # visualization
                if VISUALIZE:
                    seg_map = batch_seg_map[0]
                    seg_image = create_colormap(seg_map).astype(np.uint8)
                    cv2.addWeighted(seg_image,ALPHA,frame,1-ALPHA,0,frame)
                    vis_text(frame,"time: {}".format(timer._time),(10,30))
                    # boxes (ymin, xmin, ymax, xmax)
                    if BBOX:
                        map_labeled = measure.label(seg_map, connectivity=1)
                        for region in measure.regionprops(map_labeled):
                            if region.area > MINAREA:
                                box = region.bbox
                                p1 = (box[1], box[0])
                                p2 = (box[3], box[2])
                                cv2.rectangle(frame, p1, p2, (77,255,9), 2)
                                vis_text(frame,LABEL_NAMES[seg_map[tuple(region.coords[0])]],(p1[0],p1[1]-10))
                    cv2.imshow('segmentation',frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
	cv2.destroyAllWindows()
    timer.stop()


if __name__ == '__main__':
    model = Model('dl', MODEL_NAME, MODEL_PATH).prepare_dl_model()
    segmentation(model)
