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

from rod.visualizer import Visualizer
from rod.helper import FPS, WebcamVideoStream
from rod.model import Model


def segmentation(model):
    detection_graph = model.detection_graph
    # fixed input sizes as model needs resize either way
    video_stream = WebcamVideoStream(model.VIDEO_INPUT,640,480).start()
    resize_ratio = 1.0 * 513 / max(video_stream.real_width,video_stream.real_height)
    target_size = (int(resize_ratio * video_stream.real_width),
                    int(resize_ratio * video_stream.real_height)) #(513, 384)
    tf_config = model.tf_config
    fps = FPS(model.FPS_INTERVAL).start()
    visualizer = Visualizer('dl')
    print("> Starting Segmentaion")
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph,config=tf_config) as sess:
            while video_stream.isActive() and visualizer.isActive():
                frame = video_stream.resized(target_size)
                batch_seg_map = sess.run('SemanticPredictions:0',
                                        feed_dict={'ImageTensor:0':
                                        [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)]})
                seg_map = batch_seg_map[0]
                boxes = []
                labels = []
                ids = []
                map_labeled = measure.label(seg_map, connectivity=1)
                for region in measure.regionprops(map_labeled):
                    if region.area > model.MINAREA:
                        box = region.bbox
                        id = seg_map[tuple(region.coords[0])]
                        label = model.LABEL_NAMES[id]
                        boxes.append(box)
                        labels.append(label)
                        ids.append(id)
                visualizer.visualize_deeplab(frame,boxes,labels,ids,seg_map,fps._glob_numFrames,fps.fps_local())
                fps.update()
    # End everything
    video_stream.stop()
    visualizer.stop()
    fps.stop()


def main():
    model_type = 'dl'
    model = Model(model_type).prepare_model()
    segmentation(model)

if __name__ == '__main__':
    main()
