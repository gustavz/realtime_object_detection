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
from rod.helper import FPS, WebcamVideoStream, create_colormap, vis_text
from rod.model import Model
from rod.config import Config


def segmentation(model,config):
    detection_graph = model.detection_graph
    # fixed input sizes as model needs resize either way
    vs = WebcamVideoStream(config.VIDEO_INPUT,640,480).start()
    resize_ratio = 1.0 * 513 / max(vs.real_width,vs.real_height)
    target_size = (int(resize_ratio * vs.real_width), int(resize_ratio * vs.real_height)) #(513, 384)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth=True
    fps = FPS(config.FPS_INTERVAL).start()
    print("> Starting Segmentaion")
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph,config=tf_config) as sess:
            while vs.isActive():
                frame = vs.resized(target_size)
                batch_seg_map = sess.run('SemanticPredictions:0',
                                feed_dict={'ImageTensor:0': [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)]})
                # visualization
                if config.VISUALIZE:
                    seg_map = batch_seg_map[0]
                    seg_image = create_colormap(seg_map).astype(np.uint8)
                    cv2.addWeighted(seg_image,config.ALPHA,frame,1-config.ALPHA,0,frame)
                    vis_text(frame,"fps: {}".format(fps.fps_local()),(10,30))
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
                    cv2.imshow(config.DL_MODEL_NAME,frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                fps.update()
    fps.stop()
    vs.stop()


if __name__ == '__main__':
    config = Config()
    model = Model('dl', config.DL_MODEL_NAME, config.DL_MODEL_PATH).prepare_dl_model()
    segmentation(model,config)
