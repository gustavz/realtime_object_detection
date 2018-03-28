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
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import yaml
import datetime

## LOAD CONFIG PARAMS ##
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
model_path          = cfg['model_path']
label_path          = cfg['label_path']
num_classes         = cfg['num_classes']
det_th              = cfg['det_th']
image_path          = cfg['image_path']

# load images
images = []
for root, dirs, files in os.walk(image_path):
    for file in files:
        if file.endswith(".jpg"):
            images.append(os.path.join(root, file))
images.sort()

#Load a (frozen) Tensorflow model into memory.
print('> Loading frozen model into memory')
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(model_path, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#Load Label Map
print('> Loading label map')
label_map = label_map_util.load_labelmap(label_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#Detection
print("> Building Graph")
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image in images:
      print("> Detecting")
      image_np = cv2.imread(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), axis=0)
      # Actual detection.
      start = datetime.datetime.now()
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      stop = datetime.datetime.now()
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      for box, score, _class in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
                    if score > det_th:
                        label = category_index[_class]['name']
                        print("label: {}\nscore: {}\nbox: {}".format(label, score, box))
      print ("elapsed time for actual detection: {} seconds".format((stop-start).total_seconds()))
      cv2.imshow('test_images',image_np)
      cv2.waitKey(2000)

print ("press 'q' to Exit")
if cv2.waitKey(999999) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
