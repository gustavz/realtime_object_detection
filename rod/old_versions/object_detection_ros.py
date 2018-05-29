#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017

@author: GustavZ
"""
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2
import yaml


# Protobuf Compilation (once necessary)
#os.system('protoc object_detection/protos/*.proto --python_out=.')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from stuff.helper_ros import FPS2, WebcamVideoStream

## LOAD CONFIG PARAMS ##
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
video_input         = cfg['video_input']
visualize           = cfg['visualize']
max_frames          = cfg['max_frames']
width               = cfg['width']
height              = cfg['height']
fps_interval        = cfg['fps_interval']
allow_memory_growth = cfg['allow_memory_growth']
det_interval        = cfg['det_interval']
det_th              = cfg['det_th']
model_name          = cfg['model_name']
model_path          = cfg['model_path']
label_path          = cfg['label_path']
num_classes         = cfg['num_classes']
enable_ros          = cfg['enable_ros']

# Init Rosnode and msg Publisher
if enable_ros:
    rosInstalled = True
    try:
        import rospy
        from stuff.helper_ros import RosDetectionPublisher
    except ImportError:
        rosInstalled = False
        print("no ros packages installed\nstarting without ros")
    if rosInstalled:
        print("##\nroscore must run and catkin_ws/devel/setup.bash must be sourced\n##")
        rospy.init_node('object_detection')        
        rospub = RosDetectionPublisher() 


# Download Model form TF's Model Zoo
def download_model():
    model_file = model_name + '.tar.gz'
    download_base = 'http://download.tensorflow.org/models/object_detection/'   
    if not os.path.isfile(model_path):
        print('Model not found. Downloading it now.')
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, model_file)
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd() + '/models/')
        os.remove(os.getcwd() + '/' + model_file)
    else:
        print('Model found. Proceed.')
        
        
def load_frozenmodel():
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    # Loading label map
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return detection_graph, category_index
    

def detection(detection_graph, category_index):
    # Session Config: Limit GPU Memory Usage
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=allow_memory_growth
    
    cur_frames = 0
    # Detection
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph, config = config) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # fps calculation
        fps = FPS2(fps_interval).start()
        # Start Video Stream
        video_stream = WebcamVideoStream(video_input,width,height).start()
        print ("Press 'q' to Exit")
        while video_stream.isActive():
          image_np = video_stream.read()
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
          (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Publish Ros Msg
          if enable_ros and rosInstalled:
              rospub.publish(np.squeeze(boxes), np.squeeze(scores), 
                             np.squeeze(classes).astype(np.int32), num, image_np.shape, category_index)
          # Visualization of the results of a detection.
          if visualize:
              vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
              cv2.putText(image_np,"fps: {}".format(fps.fps_local()), (10,30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
              cv2.imshow('object_detection', image_np)
              # Exit Option
              if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
          else:
              cur_frames += 1
              for box, score, _class in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
                    if cur_frames%det_interval==0 and score > det_th:
                        label = category_index[_class]['name']
                        print(label, score, box)
              if cur_frames >= max_frames:
                  break
          fps.update()
    # End everything
    fps.stop()
    video_stream.stop()     
    cv2.destroyAllWindows()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

def main():                  
    download_model()    
    dg, ci = load_frozenmodel()  
    detection(dg, ci)
    
if __name__ == '__main__':
    main()
