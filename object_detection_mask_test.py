#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""q
Created on Thu Dec 21 12:01:40 2017

@author: GustavZ
"""
import numpy as np
import os
import tensorflow as tf
import copy
import yaml
import cv2
import tarfile
import six.moves.urllib as urllib
from tensorflow.core.framework import graph_pb2

# Protobuf Compilation (once necessary)
#os.system('protoc object_detection/protos/*.proto --python_out=.')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from stuff.helper import FPS2, WebcamVideoStream


## LOAD CONFIG PARAMS ##
if (os.path.isfile('config.yml')):
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
else:
    with open("config.sample.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        
video_input         = cfg['video_input']
visualize           = cfg['visualize']
vis_text            = cfg['vis_text']
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
split_model         = cfg['split_model']
log_device          = cfg['log_device']
ssd_shape           = cfg['ssd_shape']


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
     
# helper function for split model
def _node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]
        
# Load a (frozen) Tensorflow model into memory.
def load_frozenmodel():
    print('Loading frozen model into memory')
    if not split_model:
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        return detection_graph, None, None
    
    else:
        # load a frozen Model and split it into GPU and CPU graphs
        # Hardcoded for ssd_mobilenet
        input_graph = tf.Graph()
        with tf.Session(graph=input_graph):
            if ssd_shape == 600:
                shape = 7326
            else:
                shape = 1917
            score = tf.placeholder(tf.float32, shape=(None, shape, num_classes), name="Postprocessor/convert_scores")
            expand = tf.placeholder(tf.float32, shape=(None, shape, 1, 4), name="Postprocessor/ExpandDims_1")
            for node in input_graph.as_graph_def().node:
                if node.name == "Postprocessor/convert_scores":
                    score_def = node
                if node.name == "Postprocessor/ExpandDims_1":
                    expand_def = node
                    
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            dest_nodes = ['Postprocessor/convert_scores','Postprocessor/ExpandDims_1']
        
            edges = {}
            name_to_node_map = {}
            node_seq = {}
            seq = 0
            for node in od_graph_def.node:
              n = _node_name(node.name)
              name_to_node_map[n] = node
              edges[n] = [_node_name(x) for x in node.input]
              node_seq[n] = seq
              seq += 1
        
            for d in dest_nodes:
              assert d in name_to_node_map, "%s is not in graph" % d
        
            nodes_to_keep = set()
            next_to_visit = dest_nodes[:]
            while next_to_visit:
              n = next_to_visit[0]
              del next_to_visit[0]
              if n in nodes_to_keep:
                continue
              nodes_to_keep.add(n)
              next_to_visit += edges[n]
        
            nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])
        
            nodes_to_remove = set()
            for n in node_seq:
              if n in nodes_to_keep_list: continue
              nodes_to_remove.add(n)
            nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])
        
            keep = graph_pb2.GraphDef()
            for n in nodes_to_keep_list:
              keep.node.extend([copy.deepcopy(name_to_node_map[n])])
        
            remove = graph_pb2.GraphDef()
            remove.node.extend([score_def])
            remove.node.extend([expand_def])
            for n in nodes_to_remove_list:
              remove.node.extend([copy.deepcopy(name_to_node_map[n])])
        
            with tf.device('/gpu:0'):
              tf.import_graph_def(keep, name='')
            with tf.device('/cpu:0'):
              tf.import_graph_def(remove, name='')
              
        return detection_graph, score, expand


def load_labelmap():
    print('Loading label map')
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def get_tensordict(detection_graph, outputs):
    ops = detection_graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in outputs:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)
    return tensor_dict


def detection(detection_graph, category_index, score, expand):
    outputs = ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
    config.gpu_options.allow_growth=allow_memory_growth
    cur_frames = 0
    print('Starting detection')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
            # Start Video Stream
            video_stream = WebcamVideoStream(video_input,width,height).start()
            print ("Press 'q' to Exit")
            # Get handles to input and output tensors
            tensor_dict = get_tensordict(detection_graph, outputs)            
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            if 'detection_masks' in tensor_dict:
                #real_width, real_height = get_image_shape()
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, video_stream.real_height, video_stream.real_width)
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            if split_model:
                score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
                expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
            # fps calculation
            fps = FPS2(fps_interval).start()
            cur_frames = 0
            while video_stream.isActive():
                image = video_stream.read()
                image_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),0)
                # detection
                if split_model:
                    (score, expand) = sess.run([score_out, expand_out], feed_dict={image_tensor:image_expanded})
                    output_dict = sess.run(tensor_dict, feed_dict={score_in:score, expand_in: expand})
                else:
                    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_expanded})
                #num = int(output_dict['num_detections'][0])
                classes = output_dict['detection_classes'][0].astype(np.uint8)
                boxes = output_dict['detection_boxes'][0]
                scores = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
                # Visualization of the results of a detection.
                if visualize:
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image,
                        boxes,
                        classes,
                        scores,
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    if vis_text:
                        cv2.putText(image,"fps: {}".format(fps.fps_local()), (10,30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                    cv2.imshow('object_detection', image)
                    # Exit Option
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # Exit after max frames if no visualization
                    cur_frames += 1
                    for box, score, _class in zip(boxes, scores, classes):
                            if cur_frames%det_interval==0 and score > det_th:
                                label = category_index[_class]['name']
                                print("label: {}\nscore: {}\nbox: {}".format(label, score, box))
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
    graph, score, expand = load_frozenmodel()
    category = load_labelmap()
    detection(graph, category, score, expand)
    
    
if __name__ == '__main__':
    main()