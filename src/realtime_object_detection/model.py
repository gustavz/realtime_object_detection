#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: www.github.com/GustavZ
"""

import tensorflow as tf
import tarfile
import copy
import os
import six.moves.urllib as urllib
from tensorflow.core.framework import graph_pb2
import tf_utils

class Model(object):
    """
    Model Class to handle all kind of detection preparation
    """
    def __init__(self, type, model_name, model_path, label_path=None, num_classes=90, split_model=False, ssd_shape=300 ):
        assert type in ['od','dl'], "only deeplab or object_detection models"
        assert ssd_shape in [300,600], "only ssd_mobilenet models of shape 300x300 or 600x600 supported"
        self.type = type
        self.model_name = model_name
        self.model_path = model_path
        self.split_model = split_model
        self.label_path = label_path
        self.num_classes = num_classes
        self.ssd_shape = ssd_shape
        self.detection_graph = tf.Graph()
        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth=True
        self.category_index = None
        self.score = None
        self.expand = None
        print ('> Model: {}'.format(self.model_path))

    def download_model(self):
        if self.type == 'dl':
            download_base = 'http://download.tensorflow.org/models/'
        elif self.type == 'od':
            download_base = 'http://download.tensorflow.org/models/object_detection/'
        model_file = self.model_name + '.tar.gz'
        if not os.path.isfile(self.model_path):
            print('> Model not found. Downloading it now.')
            opener = urllib.request.URLopener()
            opener.retrieve(download_base + model_file, model_file)
            tar_file = tarfile.open(model_file)
            for file in tar_file.getmembers():
              file_name = os.path.basename(file.name)
              if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd() + '/models/')
            os.remove(os.getcwd() + '/' + model_file)
        else:
            print('> Model found. Proceed.')

    def _node_name(self,n):
        if n.startswith("^"):
            return n[1:]
        else:
            return n.split(":")[0]

    def load_frozenmodel(self):
        print('> Loading frozen model into memory')
        if (self.type == 'od' and self.split_model):
            # load a frozen Model and split it into GPU and CPU graphs
            # Hardcoded split points for ssd_mobilenet
            split_nodes = ['Postprocessor/convert_scores','Postprocessor/ExpandDims_1']
            input_graph = tf.Graph()
            with tf.Session(graph=input_graph,config=self.tf_config):
                if self.ssd_shape == 600:
                    shape = 7326
                else:
                    shape = 1917
                self.score = tf.placeholder(tf.float32, shape=(None, shape, self.num_classes), name=split_nodes[0])
                self.expand = tf.placeholder(tf.float32, shape=(None, shape, 1, 4), name=split_nodes[1])
                for node in input_graph.as_graph_def().node:
                    if node.name == split_nodes[0]:
                        score_def = node
                    if node.name == split_nodes[1]:
                        expand_def = node

            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)

                    edges = {}
                    name_to_node_map = {}
                    node_seq = {}
                    seq = 0
                    for node in od_graph_def.node:
                        n = self._node_name(node.name)
                        name_to_node_map[n] = node
                        edges[n] = [self._node_name(x) for x in node.input]
                        node_seq[n] = seq
                        seq += 1
                    for d in split_nodes:
                        assert d in name_to_node_map, "%s is not in graph" % d

                    nodes_to_keep = set()
                    next_to_visit = split_nodes[:]

                    while next_to_visit:
                        n = next_to_visit[0]
                        del next_to_visit[0]
                        if n in nodes_to_keep: continue
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
        else:
            # default model loading procedure
            with self.detection_graph.as_default():
              od_graph_def = tf.GraphDef()
              with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def load_labelmap(self):
        print('> Loading label map')
        label_map = tf_utils.load_labelmap(self.label_path)
        categories = tf_utils.convert_label_map_to_categories(label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = tf_utils.create_category_index(categories)

    def get_tensordict(self, outputs):
        ops = self.detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        self.tensor_dict = {}
        for key in outputs:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                self.tensor_dict[key] = self.detection_graph.get_tensor_by_name(tensor_name)
        return self.tensor_dict

    def prepare_od_model(self):
        self.download_model()
        self.load_frozenmodel()
        self.load_labelmap()
        return self

    def prepare_dl_model(self):
        self.download_model()
        self.load_frozenmodel()
        return self
