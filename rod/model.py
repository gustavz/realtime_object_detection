#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017

@author: www.github.com/GustavZ
"""
import numpy as np
import tensorflow as tf
import tarfile
import copy
import os
import sys
import time
from skimage import measure
import six.moves.urllib as urllib
from tensorflow.core.framework import graph_pb2

from rod.helper import FPS, VideoStream, SessionWorker, conv_detect2track, conv_track2detect, ImageStream, TimeLiner
from rod.visualizer import Visualizer
from rod.tf_utils import reframe_box_masks_to_image_masks
from rod.config import Config
from rod.visualizer import Visualizer
import tf_utils

##################################
########## Model Class ###########
##################################
class Model(object):
    """
    Base Tensorflow Inference Model Class
    """
    def __init__(self,config):
        self.config = config
        self.detection_graph = tf.Graph()
        self.category_index = None
        self.masks = None
        self._tf_config = tf.ConfigProto(allow_soft_placement=True)
        self._tf_config.gpu_options.allow_growth=True
        #self._tf_config.gpu_options.force_gpu_compatible=True
        #self._tf_config.gpu_options.per_process_gpu_memory_fraction = 0.01
        self._run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        self._run_metadata = False
        self._wait_thread = False
        self._is_imageD = False
        self._is_videoD = False
        self._is_rosD = False
        print ('> Model: {}'.format(self.config.MODEL_PATH))

    def download_model(self):
        """
        downlaods model from model_zoo
        """
        if self.config.MODEL_TYPE == 'dl':
            download_base = 'http://download.tensorflow.org/models/'
        elif self.config.MODEL_TYPE == 'od':
            download_base = 'http://download.tensorflow.org/models/object_detection/'
        model_file = self.config.MODEL_NAME + '.tar.gz'
        if not os.path.isfile(self.config.MODEL_PATH) and self.config.DOWNLOAD_MODEL:
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

    def node_name(self,n):
        if n.startswith("^"):
            return n[1:]
        else:
            return n.split(":")[0]

    def load_frozen_graph(self):
        """
        loads graph from frozen model file
        """
        print('> Loading frozen model into memory')
        if (self.config.MODEL_TYPE == 'od' and self.config.SPLIT_MODEL):
            # load a frozen Model and split it into GPU and CPU graphs
            # Hardcoded split points for ssd_mobilenet
            tf.reset_default_graph()
            if self.config.SSD_SHAPE == 600:
                shape = 7326
            else:
                shape = 1917
            self.score = tf.placeholder(tf.float32, shape=(None, shape, self.config.NUM_CLASSES), name=self.config.SPLIT_NODES[0])
            self.expand = tf.placeholder(tf.float32, shape=(None, shape, 1, 4), name=self.config.SPLIT_NODES[1])
            #self.tofloat = tf.placeholder(tf.float32, shape=(None), name=self.config.SPLIT_NODES[2])
            for node in tf.get_default_graph().as_graph_def().node:
                if node.name == self.config.SPLIT_NODES[0]:
                    score_def = node
                if node.name == self.config.SPLIT_NODES[1]:
                    expand_def = node
                #if node.name == self.config.SPLIT_NODES[2]:
                #    tofloat_def = node

            with self.detection_graph.as_default():
                graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.config.MODEL_PATH, 'rb') as fid:
                    serialized_graph = fid.read()
                    graph_def.ParseFromString(serialized_graph)

                    edges = {}
                    name_to_node_map = {}
                    node_seq = {}
                    seq = 0
                    for node in graph_def.node:
                        n = self.node_name(node.name)
                        name_to_node_map[n] = node
                        edges[n] = [self.node_name(x) for x in node.input]
                        node_seq[n] = seq
                        seq += 1
                    for d in self.config.SPLIT_NODES:
                        assert d in name_to_node_map, "%s is not in graph" % d

                    nodes_to_keep = set()
                    next_to_visit = self.config.SPLIT_NODES[:]

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
              graph_def = tf.GraphDef()
              with tf.gfile.GFile(self.config.MODEL_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

    def load_category_index(self):
        """
        creates categorie_index from label_map
        """
        print('> Loading label map')
        label_map = tf_utils.load_labelmap(self.config.LABEL_PATH)
        categories = tf_utils.convert_label_map_to_categories(label_map, max_num_classes=self.config.NUM_CLASSES, use_display_name=True)
        self.category_index = tf_utils.create_category_index(categories)

    def get_tensor_dict(self, outputs):
        """
        returns tensordict for given tensornames list
        """
        ops = self.detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        self.tensor_dict = {}
        for key in outputs:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                self.tensor_dict[key] = self.detection_graph.get_tensor_by_name(tensor_name)
        return self.tensor_dict

    def prepare_model(self):
        """
        first step prepare model
        needs to be called by subclass in re-write process

        Necessary: subclass needs to init
        self._input_stream
        """
        if self.config.MODEL_TYPE is 'od':
            self.download_model()
            self.load_frozen_graph()
            self.load_category_index()
        elif self.config.MODEL_TYPE is 'dl':
            self.download_model()
            self.load_frozen_graph()
        self.fps = FPS(self.config.FPS_INTERVAL).start()
        self._visualizer = Visualizer(self.config).start()
        return self

    def isActive(self):
        """
        checks if stream and visualizer are active
        """
        return self._input_stream.isActive() and self._visualizer.isActive()

    def stop(self):
        """
        stops all Model sub classes
        """
        self._input_stream.stop()
        self._visualizer.stop()
        self.fps.stop()
        if self.config.SPLIT_MODEL and self.config.MODEL_TYPE is 'od':
            self._gpu_worker.stop()
            self._cpu_worker.stop()

    def detect(self):
        """
        needs to be written by subclass
        """
        self.detection = None

    def run(self):
        """
        runs detection loop on video or image
        listens on isActive()
        """
        print("> starting detection")
        self.start()
        while self.isActive():
            # detection
            self.detect()
            # Visualization
            if not self._wait_thread:
                self.visualize_detection()
                self.fps.update()
        self.stop()

    def start(self):
        """
        starts fps and visualizer class
        """
        self.fps.start()
        self._visualizer = Visualizer(self.config).start()

    def visualize_detection(self):
        self.detection = self._visualizer.visualize_detection(self.frame,self.boxes,
                                                            self.classes,self.scores,
                                                            self.masks,self.fps.fps_local(),
                                                            self.category_index,self._is_imageD)

    def prepare_ros(self,node):
        """
        prepares ros Node and ROSInputstream
        only in ros branch usable due to ROS realted package stuff
        """
        assert node in ['detection_node','deeplab_node'], "only 'detection_node' and 'deeplab_node' supported"
        import rospy
        from ros import ROSStream, DetectionPublisher, SegmentationPublisher
        self._is_rosD = True
        rospy.init_node(node)
        self._input_stream = ROSStream(self.config.ROS_INPUT)
        if node is 'detection_node':
            self._ros_publisher = DetectionPublisher()
        if node is 'deeplab_node':
            self._ros_publisher = SegmentationPublisher()
        # check for frame
        while True:
            self.frame = self._input_stream.read()
            time.sleep(1)
            print("...waiting for ROS image")
            if self.frame is not None:
                self.stream_height,self.stream_width = self.frame.shape[0:2]
                break

    def prepare_timeliner(self):
        """
        prepares timeliner and sets tf Run options
        """
        self._run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self._run_metadata = tf.RunMetadata()
        self.timeliner = TimeLiner()

    def prepare_tracker(self):
        """
        prepares KCF tracker
        """
        sys.path.append(os.getcwd()+'/rod/kcf')
        import KCF
        self._tracker = KCF.kcftracker(False, True, False, False)
        self._tracker_counter = 0
        self._track = False

    def run_tracker(self):
        """
        runs KCF tracker on videoStream frame
        !does not work on images, obviously!
        """
        self.frame = self._input_stream.read()
        if self._first_track:
            self._trackers = []
            self._tracker_boxes = self.boxes
            num_tracked = 0
            for box in self.boxes[~np.all(self.boxes == 0, axis=1)]:
                    self._tracker.init(conv_detect2track(box,self._input_stream.real_width,
                                        self._input_stream.real_height),self.tracker_frame)
                    self._trackers.append(self._tracker)
                    num_tracked += 1
                    if num_tracked <= self.config.NUM_TRACKERS:
                        break
            self._first_track = False

        for idx,self._tracker in enumerate(self._trackers):
            tracker_box = self._tracker.update(self.frame)
            self._tracker_boxes[idx,:] = conv_track2detect(tracker_box,
                                                    self._input_stream.real_width,
                                                    self._input_stream.real_height)
        self._tracker_counter += 1
        self.boxes = self._tracker_boxes
        # Deactivate Tracker
        if self._tracker_counter >= self.config.TRACKER_FRAMES:
            self._track = False
            self._tracker_counter = 0

    def activate_tracker(self):
        """
        activates KCF tracker
        deactivates mask detection
        """
        #self.masks = None
        self.tracker_frame = self.frame
        self._track = True
        self._first_track = True



##################################
### ObjectDetectionModel Class ###
##################################
class ObjectDetectionModel(Model):
    """
    object_detection model class
    """
    def __init__(self,config):
        super(ObjectDetectionModel, self).__init__(config)

    def prepare_input_stream(self):
        """
        prepares Input Stream
        stream types: 'video','image','ros'
        gets called by prepare model
        """
        if self.input_type is 'video':
            self._is_videoD = True
            self._input_stream = VideoStream(self.config.VIDEO_INPUT,self.config.WIDTH,
                                                    self.config.HEIGHT).start()
            self.stream_height = self._input_stream.real_height
            self.stream_width = self._input_stream.real_width
        elif self.input_type is 'image':
            self._is_imageD = True
            self._input_stream = ImageStream(self.config.IMAGE_PATH,self.config.LIMIT_IMAGES,
                                            (self.config.WIDTH,self.config.HEIGHT)).start()
            self.stream_height = self.config.HEIGHT
            self.stream_width = self.config.WIDTH
        elif self.input_type is 'ros':
            self.prepare_ros('detection_node')
        # Timeliner for image detection
        if self.config.WRITE_TIMELINE:
            self.prepare_timeliner()


    def prepare_model(self,input_type):
        """
        prepares Object_Detection model
        input_type: must be 'image', 'video', or 'ros'
        """
        assert input_type in ['image','video','ros'], "only 'image','video' and 'ros' input possible"
        super(ObjectDetectionModel, self).prepare_model()
        self.input_type = input_type
        # Tracker
        if self.config.USE_TRACKER:
            self.prepare_tracker()
        print("> Building Graph")
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph,config=self._tf_config) as self._sess:
                # Prepare Input Stream
                self.prepare_input_stream()
                # Define Input and Ouput tensors
                self._tensor_dict = self.get_tensor_dict(['num_detections', 'detection_boxes',
                                                        'detection_scores','detection_classes', 'detection_masks'])
                self._image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Mask Transformations
                if 'detection_masks' in self._tensor_dict:
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    detection_boxes = tf.squeeze(self._tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(self._tensor_dict['detection_masks'], [0])
                    real_num_detection = tf.cast(self._tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = reframe_box_masks_to_image_masks(detection_masks, detection_boxes,self.stream_height,self.stream_width)
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    self._tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                if self.config.SPLIT_MODEL:
                    self._score_out = self.detection_graph.get_tensor_by_name('{}:0'.format(self.config.SPLIT_NODES[0]))
                    self._expand_out = self.detection_graph.get_tensor_by_name('{}:0'.format(self.config.SPLIT_NODES[1]))
                    self._score_in = self.detection_graph.get_tensor_by_name('{}_1:0'.format(self.config.SPLIT_NODES[0]))
                    self._expand_in = self.detection_graph.get_tensor_by_name('{}_1:0'.format(self.config.SPLIT_NODES[1]))
                    # Threading
                    self._gpu_worker = SessionWorker("GPU",self.detection_graph,self._tf_config)
                    self._cpu_worker = SessionWorker("CPU",self.detection_graph,self._tf_config)
                    self._gpu_opts = [self._score_out,self._expand_out]
                    self._cpu_opts = [self._tensor_dict['detection_boxes'],
                                    self._tensor_dict['detection_scores'],
                                    self._tensor_dict['detection_classes'],
                                    self._tensor_dict['num_detections']]
            return self

    def run_default_sess(self):
        """
        runs default session
        """
        # default session)
        self.frame = self._input_stream.read()
        output_dict = self._sess.run(self._tensor_dict,
                                    feed_dict={self._image_tensor:
                                    self._visualizer.expand_and_convertRGB_image(self.frame)},
                                    options=self._run_options, run_metadata=self._run_metadata)
        self.num = output_dict['num_detections'][0]
        self.classes = output_dict['detection_classes'][0]
        self.boxes = output_dict['detection_boxes'][0]
        self.scores = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            self.masks = output_dict['detection_masks'][0]

    def run_thread_sess(self):
        """
        runs seperate gpu and cpu session threads
        """
        if self._gpu_worker.is_sess_empty():
            # put new queue
            self.frame = self._input_stream.read()
            gpu_feeds = {self._image_tensor: self._visualizer.expand_and_convertRGB_image(self.frame)}
            if self.config.VISUALIZE:
                gpu_extras = self.frame # for visualization frame
            else:
                gpu_extras = None
            self._gpu_worker.put_sess_queue(self._gpu_opts,gpu_feeds,gpu_extras)
        g = self._gpu_worker.get_result_queue()
        if g is None:
            # gpu thread has no output queue. ok skip, let's check cpu thread.
            pass
        else:
            # gpu thread has output queue.
            score,expand,self._frame = g["results"][0],g["results"][1],g["extras"]
            if self._cpu_worker.is_sess_empty():
                # When cpu thread has no next queue, put new queue.
                # else, drop gpu queue.
                cpu_feeds = {self._score_in: score, self._expand_in: expand}
                cpu_extras = self.frame
                self._cpu_worker.put_sess_queue(self._cpu_opts,cpu_feeds,cpu_extras)
        c = self._cpu_worker.get_result_queue()
        if c is None:
            # cpu thread has no output queue. ok, nothing to do. continue
            self._wait_thread = True
            return # If CPU RESULT has not been set yet, no fps update
        else:
            self._wait_thread = False
            self.boxes,self.scores,self.classes,self.num,self.frame = c["results"][0],c["results"][1],c["results"][2],c["results"][3],c["extras"]

    def run_split_sess(self):
        """
        runs split session WITHOUT threading
        optional: timeline writer
        """
        self.frame = self._input_stream.read()
        score, expand = self._sess.run(self._gpu_opts,feed_dict={self._image_tensor:
                                        self._visualizer.expand_and_convertRGB_image(self.frame)},
                                        options=self._run_options, run_metadata=self._run_metadata)
        if self.config.WRITE_TIMELINE:
            self.timeliner.write_timeline(self._run_metadata.step_stats,
                                        '{}/timeline_{}_SM1.json'.format(
                                        self.config.RESULT_PATH,self.config.DISPLAY_NAME))
        # CPU Session
        self.boxes,self.scores,self.classes,self.num = self._sess.run(self._cpu_opts,
                                                                    feed_dict={self._score_in:score,
                                                                    self._expand_in: expand},
                                                                    options=self._run_options,
                                                                    run_metadata=self._run_metadata)
        if self.config.WRITE_TIMELINE:
            self.timeliner.write_timeline(self._run_metadata.step_stats,
                                        '{}/timeline_{}_SM2.json'.format(
                                        self.config.RESULT_PATH,self.config.DISPLAY_NAME))


    def reformat_detection(self):
        """
        reformats detection
        """
        self.num = int(self.num)
        self.boxes = np.squeeze(self.boxes)
        self.classes = np.squeeze(self.classes).astype(np.uint8)
        self.scores = np.squeeze(self.scores)

    def detect(self):
        """
        Object_Detection Detection function
        optional: multi threading split session, timline writer
        """
        if not (self.config.USE_TRACKER and self._track):
            if self.config.SPLIT_MODEL:
                if self.config.MULTI_THREADING:
                    self.run_thread_sess()
                    if self._wait_thread: # checks if thread has output
                        return
                else:
                    self.run_split_sess()
            else:
                self.run_default_sess()
                if self.config.WRITE_TIMELINE:
                    self.timeliner.write_timeline(self._run_metadata.step_stats,
                                            '{}/timeline_{}.json'.format(
                                            self.config.RESULT_PATH,self.config.DISPLAY_NAME))
            self.reformat_detection()
            # Activate Tracker
            if self.config.USE_TRACKER and not self._is_imageD:
                self.activate_tracker()
        # Tracking
        else:
            self.run_tracker()

        # Publish ROS Message
        if self._is_rosD:
            self._ros_publisher.publish(self.boxes,self.scores,self.classes,self.num,self.category_index,self.frame.shape,self.masks,self.fps.fps_local())



##################################
###### DeepLabModel Class ########
##################################
class DeepLabModel(Model):
    def __init__(self,config):
        super(DeepLabModel, self).__init__(config)

    def prepare_input_stream(self):
        if self.input_type is 'video':
            self._is_videoD = True
            self._input_stream = VideoStream(self.config.VIDEO_INPUT,self.config.WIDTH,self.config.HEIGHT).start()
        elif self.input_type is 'image':
            self._is_imageD = True
            self._input_stream = ImageStream(self.config.IMAGE_PATH,self.config.LIMIT_IMAGES).start()
            if self.config.WRITE_TIMELINE:
                self.prepare_timeliner()
        elif self._input_type is 'ros':
            self.prepare_ros('deeplab_node')



    def prepare_model(self,input_type):
        """
        prepares DeepLab model
        input_type: must be 'image', 'video', or 'ros'
        """
        assert input_type in ['image','video','ros'], "only image, video or ros input possible"
        super(DeepLabModel, self).prepare_model()
        self.input_type = input_type
        # Tracker
        if self.config.USE_TRACKER:
            self.prepare_tracker()
        # Input Stream
        self.category_index = None
        self.prepare_input_stream()
        print("> Building Graph")
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph,config=self._tf_config) as self._sess:
                return self

    def detect(self):
        """
        DeepLab Detection function
        """
        if not (self.config.USE_TRACKER and self._track):
            self.frame = self._input_stream.read()
            height,width,_ = self.frame.shape
            resize_ratio = 1.0 * 513 / max(self._input_stream.real_width,self._input_stream.real_height)
            target_size = (int(resize_ratio * self._input_stream.real_width),int(resize_ratio * self._input_stream.real_height)) #(513, 342)?(513,384)
            self.frame = self._visualizer.resize_image(self.frame, target_size)
            batch_seg_map = self._sess.run('SemanticPredictions:0',
                                            feed_dict={'ImageTensor:0':
                                            [self._visualizer.convertRGB_image(self.frame)]},
                                            options=self._run_options, run_metadata=self._run_metadata)
            if self.config.WRITE_TIMELINE:
                self._timeliner.write_timeline(self._run_metadata.step_stats,
                                        '{}/timeline_{}.json'.format(
                                        self.config.RESULT_PATH,self.config.DISPLAY_NAME))
            seg_map = batch_seg_map[0]
            self.boxes = []
            self.labels = []
            self.ids = []
            if self.config.BBOX:
                map_labeled = measure.label(seg_map, connectivity=1)
                for region in measure.regionprops(map_labeled):
                    if region.area > self.config.MINAREA:
                        box = region.bbox
                        id = seg_map[tuple(region.coords[0])]
                        label = self.config.LABEL_NAMES[id]
                        self.boxes.append(box)
                        self.labels.append(label)
                        self.ids.append(id)
            # deeplab workaround
            self.num = len(self.boxes)
            self.classes = self.ids
            self.scores = self.labels
            self.masks = seg_map
            # Activate Tracker
            if self.config.USE_TRACKER and not self._is_imageD:
                self.activate_tracker()
        else:
            self.run_tracker()

        # publish ros
        if self._is_rosD:
            self._ros_publisher.publish(self.boxes,self.labels,self.masks,self.frame.shape,self.fps.fps_local())
