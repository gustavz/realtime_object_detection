#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:53:52 2017

@author: GustavZ
"""
import datetime
import cv2
import threading
import time
import numpy as np
import tensorflow as tf
import tarfile
import copy
import os
import json
import six.moves.urllib as urllib
from tensorflow.core.framework import graph_pb2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import sys
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY2:
    import Queue
elif PY3:
    import queue as Queue


class Model:
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
        self.category_index = None
        self.score = None
        self.expand = None

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
            with tf.Session(graph=input_graph):
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
        label_map = label_map_util.load_labelmap(self.label_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

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



class FPS:
    """
    Class for FPS calculation
    """
    def __init__(self, interval):
        self._glob_start = None
        self._glob_end = None
        self._glob_numFrames = 0
        self._local_start = None
        self._local_numFrames = 0
        self._interval = interval
        self._curr_time = None
        self._curr_local_elapsed = None
        self._first = False

    def start(self):
        self._glob_start = datetime.datetime.now()
        self._local_start = self._glob_start
        return self

    def stop(self):
        self._glob_end = datetime.datetime.now()
        print('> [INFO] elapsed time (total): {:.2f}'.format(self.elapsed()))
        print('> [INFO] approx. FPS: {:.2f}'.format(self.fps()))

    def update(self):
        self.first = True
        self._curr_time = datetime.datetime.now()
        self.curr_local_elapsed = (self._curr_time - self._local_start).total_seconds()
        self._glob_numFrames += 1
        self._local_numFrames += 1
        if self._curr_local_elapsed > self._interval:
          print("> FPS: {}".format(self.fps_local()))
          self._local_numFrames = 0
          self._local_start = self._curr_time

    def elapsed(self):
        return (self._glob_end - self._glob_start).total_seconds()

    def fps(self):
        return self._glob_numFrames / self.elapsed()

    def fps_local(self):
        if self._first:
            return round(self._local_numFrames / self._curr_local_elapsed,1)
        else:
            return 0.0


class Timer:
    """
    Timer class for benchmark test purposes
    """
    def __init__(self):
        self._tic = None
        self._toc = None
        self._time = None
        self._log = []
        self._totaltime = None
        self._totalnumber = None
        self._meantime = None
        self._mediantime = None
        self._mintime = None
        self._maxtime = None
        self._stdtime = None
        self._meanfps = None
        self._medianfps = None

    def start(self):
        return self

    def tic(self):
        self._tic = datetime.datetime.now()

    def toc(self):
        self._toc = datetime.datetime.now()
        self._time = (self._toc-self._tic).total_seconds()
        self._log.append(self._time)

    def update(self):
        self._totaltime = np.sum(self._log)
        self._totalnumber = len(self._log)
        self._meantime = np.mean(self._log)
        self._mediantime = np.median(self._log)
        self._mintime = np.min(self._log)
        self._maxtime = np.max(self._log)
        self._stdtime = np.std(self._log)
        self._meanfps = 1/np.mean(self._log)
        self._medianfps = 1/np.median(self._log)

    def stop(self):
        self.update()
        print ("> [INFO] total detection time for {} images: {}".format(self._totalnumber,self._totaltime))
        print ("> [INFO] mean detection time: {}".format(self._meantime))
        print ("> [INFO] median detection time: {}".format(self._mediantime))
        print ("> [INFO] min detection time: {}".format(self._mintime))
        print ("> [INFO] max detection time: {}".format(self._maxtime))
        print ("> [INFO] std dev detection time: {}".format(self._stdtime))
        print ("> [INFO] resulting mean fps: {}".format(self._meanfps))
        print ("> [INFO] resulting median fps: {}".format(self._medianfps))


class WebcamVideoStream:
    """
    Class for Video Input frame capture
    Based on OpenCV VideoCapture
    adapted from https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    """
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.frame_counter = 1
        self.width = width
        self.height = height
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        #Debug stream shape
        self.real_width = int(self.stream.get(3))
        self.real_height = int(self.stream.get(4))
        print("> Start video stream with shape: {},{}".format(self.real_width,self.real_height))
        print("> Press 'q' to Exit")

    def start(self):
        # start the thread to read frames from the video stream
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                cv2.destroyAllWindows()
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            self.frame_counter += 1

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def isActive(self):
        # check if VideoCapture is still Opened
        return self.stream.isOpened

    def expanded(self):
        return np.expand_dims(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), axis=0)

    def resized(self,target_size):
        return cv2.resize(self.frame, target_size)

"""
Load Test Images
"""
def load_images(image_path,limit=None):
    if not limit:
        limit = float('inf')
    images = []
    for root, dirs, files in os.walk(image_path):
        for idx,file in enumerate(files):
            if idx >=limit:
                images.sort()
                return images
            if file.endswith(".jpg"):
                images.append(os.path.join(root, file))
    images.sort()
    return images
"""
Visualization functions
"""
def vis_text(image, string, pos):
    cv2.putText(image,string,(pos),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

def vis_detection(image, visualize, boxes, classes, scores, masks, category_index, det_interval, det_th, max_frames, fps=None):
    if visualize:
        vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=masks,
        use_normalized_coordinates=True,
        line_thickness=8)
        if fps:
            vis_text(image,"fps: {}".format(fps.fps_local()), (10,30))
        cv2.imshow('object_detection', image)
    elif not visualize and fps:
        # Exit after max frames if no visualization
        for box, score, _class in zip(boxes, scores, classes):
            if fps._glob_numFrames %det_interval==0 and score > det_th:
                label = category_index[_class]['name']
                print("label: {}\nscore: {}\nbox: {}".format(label, score, box))
    elif fps == "console":
        for box, score, _class in zip(boxes, scores, classes):
            if score > det_th:
                label = category_index[_class]['name']
                print("label: {}\nscore: {}\nbox: {}".format(label, score, box))
    # Exit Option
    if visualize:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    elif not visualize and fps:
        if fps._glob_numFrames >= max_frames:
            return False
    return True

def create_colormap(seg_map):
    """
    Takes A 2D array storing the segmentation labels.
    Returns A 2D array where each element is the color indexed
    by the corresponding element in the input label to the PASCAL color map.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap[seg_map]

"""
Tracker functions
"""
def conv_detect2track(box, width, height):
    # transforms normalized to absolut coords
    ymin, xmin, ymax, xmax = box
    ymin = ymin*height
    xmin = xmin*width
    ymax = ymax*height
    xmax = xmax*width
    boxwidth= xmax - xmin
    boxheight = ymax - ymin

    newbox = [xmin,ymin, boxwidth, boxheight]
    #newbox = map(int,newbox)
    return newbox

def conv_track2detect(box, width, height):
    # transforms absolut to normalized coords
    dw = 1./width
    dh = 1./height
    x, y, boxwidth, boxheight = box #map(float,box)
    xmin = x * dw
    ymin = y * dh
    xmax = (x+boxwidth) * dw
    ymax = (y+boxheight) * dh

    newbox = np.array([ymin,xmin,ymax,xmax])
    return newbox


class SessionWorker():
    """
    TensorFlow Session Thread for split_model spead Hack
    from https://github.com/naisy/realtime_object_detection/blob/master/lib/session_worker.py

     usage:
     before:
         results = sess.run([opt1,opt2],feed_dict={input_x:x,input_y:y})
     after:
         opts = [opt1,opt2]
         feeds = {input_x:x,input_y:y}
         woker = SessionWorker("TAG",graph,config)
         worker.put_sess_queue(opts,feeds)
         q = worker.get_result_queue()
         if q is None:
             continue
         results = q['results']
         extras = q['extras']

    extras: None or frame image data for draw. GPU detection thread doesn't wait result. Therefore, keep frame image data if you want to draw detection result boxes on image.
    """
    def __init__(self,tag,graph,config):
        self.lock = threading.Lock()
        self.sess_queue = Queue.Queue()
        self.result_queue = Queue.Queue()
        self.tag = tag
        t = threading.Thread(target=self.execution,args=(graph,config))
        t.setDaemon(True)
        t.start()
        return

    def execution(self,graph,config):
        self.is_thread_running = True
        try:
            with tf.Session(graph=graph,config=config) as sess:
                while self.is_thread_running:
                        while not self.sess_queue.empty():
                            q = self.sess_queue.get(block=False)
                            opts = q["opts"]
                            feeds= q["feeds"]
                            extras= q["extras"]
                            if feeds is None:
                                results = sess.run(opts)
                            else:
                                results = sess.run(opts,feed_dict=feeds)
                            self.result_queue.put({"results":results,"extras":extras})
                            self.sess_queue.task_done()
                        time.sleep(0.005)
        except:
            import traceback
            traceback.print_exc()
        self.stop()
        return

    def is_sess_empty(self):
        if self.sess_queue.empty():
            return True
        else:
            return False

    def put_sess_queue(self,opts,feeds=None,extras=None):
        self.sess_queue.put({"opts":opts,"feeds":feeds,"extras":extras})
        return

    def is_result_empty(self):
        if self.result_queue.empty():
            return True
        else:
            return False

    def get_result_queue(self):
        result = None
        if not self.result_queue.empty():
            result = self.result_queue.get(block=False)
            self.result_queue.task_done()
        return result

    def stop(self):
        self.is_thread_running=False
        with self.lock:
            while not self.sess_queue.empty():
                q = self.sess_queue.get(block=False)
                self.sess_queue.task_done()
        return

class TimeLiner:
    """
    TimeLiner Class for creating multiple session json timing files
    """
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
			json.dump(self._timeline_dict, f)


class RosDetectionPublisher:
    """
    Class for publishing Ros messages
    Not yet used
    """
    def __init__(self):
        self.objDetPub = rospy.Publisher('objectDetection', Detection, queue_size=10)

    def publish(self, boxes, scores, classes, num, image_shape, category_index):
        #obj = []
        # create an empty python array
        msg = Detection()
        for i in range(boxes.shape[0]):
            if scores[i] > 0.5:
                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]['name']
                else:
                    class_name = 'N/A'
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            (left, right, top, bottom) = (int(xmin * image_shape[1]), int(xmax * image_shape[1]), int(ymin * image_shape[0]), int(ymax * image_shape[0]))
            #obj.append([class_name, int(100*scores[i]), left, top, right, bottom])
            display_str = '##\nnumber {} {}: {}% at image coordinates (({}, {}) to ({}, {}))\n##'.format(i, class_name, int(100*scores[i]), left, top, right, bottom)
            print(display_str)
            # fill array with data
            object = Object()
            object.class_name = class_name
            object.certainty = int(100*scores[i])
            object.p1.x = left
            object.p1.y = top
            object.p2.x = right
            object.p2.y = bottom
            msg.objects.append(object)
            #print('OBJECT', object)

        # publish msg
        #for i in range(len(msg.objects)):
        #    print('MSG ',i, msg.objects[i])
        self.objDetPub.publish(msg)
