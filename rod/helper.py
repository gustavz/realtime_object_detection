#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:53:52 2017

@author: www.github.com/GustavZ
"""
# python 2 compability
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import cv2
import threading
import time
import numpy as np
import tensorflow as tf
import os
import json
import sys
if sys.version_info[0] == 2:
    import Queue
elif sys.version_info[0] == 3:
    import queue as Queue
from tensorflow.python.client import timeline


class FPS(object):
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
        self._log =[]

    def start(self):
        self._glob_start = datetime.datetime.now()
        self._local_start = self._glob_start
        return self

    def stop(self):
        self._glob_end = datetime.datetime.now()
        print('> [INFO] elapsed frames (total): {}'.format(self._glob_numFrames))
        print('> [INFO] elapsed time (total): {:.2f}'.format(self.elapsed()))
        print('> [INFO] approx. FPS: {:.2f}'.format(self.fps()))

    def update(self):
        self._first = True
        self._curr_time = datetime.datetime.now()
        self._curr_local_elapsed = (self._curr_time - self._local_start).total_seconds()
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


class Timer(object):
    """
    Timer class for benchmark test purposes
    Usage: start -> tic -> (tictic -> tic ->) toc -> stop
    Alternative: start -> update -> stop
    """
    def __init__(self):
        self._tic = None
        self._tictic = None
        self._toc = None
        self._time = 1
        self._cache = []
        self._log = []
        self._first = True

    def start(self):
        return self

    def tic(self):
        self._tic = datetime.datetime.now()

    def tictic(self):
        self._tictic = datetime.datetime.now()
        self._cache.append((self._tictic-self._tic).total_seconds())
        self._tic = self._tictic

    def toc(self):
        self._toc = datetime.datetime.now()
        self._time = (self._toc-self._tic).total_seconds() + np.sum(self._cache)
        self._log.append(self._time)
        self._cache = []

    def update(self):
        if self._first:
            self._tic = datetime.datetime.now()
            self._toc = self._tic
            self._first = False
            self._frame = 1
        else:
            self._frame += 1
            self._tic = datetime.datetime.now()
            self._time = (self._tic-self._toc).total_seconds()
            self._log.append(self._time)
            self._toc = self._tic
            if len(self._log)>1000:
                self.stop()
                self._log = []

    def get_frame(self):
        return len(self._log)

    def get_fps(self):
        return round(1./self._time,1)

    def _calc_stats(self):
        self._totaltime = np.sum(self._log)
        self._totalnumber = len(self._log)
        self._meantime = np.mean(self._log)
        self._mediantime = np.median(self._log)
        self._mintime = np.min(self._log)
        self._maxtime = np.max(self._log)
        self._stdtime = np.std(self._log)
        self._meanfps = 1./np.mean(self._log)
        self._medianfps = 1./np.median(self._log)

    def stop(self):
        self._calc_stats()
        print ("> [INFO] total detection time for {} images: {}".format(self._totalnumber,self._totaltime))
        print ("> [INFO] mean detection time: {}".format(self._meantime))
        print ("> [INFO] median detection time: {}".format(self._mediantime))
        print ("> [INFO] min detection time: {}".format(self._mintime))
        print ("> [INFO] max detection time: {}".format(self._maxtime))
        print ("> [INFO] std dev detection time: {}".format(self._stdtime))
        print ("> [INFO] resulting mean fps: {}".format(self._meanfps))
        print ("> [INFO] resulting median fps: {}".format(self._medianfps))


class InputStream(object):
    """
    input stream base class
    """
    def __init__(self):
        self.real_height = 0
        self.real_width = 0
        self.stopped = False
        self.frame = None

    def start(self):
        self.stopped = False
        return self

    def stop(self):
        self.stopped = True

    def isActive(self):
        return not self.stopped

    def read(self):
        return self.frame


class ImageStream(InputStream):
    """
    Test Image handling class
    """
    def __init__(self,image_path,limit=None,image_shape=None):
        super(ImageStream, self).__init__()
        self.frame_shape = image_shape
        self.frame_path = image_path
        self.frames = []
        self.limit = limit
        if not limit:
            self.limit = float('inf')

    def start(self):
        self.load_images()
        self.stopped = False
        return self

    def load_images(self):
        for root, dirs, files in os.walk(self.frame_path):
            for idx,file in enumerate(files):
                if idx >=self.limit:
                    self.frames.sort()
                    return self.frames
                if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".JPEG") or file.endswith(".png"):
                    self.frames.append(os.path.join(root, file))
        self.frames.sort()

    def read(self):
        if self.frame_shape is not None:
            self.frame = cv2.resize(cv2.imread(self.frames.pop()),(self.frame_shape[:2]))
        else:
            self.frame = cv2.imread(self.frames.pop())
        self.real_height,self.real_width,_ = self.frame.shape
        return self.frame

    def isActive(self):
        if self.frames and not self.stopped:
            return True
        else:
            return False


class VideoStream(InputStream):
    """
    Class for Video Input frame capture
    Based on OpenCV VideoCapture
    adapted from https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    """
    def __init__(self, src, width, height):
        super(VideoStream, self).__init__()
        # initialize the video camera stream and read the first frame
        # from the stream
        self.frame_counter = 1
        self.width = width
        self.height = height
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        (self.grabbed, self.frame) = self.stream.read()
        #Debug stream shape
        self.real_width = int(self.stream.get(3))
        self.real_height = int(self.stream.get(4))

    def start(self):
        # start the thread to read frames from the video stream
        print("> Start video stream with shape: {},{}".format(self.real_width,self.real_height))
        threading.Thread(target=self.update, args=()).start()
        self.stopped = False
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            self.frame_counter += 1

    def isActive(self):
        # check if VideoCapture is still Opened
        return self.stream.isOpened

    def expanded(self):
        return np.expand_dims(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), axis=0)

    def resized(self,target_size):
        return cv2.resize(self.frame, target_size)


"""
Tracker converter functions
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


def get_model_list(models_path):
    """
    Returns List of Model names in models_path
    """
    for root, dirs, files in os.walk(models_path):
        if root.count(os.sep) - models_path.count(os.sep) == 0:
            for idx,model in enumerate(dirs):
                model_list=[]
                model_list.append(dirs)
                model_list = np.squeeze(model_list)
                model_list.sort()
    print("> Loaded following sequention of models: \n{}".format(model_list))
    return model_list


def check_if_optimized_model(model_dir):
    """
    check if there is an optimized graph in the model_dir
    """
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if 'optimized' in file:
                return True
                print('> found: optimized graph')
    return False


class SessionWorker(object):
    """
    TensorFlow Session Thread for split_model speed Hack

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

    extras: None or input image frame.
    	(reason: GPU detection thread does not wait for result.
    		Therefore, keep frame if VISUALIZE=TRUE.)
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


class TimeLiner(object):
    """
    TimeLiner Class for creating multiple session json timing files
    modified from: https://github.com/ikhlestov/tensorflow_profiling
    """
    def __init__(self):
        self._timeline_dict = None

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

    def write_timeline(self,step_stats,file_name):
        fetched_timeline = timeline.Timeline(step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open(file_name, 'w') as f:
        	f.write(chrome_trace)
