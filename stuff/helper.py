#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:53:52 2017

@author: GustavZ
"""
import datetime
import cv2
from threading import Thread
import rospy
from ros import Detection, Object

class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()
    
    
class FPS2:
    def __init__(self, interval):
        self._glob_start = None
        self._glob_end = None
        self._glob_numFrames = 0
        self._local_start = None
        self._local_numFrames = 0
        self._interval = interval

    def start(self):
        self._glob_start = datetime.datetime.now()
        self._local_start = self._glob_start
        return self

    def stop(self):
        self._glob_end = datetime.datetime.now()

    def update(self):
        curr_time = datetime.datetime.now()
        curr_local_elapsed = (curr_time - self._local_start).total_seconds()
        self._glob_numFrames += 1
        self._local_numFrames += 1
        if curr_local_elapsed > self._interval:
          print("FPS: ", round(self._local_numFrames / curr_local_elapsed,1))
          self._local_numFrames = 0
          self._local_start = curr_time

    def elapsed(self):
        return (self._glob_end - self._glob_start).total_seconds()

    def fps(self):
        return self._glob_numFrames / self.elapsed()
    
    
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
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

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        
    def isActive(self):
        # check if VideoCapture is still Opened
        return self.stream.isOpened
    
class RosDetectionPublisher:
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
