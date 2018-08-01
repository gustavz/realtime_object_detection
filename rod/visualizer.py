#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: www.github.com/GustavZ

"""
import collections
import numpy as np
import cv2
import random
from rod.config import Config

class Visualizer(object):
    """
    Visualizer Class to handle all kind of detection visualizations on an image
    """
    STANDARD_COLORS = [
        (0, 0, 0),#Black
        (60, 180, 75),#Green
        (230, 25, 75),#Red
        (255, 225, 25),#Yellow
        (0, 130, 200),#Blue
        (245, 130, 48),#Orange
        (145, 30, 180),#Purple
        (70, 240, 240),#Cyan
        (0, 128, 128),#Teal
        (210, 245, 60),#Lime
        (250, 190, 190),#Pink
        (240, 50, 230),#Magenta
        (230, 190, 255),#Lavender
        (170, 110, 40),#Brown
        (255, 250, 200),#Beige
        (0, 252, 124), # LawnGreen
        (128, 0, 0),#Maroon
        (170, 255, 195),#Mint
        (128, 128, 0),#Olive
        (255, 215, 180),#Coral
        (0, 0, 128),#Navy
        (60, 180, 75),#Green
        (230, 25, 75),#Red
        (255, 225, 25),#Yellow
        (0, 130, 200),#Blue
        (245, 130, 48),#Orange
        (145, 30, 180),#Purple
        (70, 240, 240),#Cyan
        (0, 128, 128),#Teal
        (210, 245, 60),#Lime
        (250, 190, 190),#Pink
        (240, 50, 230),#Magenta
        (230, 190, 255),#Lavender
        (170, 110, 40),#Brown
        (255, 250, 200),#Beige
        (0, 252, 124), # LawnGreen
        (128, 0, 0),#Maroon
        (170, 255, 195),#Mint
        (128, 128, 0),#Olive
        (255, 215, 180),#Coral
        (0, 0, 128),#Navy
        (60, 180, 75),#Green
        (230, 25, 75),#Red
        (255, 225, 25),#Yellow
        (0, 130, 200),#Blue
        (245, 130, 48),#Orange
        (145, 30, 180),#Purple
        (70, 240, 240),#Cyan
        (0, 128, 128),#Teal
        (210, 245, 60),#Lime
        (250, 190, 190),#Pink
        (240, 50, 230),#Magenta
        (230, 190, 255),#Lavender
        (170, 110, 40),#Brown
        (255, 250, 200),#Beige
        (0, 252, 124), # LawnGreen
        (128, 0, 0),#Maroon
        (170, 255, 195),#Mint
        (128, 128, 0),#Olive
        (255, 215, 180),#Coral
        (0, 0, 128),#Navy
        (60, 180, 75),#Green
        (230, 25, 75),#Red
        (255, 225, 25),#Yellow
        (0, 130, 200),#Blue
        (245, 130, 48),#Orange
        (145, 30, 180),#Purple
        (70, 240, 240),#Cyan
        (0, 128, 128),#Teal
        (210, 245, 60),#Lime
        (250, 190, 190),#Pink
        (240, 50, 230),#Magenta
        (230, 190, 255),#Lavender
        (170, 110, 40),#Brown
        (255, 250, 200),#Beige
        (0, 252, 124), # LawnGreen
        (128, 0, 0),#Maroon
        (170, 255, 195),#Mint
        (128, 128, 0),#Olive
        (255, 215, 180),#Coral
        (0, 0, 128),#Navy
        (60, 180, 75),#Green
        ]

    STANDARD_COLORS_ARRAY = np.asarray(STANDARD_COLORS).astype(np.uint8)

    def __init__(self,config):
        self.config = config
        # private params
        self._line_thickness = 2
        self._font_thickness = 1
        self._font_face = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.5
        self._line_type = cv2.LINE_AA
        self.cur_frame = 0
        self._shuffle = False

        # public params
        self.image = None
        self.stopped = False


    def _shuffle_colors(self):
        """
        shuffles STANDARD_COLORS
        """
        np.random.shuffle(self.STANDARD_COLORS_ARRAY)
        np.random.shuffle(self.STANDARD_COLORS)


    def _draw_bounding_box_on_image(self,
                                   ymin,
                                   xmin,
                                   ymax,
                                   xmax,
                                   color=(0, 0, 255),
                                   display_str_list=(),
                                   use_normalized_coordinates=True):

      im_height, im_width = self.image.shape[:2]
      if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
      else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

      ####################
      # draw objectbox
      ####################
      points = np.array([[left, top], [left, bottom], [right, bottom], [right, top], [left, top]])
      cv2.polylines(self.image, np.int32([points]),
                    isClosed=False, thickness=self._line_thickness, color=color, lineType = self._line_type)

      ####################
      # calculate str width and height
      ####################
      display_str_heights = [cv2.getTextSize(text=ds, fontFace=self._font_face, fontScale=self._font_scale,
                            thickness=self._font_thickness)[0][1] for ds in display_str_list]
      total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

      if top > total_display_str_height:
        text_bottom = top
      else:
        text_bottom = bottom + total_display_str_height

      ####################
      # draw textbox and text
      ####################
      for display_str in display_str_list[::-1]:
        #
        [(text_width, text_height), baseLine] = cv2.getTextSize(text=display_str, fontFace=self._font_face,
                                                                fontScale=self._font_scale, thickness=self._font_thickness)
        margin = np.ceil(0.05 * text_height)

        cv2.rectangle(self.image, (int(left), int(text_bottom - 3 * baseLine - text_height - 2 * margin)),
                        (int(left + text_width), int(text_bottom - baseLine)), color=color, thickness=-1)
        cv2.putText(self.image, display_str, org=(int(left + margin),
                    int(text_bottom - text_height - margin)),
                    fontFace=self._font_face, fontScale=self._font_scale,
                    thickness=self._font_thickness, color=(0, 0, 0))

        text_bottom -= text_height - 2 * margin


    def _draw_mask_on_image(self, mask):
      """
      Draws mask on image.
      """
      mask = self.STANDARD_COLORS_ARRAY[mask]
      cv2.addWeighted(mask,self.config.ALPHA,self.image,1.0,0,self.image)



    def _visualize_boxes_and_labels_on_image(
        self,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=20,
        min_score_thresh=0.5):
        """
        visualizes binary classes, scores, masks and bounding boxes on image
        """
        # Create a display string (and color) for every box location, group any boxes
        # that correspond to the same location.
        box_to_display_str_map = collections.defaultdict(list)
        box_to_color_map = collections.defaultdict(str)
        box_to_instance_masks_map = {}

        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                if instance_masks is not None:
                    box_to_instance_masks_map[box] = instance_masks[i]
                display_str = ''
                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]['name']
                else:
                    class_name = 'N/A'
                display_str = str(class_name)
                if not display_str:
                    if scores is None:
                        display_str = '?%'
                    else:
                        display_str = '{}%'.format(int(100*scores[i]))
                else:
                    if scores is None:
                        display_str = '{}: ?%'.format(display_str)
                    else:
                        display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
                box_to_display_str_map[box].append(display_str)
                box_to_color_map[box] = self.STANDARD_COLORS[classes[i] % len(self.STANDARD_COLORS)]
        first = True
        mask = None
        # Draw all boxes onto image.
        for idx,(box, color) in enumerate(box_to_color_map.items()):
            ymin, xmin, ymax, xmax = box
            if instance_masks is not None:

                if self._shuffle:
                    # draw mask for each object
                    self._draw_mask_on_image(box_to_instance_masks_map[box]*(idx+1))
                else:
                    # stack all masks and draw one big mask
                    if first:
                        first = False
                        mask = box_to_instance_masks_map[box]*(idx+1)
                    else:
                        mask = np.bitwise_or(mask, box_to_instance_masks_map[box])

            self._draw_bounding_box_on_image(
                ymin,
                xmin,
                ymax,
                xmax,
                color=color,
                display_str_list=box_to_display_str_map[box],
                use_normalized_coordinates=use_normalized_coordinates)

        # Draw Masks on Image (only one color for all masks)
        if mask is not None and not self._shuffle:
            self._draw_mask_on_image(mask)


    def _draw_text_on_image(self, string, position, color = (77, 255, 9)):
        """
        visualizes colored text on image on given position with OpenCV
        """
        cv2.putText(self.image, string, (position),
                    fontFace = self._font_face, fontScale = self._font_scale,
                    color = color, thickness = self._font_thickness)


    def _exit_visualization(self,millis=1):
        """
        - sets exit variable on key 'q'
        - saves screenshot on key 's'
        """
        k = cv2.waitKey(millis) & 0xFF
        if k  == ord('q'): # wait for 'q' key to exit
            print("> User exit request")
            self.stopped = True
        elif k == ord('s'): # wait for 's' key to save screenshot
            self.save_image()



    def _exit_print(self):
        """
        sets exit variable if max frames are reached
        """
        if self.cur_frame >= self.config.MAX_FRAMES:
            self.stopped = True


    def _print_detection(self,boxes,scores,classes,category_index):
        """
        prints detection result above threshold to console
        """
        for box, score, _class in zip(boxes, scores, classes):
            if self.cur_frame%self.config.PRINT_INTERVAL==0 and score > self.config.PRINT_TH:
                label = category_index[_class]['name']
                print("label: {}\nscore: {}\nbox: {}".format(label, score, box))


    def _draw_single_box_on_image(self,box,label,id):
        """
        draws single box and label on image
        """
        p1 = (box[1], box[0])
        p2 = (box[3], box[2])
        if self.config.DISCO_MODE:
            color = random.choice(self.STANDARD_COLORS)
        else:
            color = self.STANDARD_COLORS[id]
        cv2.rectangle(self.image, p1, p2, color, 2)
        self._draw_text_on_image(label,(p1[0],p1[1]-10),color)

    def visualize_detection(self,
                            image,
                            boxes,
                            classes,
                            scores, #labels
                            masks,  #seg_map
                            fps='N/A',
                            category_index=None,
                            shuffle=False):
        """
        visualization function for object_detection
        """
        self._shuffle = shuffle
        self.image = image
        self.cur_frame += 1
        if self.config.DISCO_MODE:
            self._shuffle_colors()
        # object_detection workaround
        if self.config.MODEL_TYPE is 'od':
            if self.config.VISUALIZE:
                self._visualize_boxes_and_labels_on_image(
                boxes,
                classes,
                scores,
                category_index,
                instance_masks=masks,
                min_score_thresh=self.config.PRINT_TH,
                use_normalized_coordinates=True)
                if self.config.VIS_FPS:
                    self._draw_text_on_image("fps: {}".format(fps), (5,20))
                self.show_image()
                self._exit_visualization()
            else:
                self._print_detection(boxes,scores,classes,category_index)
                self._exit_print()
        # deeplab workaround
        elif self.config.MODEL_TYPE is 'dl':
            if self.config.VISUALIZE:
                for box,id,label in zip(boxes,classes,scores): #classes=ids,scores=labels
                    self._draw_single_box_on_image(box,label,id)
                if masks is not None:
                    self._draw_mask_on_image(masks) #masks = seg_map
                if self.config.VIS_FPS:
                    self._draw_text_on_image("fps: {}".format(fps),(5,20))
                self.show_image()
                self._exit_visualization()
            else:
                self._exit_print()
        # write detection to image file
        if self.config.SAVE_RESULT:
            self.save_image()

        return self.image

    def show_image(self):
        """
        shows image in OpenCV Window
        """
        cv2.imshow(self.config.DISPLAY_NAME, self.image)

    def isActive(self):
        return not self.stopped

    def start(self):
        print("> Press 'q' to Exit")
        self.cur_frame = 0
        self.stopped = False
        return self

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()

    def expand_and_convertRGB_image(self,image):
        return np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)

    def convertRGB_image(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def save_image(self):
        save_path = '{}/detection{}_{}.jpg'.format(self.config.RESULT_PATH,self.cur_frame,self.config.DISPLAY_NAME)
        cv2.imwrite(save_path,self.image)
        print("> Saving detection to: {}".format(save_path))

    def resize_image(self,image,shape):
        return cv2.resize(image,shape)
