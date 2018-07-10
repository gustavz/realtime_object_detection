import rospy
from cv_bridge import CvBridge, CvBridgeError
from objdetection.msg import Detection, Object, Segmentation
from helper import InputStream
from sensor_msgs.msg import RegionOfInterest, Image

class DetectionPublisher(object):
    """
    Publish ROS detection messages
    """
    def __init__(self):
        self.DetPub = rospy.Publisher('Detection', Detection, queue_size=10)
        self._bridge = CvBridge()

    def publish(self,boxes,scores,classes,num,category_index,image_shape,masks=None,fps=0):
        # init detection message
        msg = Detection()
        for i in range(boxes.shape[0]):
            if scores[i] > 0.5:
                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]['name']
                else:
                    class_name = 'N/A'
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                (left, right, top, bottom) = (int(xmin * image_shape[1]), int(xmax * image_shape[1]), int(ymin * image_shape[0]), int(ymax * image_shape[0]))
                box = RegionOfInterest()
                box.x_offset = left        # Leftmost pixel of the ROI
                box.y_offset = top         # Topmost pixel of the ROI
                box.height = bottom - top  # Height of ROI
                box.width = right - left   # Width of ROI
                # fill detection message with objects
                obj = Object()
                obj.box = box
                obj.class_name = class_name
                obj.score = int(100*scores[i])
                if masks is not None:
                    obj.mask = self._bridge.cv2_to_imgmsg(masks[i], encoding="passthrough")
                msg.objects.append(obj)
        msg.fps = fps
        # publish detection message
        self.DetPub.publish(msg)

class SegmentationPublisher(object):
    """
    Publish ROS Segmentation messages
    """
    def __init__(self):
        self.SegPub = rospy.Publisher('Segmentation', Segmentation, queue_size=10)
        self._bridge = CvBridge()

    def publish(self,boxes,labels,seg_map,image_shape,fps=0):
        # init detection message
        msg = Segmentation()
        boxes = []
        for i in range(boxes.shape[0]):
            class_name = labels[i]
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            (left, right, top, bottom) = (int(xmin * image_shape[1]), int(xmax * image_shape[1]), int(ymin * image_shape[0]), int(ymax * image_shape[0]))
            box = RegionOfInterest()
            box.x_offset = left        # Leftmost pixel of the ROI
            box.y_offset = top         # Topmost pixel of the ROI
            box.height = bottom - top  # Height of ROI
            box.width = right - left   # Width of ROI
            # fill segmentation message
            msg.boxes.append(box)
            msg.class_names.append(class_name)
        if masks is not None:
            msg.seg_map = self._bridge.cv2_to_imgmsg(seg_map, encoding="passthrough")
        msg.fps = fps
        # publish detection message
        self.SegPub.publish(msg)

class ROSStream(InputStream):
    """
    Capture video via ROS topic
    """
    def __init__(self, input):
        super(ROSStream, self).__init__()
        self._bridge = CvBridge()
        rospy.Subscriber(input, Image, self.imageCallback)

    def imageCallback(self, data):
        try:
            image_raw = self._bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.frame = image_raw
