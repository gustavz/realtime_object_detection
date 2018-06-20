import rospy
import time
from cv_bridge import CvBridge, CvBridgeError
from objdetection.msg import Detection, Object
from sensor_msgs.msg import RegionOfInterest, Image

class DetectionPublisher(object):
    """
    Publish ROS detection messages
    """
    def __init__(self):
        self.DetPub = rospy.Publisher('Detection', Detection, queue_size=10)

    def publish(self, boxes, scores, classes, num, image_shape, category_index, masks=None, fps=0):
        # init detection message
        msg = Detection()
        for i in range(boxes.shape[0]):
            if scores[i] > 0.5:
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                box = RegionOfInterest()
                box.x_offset = np.asscalar(xmin)
                box.y_offset = np.asscalar(ymin)
                box.height = np.asscalar(ymax - ymin)
                box.width = np.asscalar(xmax - xmin)

                display_str = '##\nnumber {} {}: {}% at image coordinates (({}, {}) to ({}, {}))\n##'.format(i, class_name, int(100*scores[i]), left, top, right, bottom)
                print(display_str)

                # fill detection message with objects
                obj = Object()
                obj.box = box
                obj.class_name = class_name
                obj.score = int(100*scores[i])
                obj.mask = masks[i]
                obj.fps = fps
                msg.objects.append(obj)

        # publish detection message
        self.DetPub.publish(msg)

class ROSInput(object):
    """
    Capture video via ROS topic
    """
    def __init__(self, input):
        self._image = None
        self._bridge = CvBridge()
        rospy.Subscriber(input, Image, self.imageCallback)

    def imageCallback(self, data):
        try:
            image_raw = self._bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self._image = image_raw

    def isActive(self):
        return True

    @property
    def image(self):
        return self._image

    def cleanup(self):
        pass

    def isEnabled(self):
        return self._enabled
