import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from model.detector import Detector
import numpy as np
import cv2


class LaneDetector(Node):
    def __init__(self):
        super().__init__('lane_detector')

        self.declare_parameter('image_topic', 'raw_image')
        self.declare_parameter('verbose', 1)

        self.IMAGE_TOPIC = self.get_parameter('image_topic').get_parameter_value().string_value
        self.VERBOSE = self.get_parameter('verbose').get_parameter_value().integer_value

        self.publisher = self.create_publisher(Image, 'lane_detection', 10)
        self.subscriber = self.create_subscription(Image, self.IMAGE_TOPIC, self.image_callback, 10)
        self.bridge = CvBridge()
        self.model = Detector(input_shape=(512, 256, 3))

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame = cv2.resize(frame, (512, 256))

        x = frame.reshape(1, 256, 512, 3)
        prediction = self.model(x=x, verbose=self.VERBOSE)
        prediction = (prediction >= 0.5).astype(np.uint8) * 255

        message = self.bridge.cv2_to_imgmsg(prediction, 'mono8')
        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
