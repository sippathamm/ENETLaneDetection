import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from model.predictor import Predictor
import numpy as np
import cv2


class SteeringPredictor(Node):
    def __init__(self):
        super().__init__('steering_predictor')

        self.declare_parameter('cmd_steering_topic', 'cmd_servo')
        self.declare_parameter('image_topic', 'raw_image')
        self.declare_parameter('turn_right_steering_angle_rad', -0.27)
        self.declare_parameter('turn_left_steering_angle_rad', 0.3)
        self.declare_parameter('turn_right_cmd_steering', -1000)
        self.declare_parameter('turn_left_cmd_steering', 1000)
        self.declare_parameter('verbose', 1)

        self.CMD_STEERING_TOPIC = self.get_parameter('cmd_steering_topic').get_parameter_value().string_value
        self.IMAGE_TOPIC = self.get_parameter('image_topic').get_parameter_value().string_value
        self.TURN_RIGHT_STEERING_ANGLE_RAD = self.get_parameter('turn_right_steering_angle_rad').get_parameter_value().double_value
        self.TURN_LEFT_STEERING_ANGLE_RAD = self.get_parameter('turn_left_steering_angle_rad').get_parameter_value().double_value
        self.TURN_RIGHT_CMD_STEERING = self.get_parameter('turn_right_cmd_steering').get_parameter_value().integer_value
        self.TURN_LEFT_CMD_STEERING = self.get_parameter('turn_left_cmd_steering').get_parameter_value().integer_value
        self.VERBOSE = self.get_parameter('verbose').get_parameter_value().integer_value

        self.publisher = self.create_publisher(Int16, self.CMD_STEERING_TOPIC, 10)
        self.publisher_ = self.create_publisher(Image, 'steering_debug_image', 10)
        self.subscriber = self.create_subscription(Image, self.IMAGE_TOPIC, self.image_callback, 10)
        self.bridge = CvBridge()
        self.steering_wheel_image = cv2.imread('/home/parallels/dev_ws/src/lane_keeping/lane_keeping/steering-wheel-14-256.png', cv2.IMREAD_UNCHANGED)
        self.steering_wheel_image = cv2.resize(self.steering_wheel_image, (100, 100))
        self.model = Predictor(input_shape=(128, 64, 3))

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        background = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frame = cv2.resize(frame, (512, 256))
        cutoff_frame = frame[136:, :]
        resized_frame = cv2.resize(cutoff_frame, (128, 64))

        x = resized_frame.reshape(1, 64, 128, 3)
        prediction = self.model(x=x, verbose=self.VERBOSE)
        prediction = min(max(prediction[0], -self.TURN_RIGHT_STEERING_ANGLE_RAD), self.TURN_LEFT_STEERING_ANGLE_RAD)

        steering_angle_rad = Int16()
        steering_angle_rad.data = int((prediction - self.TURN_LEFT_STEERING_ANGLE_RAD) *
                                      (self.TURN_RIGHT_CMD_STEERING - self.TURN_LEFT_CMD_STEERING) /
                                      (self.TURN_RIGHT_STEERING_ANGLE_RAD - self.TURN_LEFT_STEERING_ANGLE_RAD) +
                                      self.TURN_LEFT_CMD_STEERING)
        self.get_logger().info('Steering angle: %d' % steering_angle_rad.data)
        self.publisher.publish(steering_angle_rad)

        (h, w) = self.steering_wheel_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, np.rad2deg(prediction) * 2, 1.0)
        rotated_steering_wheel = cv2.warpAffine(self.steering_wheel_image, rotation_matrix, (w, h),
                                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        self.add_transparent_image(background, rotated_steering_wheel, 30, 50)
        output_image = self.bridge.cv2_to_imgmsg(background, 'bgr8')
        self.publisher_.publish(output_image)

    @staticmethod
    def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
        bg_h, bg_w, bg_channels = background.shape
        fg_h, fg_w, fg_channels = foreground.shape

        assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
        assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

        # center by default
        if x_offset is None: x_offset = (bg_w - fg_w) // 2
        if y_offset is None: y_offset = (bg_h - fg_h) // 2

        w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
        h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

        if w < 1 or h < 1: return

        # clip foreground and background images to the overlapping regions
        bg_x = max(0, x_offset)
        bg_y = max(0, y_offset)
        fg_x = max(0, x_offset * -1)
        fg_y = max(0, y_offset * -1)
        foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
        background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

        # separate alpha and color channels from the foreground image
        foreground_colors = foreground[:, :, :3]
        alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

        # construct an alpha_mask that matches the image shape
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

        # combine the background with the overlay image weighted by alpha
        composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

        # overwrite the section of the background image that has been updated
        background[bg_y:bg_y + h, bg_x:bg_x + w] = composite



def main(args=None):
    rclpy.init(args=args)
    node = SteeringPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
