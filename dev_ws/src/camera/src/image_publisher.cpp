#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"
#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>

#define CAMERA_INDEX 0

using namespace std::chrono_literals;

class ImagePublisher : public rclcpp::Node
{
public:
  ImagePublisher () : Node("image_publisher")
  {
    VideoCapture_ = cv::VideoCapture(CAMERA_INDEX);
    if (!VideoCapture_.isOpened())
    {
      throw std::runtime_error("Unable to open camera");
    }
    Publisher_ = this->create_publisher<sensor_msgs::msg::Image>("raw_image", 10);
    Timer_ = this->create_wall_timer(33.33ms, std::bind(&ImagePublisher::TimerCallback, this));
  }

private:
  void TimerCallback()
  {
    VideoCapture_ >> Frame_;

    auto const Message = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", Frame_).toImageMsg();

    Publisher_->publish(*Message.get());

    RCLCPP_INFO(this->get_logger(), "Image is being published");
  }

  cv::VideoCapture VideoCapture_;
  cv::Mat Frame_;
  rclcpp::TimerBase::SharedPtr Timer_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr Publisher_;
};

int main (int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImagePublisher>();
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
