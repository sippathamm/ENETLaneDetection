from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('steering_predictor'),
        'config',
        'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='steering_predictor',
            executable='steering_predictor',
            parameters=[params],
            output='screen',
        )
    ])
