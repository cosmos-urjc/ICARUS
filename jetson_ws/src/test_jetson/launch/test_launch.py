from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='test_jetson',
            executable='publisher_node',
            name='simple_publisher'
        ),
        Node(
            package='test_jetson',
            executable='subscriber_node',
            name='simple_subscriber'
        )
    ])
