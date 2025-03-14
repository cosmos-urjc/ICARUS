from setuptools import setup
import os
from glob import glob

package_name = 'test_jetson'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='A ROS2 package with a simple publisher and subscriber',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publisher_node = test_jetson.publisher_node:main',
            'subscriber_node = test_jetson.subscriber_node:main',
        ],
    },
)
