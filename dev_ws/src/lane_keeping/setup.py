from setuptools import find_packages, setup
import glob
import os

package_name = 'lane_keeping'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 'model'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob.glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='parallels',
    maintainer_email='parallels@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'steering_predictor = lane_keeping.steering_predictor:main',
            'lane_detector = lane_keeping.lane_detector:main',
        ],
    },
)
