"""Third-Party Libraries"""
import rclpy
import cv2

"""Internal Modules"""
from dual_arm_xs import InterbotixManipulatorXS
from process import Process
from camera import Camera

# Initialize Ros Node once for shared use
rclpy.init()

# Allow streaming in thread
cv2.startWindowThread()

# Initialize camera
camera = Camera()

# Get apriltags
print("Looking for tags")
tags = camera.get_tag_orientation()

# Create the Interbotix manipulators for the two robots
bot_left = InterbotixManipulatorXS(
    robot_model='wx250s',
    group_name='arm',
    robot_name='left_arm',
    gripper_name='gripper',
    node_name='node1',
    tag=tags[0] # Leftmost
)
bot_right = InterbotixManipulatorXS(
    robot_model='wx250s',
    group_name='arm',
    robot_name='right_arm',
    gripper_name='gripper',
    node_name='node2',
    tag=tags[1] # Rightmost
)

print("Press 'f' to start")
while True:
    process = Process(camera, bot_left, bot_right)
    if cv2.waitKey(1) & 0xFF == ord('f'):
        process.start()
