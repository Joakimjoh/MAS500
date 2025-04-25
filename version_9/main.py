"""Third-Party Libraries"""
import rclpy
import cv2

"""Internal Modules"""
from dual_arm_xs import InterbotixManipulatorXS
from process import Process
from camera import Camera
from tag import Tag

# Initialize Ros Node once for shared use
rclpy.init()

# Allow streaming in thread
cv2.startWindowThread()

# Initialize camera
camera = Camera()

# Create the Interbotix manipulators for the two robots
bot_left = InterbotixManipulatorXS(
    robot_model='wx250s',
    group_name='arm',
    robot_name='left_arm',
    gripper_name='gripper',
    node_name='node1',
    tag=Tag(camera, "left") # Define left tag
)
bot_right = InterbotixManipulatorXS(
    robot_model='wx250s',
    group_name='arm',
    robot_name='right_arm',
    gripper_name='gripper',
    node_name='node2',
    tag= Tag(camera, "right") # Define right tag
)

print("Press Enter to start")
process = Process(camera, bot_left, bot_right)
while True:
    if camera.key == 13:  # 13 is the Enter key
        process.unfold()
