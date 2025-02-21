"""Standard Library"""
import threading

"""Third-Party Libraries"""
import rclpy
import cv2

"""Internal Modules"""
from dual_arm_xs import InterbotixManipulatorXS
from camera import Camera
from camera_detection import detect_stretched, get_left_right_point
from arm_control import step1

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
    tag=tags[0] # Leftmost orientation
)
bot_right = InterbotixManipulatorXS(
    robot_model='wx250s',
    group_name='arm',
    robot_name='right_arm',
    gripper_name='gripper',
    node_name='node2',
    tag=tags[1] # Rightmost orientation
)

while True:
    # Step 1 grap points and lay object flat
    if cv2.waitKey(1) & 0xFF == ord('f'):
        # Detect red objects and get their coordinates
        point_left_pixel, point_right_pixel = get_left_right_point(camera.frame)
        if point_right_pixel is not None and point_right_pixel is not None:
            point_left = camera.pixel_to_coordsystem(bot_left.rotation_vector, bot_left.translation_vector, point_left_pixel)
            point_right = camera.pixel_to_coordsystem(bot_right.rotation_vector, bot_right.translation_vector, point_right_pixel)

            detect_stretched(camera.frame)

            thread5 = threading.Thread(target=detect_stretched, args=(camera))
            thread5.daemon = True

            thread5.start()

            thread1 = threading.Thread(target=step1, args=(bot_left, point_left))

            thread1.start()

            thread1.join()
