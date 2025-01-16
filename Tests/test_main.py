import cv2
import numpy as np
import pyrealsense2 as rs
from test_cam import get_left_right, get_line_straight
from move_robot import move_to_xyz_with_gripper
            
# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable reduced-resolution streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream at 640x480
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB stream at 640x480

x1, y1, z1, x2, y2, z2 = get_left_right(pipeline, config)

xyz = []
xyz.append(x1, y1, z1)
print(x1, y1, z1, x2, y2, z2)

move_to_xyz_with_gripper(xyz)

straight = get_line_straight(pipeline, config)

if straight:
    print('yes')