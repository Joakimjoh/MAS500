#from initialize import initialize_robots
from launch import launch_both_robots
from cam_dec import start_camera, scan_item, right_left_corners, check_unfolded
from robot_control import pickup_stretch
import globals
import threading
import time
import pyrealsense2 as rs

import cv2
import numpy as np

# Launch both robots
launch_both_robots()

# Initialize the robots
#robot1, robot2 = initialize_robots()

# Start camera feed in a thread
#start_camera_thread = threading.Thread(target=start_camera)
#start_camera_thread.start()
    
while True:
    time.sleep(3)
    globals.detect = True
    if globals.detect:
        # Get right and left most corners
        left_pos, right_pos = right_left_corners()
        # Move robot arms to corners pick up stretch till line between is straight and lay down
        pickup_stretch(left_pos, right_pos)

        # Find out item type and move until unfolded
        while True:
            left_corner, right_corner, item_type = scan_item() # Scan with Ai model to check what type of item and get corners

            if item_type == 1:
                print("T-shirt")
            elif item_type == 2:
                print("Sweater")

            # Move robot arms to corners pick up stretch till line between is straight and lay down
            pickup_stretch(left_corner, right_corner)

            # If item unfolded, reset
            if check_unfolded():
                globals.detect = False
                break