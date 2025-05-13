"""Third-Party Libraries"""
import rclpy
import cv2
import os

"""Internal Modules"""
from dual_arm_xs import InterbotixManipulatorXS
from process import Process
from camera import Camera
from tag import Tag

# Initialize ROS 2
rclpy.init()

# Start OpenCV window thread
cv2.startWindowThread()


# Initialize camera
camera = Camera()
while camera.frame is None:
    pass  # Wait for first valid frame

# Check if calibration files exist 
if not os.path.exists("region.csv") or not os.path.exists("empty_workspace.png"):
    print("Missing calibration file(s). Clear the workspace and press Enter to Continue.")
    while camera.key != 13:
        pass

# Initialize dual-arm robots with AprilTag references
bot_left = InterbotixManipulatorXS(
    robot_model='wx250s',
    group_name='arm',
    robot_name='left_arm',
    gripper_name='gripper',
    node_name='node1',
    tag=Tag(camera, "left")
)

bot_right = InterbotixManipulatorXS(
    robot_model='wx250s',
    group_name='arm',
    robot_name='right_arm',
    gripper_name='gripper',
    node_name='node2',
    tag=Tag(camera, "right")
)

# Start repeated unfolding processes
process_count = 1
processes = {}

while True:
    print("Press Enter to start unfolding process...")
    
    # Wait for Enter key (key code 13)
    while camera.key != 13:
        pass

    print(f"\nStarting unfolding process {process_count}")
    process = Process(camera, bot_left, bot_right)
    processes[process_count] = process
    process.unfold()
    print(f"Process {process_count} finished.")
    
    process_count += 1
