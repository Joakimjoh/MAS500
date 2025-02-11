from dual_arm_xs import InterbotixManipulatorXS
from data import get_squares_depth_data
import pyrealsense2 as rs
import parameters
import rclpy
import json
import os

def initialize_robots():
    # Initialize Ros Node once for shared use
    rclpy.init()

    # Create the Interbotix manipulators for the two robots
    bot1 = InterbotixManipulatorXS(
        robot_model='wx250s',
        group_name='arm',
        robot_name='arm1',
        gripper_name='gripper',
        node_name='node1'
    )
    bot2 = InterbotixManipulatorXS(
        robot_model='wx250s',
        group_name='arm',
        robot_name='arm2',
        gripper_name='gripper',
        node_name='node2'
    )

    return bot1, bot2

def initialize_camera():
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # Adjust RGB sensor settings
    device = pipeline.get_active_profile().get_device()
    depth_sensor = device.query_sensors()[0]  # Depth sensor
    rgb_sensor = device.query_sensors()[1]  # RGB sensor

    # Set RGB sensor options
    rgb_sensor.set_option(rs.option.saturation, 30)  # Set saturation to 30
    rgb_sensor.set_option(rs.option.sharpness, 100)  # Set sharpness to 100
    # Set depth sensor options
    depth_sensor.set_option(rs.option.visual_preset, 5)
    
    align = rs.align(rs.stream.color)

    return pipeline, align, profile

def initialize_depth_data(pipeline, align):
    # Check if chessboard depth data file exists
    if os.path.exists(parameters.DEPTH_DATA_FILE):
        print("Depth data file found, loading...")
        with open(parameters.DEPTH_DATA_FILE, "r") as f:
            depth_data = json.load(f)
    else:
        print("No depth data file found, capturing...")
        depth_data = get_squares_depth_data(pipeline, align)

    # Convert depth data to a dictionary for fast lookup
    depth_data_dict = {(point['x'], point['y']): point for point in depth_data}

    return depth_data_dict