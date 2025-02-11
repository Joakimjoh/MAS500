import time
import pyrealsense2 as rs
# from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

# def initialize_robots():
#     """
#     Initializes two Interbotix VX250 6-DOF robotic arms and keeps trying until both are ready.

#     Returns:
#         tuple: A tuple containing two initialized robot objects (robot1, robot2).
#     """
#     while True:
#         try:
#             # Initialize the first robot
#             robot1 = InterbotixManipulatorXS(
#                 robot_model="wx250s",
#                 robot_name="wx250s_robot1",
#                 use_gripper=True
#             )
#             print("Robot 1 initialized successfully.")


#             if robot1:
#                 print("Both robots are ready for commands!")
#                 return robot1

#         except Exception as e:
#             print(f"An error occurred while initializing robots: {e}")

#         print("Retrying initialization in 5 seconds...")
#         time.sleep(5)  # Wait for 5 seconds before retrying

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

    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()

    # Get intrinsics
    intrinsics_depth = depth_stream.get_intrinsics()

    return pipeline, align, intrinsics_depth