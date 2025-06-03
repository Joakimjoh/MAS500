import pyrealsense2 as rs
import cv2
import numpy as np

# Configure and start RealSense pipeline
# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

# Get camera intrinsics
profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color)
color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                          [0, color_intrinsics.fy, color_intrinsics.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(color_intrinsics.coeffs)

depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 5)

try:
    while True:
        # Wait for frames from camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue  # Skip if no frame is received

        # Get frame dimensions
        width = depth_frame.get_width()
        height = depth_frame.get_height()

        # Compute center pixel
        center_x = width // 2
        center_y = height // 2

        # Get depth at the center
        depth = depth_frame.get_distance(center_x, center_y)

        # Convert depth frame to an image for display
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Display depth image
        cv2.imshow('Depth Frame', depth_colormap)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF  # Get key input

        if key == ord('f'):  # If 'F' is pressed
            print(f"Depth at center ({center_x}, {center_y}): {depth:.3f} meters")

        elif key == ord('q'):  # Press 'Q' to quit
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
