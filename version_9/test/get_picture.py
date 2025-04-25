import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Set depth sensor preset
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 5)

try:
    while True:
        # Wait for the next frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Convert to numpy arrays for OpenCV processing
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Display the frames
        cv2.imshow('Color Frame', color_image)

        # Normalize depth image for better visibility
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Depth Frame', depth_colormap)

        # Check if 'F' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):  # If 'F' key is pressed
            # Save the frames when 'F' is pressed
            cv2.imwrite("color_frame.jpg", color_image)
            cv2.imwrite("depth_frame.png", depth_colormap)
            print("Color and Depth images captured and saved.")
            break

        # Close the window if 'Esc' is pressed
        if key == 27:  # 'Esc' key to exit
            break

finally:
    # Stop the pipeline and close all windows
    pipeline.stop()
    cv2.destroyAllWindows()
