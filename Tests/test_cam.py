import pyrealsense2 as rs
import cv2
import numpy as np
import math

def get_left_right(pipeline, config):

    # Align depth to color
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Start streaming
    pipeline.start(config)

    # Adjust RGB sensor settings
    device = pipeline.get_active_profile().get_device()
    depth_sensor = device.query_sensors()[0]  # Depth sensor
    rgb_sensor = device.query_sensors()[1]  # RGB sensor

    rgb_sensor.set_option(rs.option.saturation, 30)  # Set saturation to 30
    rgb_sensor.set_option(rs.option.sharpness, 100)  # Set sharpness to 100
    depth_sensor.set_option(rs.option.visual_preset, 5)

    # Offset for depth measurement
    offset = 5
                
    # Wait for aligned frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Get RGB and aligned depth frames
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    # Convert frames to numpy arrays
    frame = np.asanyarray(color_frame.get_data())

    # Convert RGB frame to HSV for color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for red color
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color and reduce noise
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply Gaussian blur and morphological operations to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of red objects
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Filter out contours smaller than 5000 in area
        contours = [c for c in contours if cv2.contourArea(c) > 5000]

        if contours:
            # Use the largest contour after filtering
            largest_contour = contours[0]

            # Outline the red object on the RGB frame
            cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2)  # Blue outline

            # Find the extreme points in the largest contour
            left_point = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
            right_point = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])

            # Adjust the points inward by the offset
            left_point_inward = (min(left_point[0] + offset, frame.shape[1] - 1), left_point[1])
            right_point_inward = (max(right_point[0] - offset, 0), right_point[1])

            # Get Z-coordinates (depth) for the adjusted points using aligned depth frame
            left_z = depth_frame.get_distance(left_point_inward[0], left_point_inward[1])
            right_z = depth_frame.get_distance(right_point_inward[0], right_point_inward[1])

            # Get camera intrinsics
            profile = pipeline.get_active_profile()
            depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
            cx, cy = depth_intrinsics.ppx, depth_intrinsics.ppy

            # Calculate real-world coordinates for the left point
            left_x_meters = (left_point_inward[0] - cx) * left_z / fx
            left_y_meters = (left_point_inward[1] - cy) * left_z / fy

            # Calculate real-world coordinates for the right point
            right_x_meters = (right_point_inward[0] - cx) * right_z / fx
            right_y_meters = (right_point_inward[1] - cy) * right_z / fy
            pipeline.stop()
            return right_x_meters, right_y_meters, right_z, left_x_meters, left_y_meters, left_z
        else:
            pipeline.stop()
            return None, None, None
        
def get_line_straight(pipeline, config):

    x1, y1, z1, x2, y2, z2 = get_left_right(pipeline, config)
    # Define two fixed points in meters (real-world coordinates)
    fixed_point1_meters = (x1, y1, z1)  # Example: (x, y, z) in meters
    fixed_point2_meters = (x2, y2, z2)   # Example: (x, y, z) in meters

    # Align depth to color
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Start streaming
    pipeline.start(config)

    # Adjust RGB sensor settings
    device = pipeline.get_active_profile().get_device()
    depth_sensor = device.query_sensors()[0]  # Depth sensor
    rgb_sensor = device.query_sensors()[1]  # RGB sensor

    rgb_sensor.set_option(rs.option.saturation, 30)  # Set saturation to 30
    rgb_sensor.set_option(rs.option.sharpness, 100)  # Set sharpness to 100
    depth_sensor.set_option(rs.option.visual_preset, 5)

    # Wait for aligned frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Get aligned depth frame
    depth_frame = aligned_frames.get_depth_frame()

    # Get camera intrinsics
    profile = pipeline.get_active_profile()
    depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
    cx, cy = depth_intrinsics.ppx, depth_intrinsics.ppy

    # Convert fixed points from meters to pixel coordinates
    def meters_to_pixels(x, y, z):
        pixel_x = int((x * fx / z) + cx)
        pixel_y = int((y * fy / z) + cy)
        return pixel_x, pixel_y

    # Convert fixed points
    fixed_point1_pixels = meters_to_pixels(*fixed_point1_meters)
    fixed_point2_pixels = meters_to_pixels(*fixed_point2_meters)

    # Retrieve depth for the fixed points
    depth1 = depth_frame.get_distance(fixed_point1_pixels[0], fixed_point1_pixels[1])
    depth2 = depth_frame.get_distance(fixed_point2_pixels[0], fixed_point2_pixels[1])

    # Calculate real-world coordinates based on actual depth
    point1_x_meters = (fixed_point1_pixels[0] - cx) * depth1 / fx
    point1_y_meters = (fixed_point1_pixels[1] - cy) * depth1 / fy
    point1_z_meters = depth1

    point2_x_meters = (fixed_point2_pixels[0] - cx) * depth2 / fx
    point2_y_meters = (fixed_point2_pixels[1] - cy) * depth2 / fy
    point2_z_meters = depth2

    # Stop the pipeline and return the coordinates
    pipeline.stop()

    print(f"Point 1 (meters): ({point1_x_meters:.4f}, {point1_y_meters:.4f}, {point1_z_meters:.4f})")
    print(f"Point 2 (meters): ({point2_x_meters:.4f}, {point2_y_meters:.4f}, {point2_z_meters:.4f})")

    return (point1_x_meters, point1_y_meters, point1_z_meters), (point2_x_meters, point2_y_meters, point2_z_meters)
