import cv2
import numpy as np
import pyrealsense2 as rs

def transform_coordinates(x, y, z):
    # Camera is 25 cm (0.25 meters) above the ground
    camera_height = 0.25
    
    # Tilt angle in degrees and converting to radians
    tilt_angle_deg = 35
    tilt_angle_rad = np.radians(tilt_angle_deg)
    
    # Calculate the new y (depth remains the same)
    new_y = z  # Distance from the camera (depth)
    
    # Calculate the new x (vertical position after tilt)
    new_x = y * np.cos(tilt_angle_rad) + z * np.sin(tilt_angle_rad)
    
    # Calculate the new z (height from the ground)
    height_from_ground = camera_height - (z * np.sin(tilt_angle_rad))
    
    # Rotate the coordinates 90 degrees to the right (counterclockwise)
    rotated_x = new_y  # New x is the negative of the depth
    rotated_y = -new_x   # New y is the original vertical position

    # Translate the coordinates by 0.25m forward and 0.2m to the right
    translated_x = rotated_x - 0.2  # Shift 0.2m to the right
    translated_y = rotated_y + 0.25 # Shift 0.25m forward

    return translated_x, translated_y, height_from_ground

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable reduced-resolution streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream at 640x480
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB stream at 640x480

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

try:
    while True:
        # Wait for aligned frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get RGB and aligned depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert frames to numpy arrays
        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Normalize depth image for display
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

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
                # Offset for depth measurement
                offset = 5
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

                # Mark the adjusted points on the RGB frame
                cv2.circle(frame, left_point_inward, 5, (0, 255, 0), -1)  # Green circle for left point
                cv2.circle(frame, right_point_inward, 5, (0, 0, 255), -1)  # Red circle for right point
                
                # Display real-world coordinates of the left and right points in the top-left corner of the frame
                cv2.putText(frame, f"L: ({left_x_meters:.2f}, {left_y_meters:.2f}, {left_z:.2f})m", 
                            (10, 30),  # Top-left corner of the frame
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(frame, f"R: ({right_x_meters:.2f}, {right_y_meters:.2f}, {right_z:.2f})m", 
                            (10, 60),  # Slightly below the first text
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                
                xx, yy, zz = transform_coordinates(right_x_meters, right_y_meters, right_z)

                cv2.putText(frame, f"R2: ({xx:.2f}, {yy:.2f}, {zz:.2f})m", 
                            (10, 90),  # Slightly below the first text
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                            

        # Display the frames
        cv2.imshow("Color Frame with Red Object Outlined", frame)
        #cv2.imshow("Depth Frame", depth_colormap)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline and close OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()