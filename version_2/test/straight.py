import numpy as np
import cv2
import pyrealsense2 as rs

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Get the RealSense camera's intrinsic and extrinsic parameters
profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color)
depth_stream = profile.get_stream(rs.stream.depth)

# Get intrinsic parameters for color and depth cameras
color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

# Get extrinsics (rotation and translation) from depth to color camera
depth_to_color_extrinsics = depth_stream.get_extrinsics_to(color_stream)

# Rotation matrix and translation vector (from depth to color camera)
rotation_matrix = np.array(depth_to_color_extrinsics.rotation).reshape(3, 3)
translation_vector = np.array(depth_to_color_extrinsics.translation)

# Camera matrix (for color stream)
camera_matrix = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                          [0, color_intrinsics.fy, color_intrinsics.ppy],
                          [0, 0, 1]])

# Function to get depth to object (from depth frame)
def get_depth_to_object(u, v):
    depth_frame = pipeline.wait_for_frames().get_depth_frame()
    if not depth_frame:
        return None
    # Get depth at (u, v) pixel
    depth = depth_frame.get_distance(u, v)  # Distance in meters
    return depth

# Function to detect red objects in the image
def detect_red_objects(color_image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV space
    lower_red1 = np.array([0, 120, 70])  # Lower bound for red (0 to 10 degrees in hue)
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])  # Upper bound for red (170 to 180 degrees in hue)
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color in two ranges
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the two masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours of detected red objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours found, return the center of the largest contour
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)

        if M["m00"] != 0:
            # Find center of the object (u, v)
            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])
            return u, v, mask  # Return the center (u, v) of the object
    return None, None, mask

# Function to calculate the tilt angle of the camera
def calculate_tilt_angle(rotation_matrix):
    # Extract the tilt angle (assuming the rotation around the X-axis)
    tilt_angle = np.arcsin(rotation_matrix[2, 2])  # Sin of the tilt angle from R[2, 2]
    return tilt_angle

# Function to calculate the height above the flat surface
def calculate_height_above_surface(depth, tilt_angle):
    # Height above the surface: Z * sin(tilt_angle)
    return depth * np.sin(tilt_angle)

# Loop to capture frames and measure distance
while True:
    # Get frames from RealSense
    frames = pipeline.wait_for_frames()

    # Get color and depth frames
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    # Convert to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())

    # Detect red objects in the color image
    u, v, mask = detect_red_objects(color_image)


    print(f"Rotation Matrix: {rotation_matrix}")
    print(f"Translation Vector: {translation_vector}")


    if u is not None and v is not None:
        # Measure distance to the object at (u, v)
        depth = get_depth_to_object(u, v)

        # If valid depth data is available
        if depth is not None:
            print(f"Distance to red object at ({u}, {v}): {depth:.2f} meters")

            # Calculate tilt angle from the rotation matrix
            tilt_angle = calculate_tilt_angle(rotation_matrix)
            print(f"Camera tilt angle: {np.degrees(tilt_angle):.2f} degrees")

            # Calculate the height above the flat surface
            height_above_surface = calculate_height_above_surface(depth, tilt_angle)
            print(f"Height above the flat surface: {height_above_surface:.2f} meters")

            # Draw circle on the red object in the color image
            cv2.circle(color_image, (u, v), 5, (0, 255, 0), -1)

    # Show the result
    cv2.imshow("Color Image", color_image)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
