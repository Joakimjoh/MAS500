import numpy as np
import cv2
import pyrealsense2 as rs
import apriltag  # For detecting AprilTags

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

# Camera matrix (for color stream)
camera_matrix = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                          [0, color_intrinsics.fy, color_intrinsics.ppy],
                          [0, 0, 1]])

# Distortion coefficients (color camera)
distortion_coeffs = np.array([color_intrinsics.coeffs[0], color_intrinsics.coeffs[1], 
                              color_intrinsics.coeffs[2], color_intrinsics.coeffs[3], 
                              color_intrinsics.coeffs[4]])

# Rotation matrix and translation vector (from depth to color camera)
rotation_matrix = np.array(depth_to_color_extrinsics.rotation).reshape(3, 3)
translation_vector = np.array(depth_to_color_extrinsics.translation)

# Initialize the AprilTag detector
detector = apriltag.Detector()

# Function to get depth to object (from depth frame)
def get_depth_to_object(u, v):
    depth_frame = pipeline.wait_for_frames().get_depth_frame()
    if not depth_frame:
        return None
    U = int(u)
    V = int(v)
    # Get depth at (u, v) pixel
    depth = depth_frame.get_distance(U, V)  # Distance in meters
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

# Function to calculate horizontal displacement (left/right from camera center)
def calculate_horizontal_displacement(u, depth):
    # Use the formula to calculate the horizontal displacement
    displacement = (u - color_intrinsics.ppx) * depth / color_intrinsics.fx  # In meters
    return displacement

# Function to convert pixel coordinates (u, v) to 3D camera coordinates (X, Y, Z)
def pixel_to_camera_coordinates(u, v, depth):
    # Invert the camera intrinsic matrix to get camera coordinates
    normalized_x = (u - color_intrinsics.ppx) / color_intrinsics.fx
    normalized_y = (v - color_intrinsics.ppy) / color_intrinsics.fy
    
    # Camera coordinates (X, Y, Z) in meters
    X = depth * normalized_x
    Y = depth * normalized_y
    Z = depth
    
    return X, Y, Z

# Function to transform camera coordinates to world coordinates
def camera_to_world_coordinates(X, Y, Z):
    # Apply rotation and translation to get world coordinates
    camera_coords = np.array([X, Y, Z])
    world_coords = rotation_matrix.dot(camera_coords) + translation_vector
    return world_coords

# Function to detect AprilTag and get its bounding box
def detect_apriltag_and_get_transform(color_image):
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray_image)
    
    if results:
        # Assuming we only use the first detected tag
        tag = results[0]
        
        # Draw bounding box around AprilTag
        (ptA, ptB, ptC, ptD) = tag.corners
        ptA = tuple(np.int32(ptA))
        ptB = tuple(np.int32(ptB))
        ptC = tuple(np.int32(ptC))
        ptD = tuple(np.int32(ptD))
        
        cv2.line(color_image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(color_image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(color_image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(color_image, ptD, ptA, (0, 255, 0), 2)
        
        # Camera coordinates of the tag's center
        tag_center = tag.center
        u, v = tag_center

        # Get depth at the tag's center (u, v)
        depth = get_depth_to_object(u, v)

        if depth is not None:
            # Convert tag center from pixel coordinates to 3D camera coordinates
            X, Y, Z = pixel_to_camera_coordinates(u, v, depth)

            # Calculate the world coordinates of the tag
            world_coords = camera_to_world_coordinates(X, Y, Z)
            
            cv2.line(color_image, ptA, ptB, (0, 0, 255), 3)  # Red line for X-axis

            cv2.line(color_image, ptA, ptD, (0, 255, 0), 3)  # Green line for Y-axis

            # We simulate this in 2D space with a small visual offset
            z_offset = np.array([int(ptA[0] + 5), int(ptA[1] - 15)])

            # Convert z_offset to a tuple of integers before passing to cv2.line
            z_offset_tuple = tuple(z_offset)

            cv2.line(color_image, ptA, z_offset_tuple, (255, 0, 0), 3)  # Blue line for Z-axis

            # Return the tag's world coordinates and drawn image
            return world_coords
    return None

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

    # Detect AprilTag and get new coordinate system
    new_origin = detect_apriltag_and_get_transform(color_image)

    if new_origin is not None:
        print(f"New coordinate system origin from AprilTag: X={new_origin[0]:.2f} m, Y={new_origin[1]:.2f} m, Z={new_origin[2]:.2f} m")

    # Detect red objects in the color image
    u, v, mask = detect_red_objects(color_image)

    if u is not None and v is not None:
        # Measure distance to the object at (u, v)
        depth = get_depth_to_object(u, v)

        # If valid depth data is available
        if depth is not None:
            print(f"Distance to red object at ({u}, {v}): {depth:.2f} meters")

            # Calculate horizontal displacement (how far left/right)
            horizontal_displacement = calculate_horizontal_displacement(u, depth)
            print(f"Red object is {horizontal_displacement:.2f} meters to the {'right' if horizontal_displacement > 0 else 'left'}")

            # Convert pixel coordinates to 3D camera coordinates
            X, Y, Z = pixel_to_camera_coordinates(u, v, depth)
            print(f"Object coordinates in camera frame: X={X:.2f} m, Y={Y:.2f} m, Z={Z:.2f} m")

            # Convert camera coordinates to world coordinates
            world_coords = camera_to_world_coordinates(X, Y, Z)
            print(f"Object world coordinates: X={world_coords[0]:.2f} m, Y={world_coords[1]:.2f} m, Z={world_coords[2]:.2f} m")

            # If AprilTag's new origin is available, use its system to calculate coordinates
            if new_origin is not None:
                # Transform the object coordinates to the new system (relative to the AprilTag origin)
                relative_coords = world_coords - new_origin
                print(f"Object coordinates in AprilTag coordinate system: X={relative_coords[0]:.2f} m, Y={relative_coords[1]:.2f} m, Z={relative_coords[2]:.2f} m")

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
