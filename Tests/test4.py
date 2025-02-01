import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
import math
from test3 import capture_squares_depth_data


# Parameters
SQUARE_SIZE = 0.044  # 4.5 cm per square (square size in meters)

# Path for saving depth data
DEPTH_DATA_FILE = "detected_squares_depth_data.json"

def detect_red_object(color_image, depth_frame, detected_squares_data):
    """Detect red objects in the image by color thresholding."""
    # Get image dimensions
    height, width, _ = color_image.shape
    
    # Calculate the center of the frame
    center_x = width // 2
    center_y = height // 2  # Center of the frame along Y-axis

    # Convert the image to HSV
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Define the range for red color (this may need adjustment depending on the shade of red)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine both masks to detect all shades of red
    mask = mask1 | mask2

    # Find contours of the red areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If a red object is detected, draw a blue dot at its center
    for contour in contours:
        # Only consider large enough contours
        if cv2.contourArea(contour) > 500:
            # Get the bounding box of the red object
            x, y, w, h = cv2.boundingRect(contour)
            cX, cY = x + w // 2, y + h // 2  # Center of the red object

            # Get the depth at the center of the red object
            depth = depth_frame.get_distance(cX, cY)

            # Convert (cX, cY) to real-world coordinates relative to the chessboard
            real_x = (cX - np.mean([point["x"] for square in detected_squares_data for point in square])) * SQUARE_SIZE
            real_y = (cY - np.mean([point["y"] for square in detected_squares_data for point in square])) * SQUARE_SIZE

            # Calculate the displacement in x (left-right) from the center of the frame
            displacement_left_right = (cX - center_x) * SQUARE_SIZE / (width // 2)

            # Calculate the displacement in y (up-down) from the center of the frame
            displacement_up_down = (cY - center_y) * SQUARE_SIZE / (height // 2)

            # Return real-world coordinates, depth, displacements from center, and pixel center
            return real_x, real_y, depth, cX, cY, displacement_left_right, displacement_up_down

    return None, None, None, None, None, None, None  # Return None if no red object was found

def display_point_on_frame(closest_point, color_image, real_x, real_y, cX, cY, depth, height_above_table, displacement_left_right, displacement_up_down):
    """Displays the camera frame with a blue dot at the detected point, coordinates, depth, and height above the table."""
    
    # Draw a blue circle at the center of the red object
    cv2.circle(color_image, (cX, cY), 10, (255, 0, 0), -1)  # Blue dot

    cv2.circle(color_image, (closest_point['x'], closest_point['y']), 5, (0, 255, 0), -1)  # Green dot
    
    # Calculate the height of the object above the table (difference in depth)
    #height_above_table = table_height - depth

    # Display the real-world coordinates, depth, height, and displacements from center
    cv2.putText(color_image, f"Real-world coordinates: X={real_x:.3f}, Y={real_y:.3f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(color_image, f"Depth: {depth:.3f} meters", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(color_image, f"Height above table: {height_above_table:.3f} meters", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(color_image, f"Displacement from center (X): {displacement_left_right:.3f} meters", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(color_image, f"Displacement from center (Y): {displacement_up_down:.3f} meters", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Show the frame with the red object marked and depth, height, displacement information
    cv2.imshow("Frame with Red Object", color_image)

def calculate_height(cX, cY, depth, closest_point):
    """Calculate height above the table using the closest point's depth and new point's depth."""
    # Depth of the closest point and the new point
    closest_point_depth_c = closest_point['depth_c']

    height_above_table = calculate_height_above_table(closest_point_depth_c, depth)

    print(depth, closest_point_depth_c, height_above_table)
    
    return height_above_table

def calculate_height_above_table(closest_point_depth, point_depth, cam_height=0.49):
    beta = math.degrees(math.acos(cam_height / closest_point_depth))

    beta_rad = math.radians(beta)

    height_above_table = cam_height - point_depth * math.cos(beta_rad)

    return height_above_table

def find_closest_point(x, y, detected_squares_data):
    """Find the closest point in detected squares data to the new point."""
    min_distance = float('inf')
    closest_point = None

    for square_data in detected_squares_data:
        for point in square_data:
            # Compare distances between new point and each saved point
            distance = np.linalg.norm(np.array([float(x), float(y)]) - np.array([point['x'], point['y']]))
            if distance < min_distance:
                min_distance = distance
                closest_point = point

    return closest_point

if __name__ == "__main__":
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

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

    # Check if chessboard depth data file exists
    if os.path.exists(DEPTH_DATA_FILE):
        print("Depth data file found, loading...")
        with open(DEPTH_DATA_FILE, "r") as f:
            detected_squares_data = json.load(f)
    else:
        print("No depth data file found, capturing...")
        detected_squares_data, color_image, depth_frame = capture_squares_depth_data(pipeline, align)

    # Use the average depth of chessboard points as the table height
    table_height = np.mean([point['depth_c'] for square in detected_squares_data for point in square])
    print(f"Table height (depth of chessboard center): {table_height:.3f} meters")

    print("\nLooking for red objects inside the chessboard area...")

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  # Align depth and color frames

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # Detect red objects and get their coordinates
        real_x, real_y, depth, cX, cY, displacement_left_right, displacement_up_down = detect_red_object(color_image, depth_frame, detected_squares_data)

        if real_x is not None and real_y is not None:
            closest_point = find_closest_point(cX, cY, detected_squares_data)
            height_above_table = calculate_height(cX, cY, depth, closest_point)
            display_point_on_frame(closest_point, color_image, real_x, real_y, cX, cY, depth, height_above_table, displacement_left_right, displacement_up_down)
        else:
            # Even if no red object is detected, still display the frame
            cv2.imshow("Frame with Red Object", color_image)

        # Exit condition (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting.")
            break

    cv2.destroyAllWindows()
