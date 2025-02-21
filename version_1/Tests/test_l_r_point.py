import pyrealsense2 as rs
import numpy as np
import cv2
import math
import os
import json
from test_calibration import capture_squares_depth_data, initialize_camera

# Path for saving depth data
DEPTH_DATA_FILE = "detected_depth_data.json"

def detect_red_object(color_image, depth_frame):
    """Detect red objects in the image by color thresholding."""

    # Get the center of the image
    center_x, center_y = color_image.shape[1] // 2, color_image.shape[0] // 2

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

    # Apply Gaussian blur and morphological operations to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of the red areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If a red object is detected, find the leftmost and rightmost points
    largest_contour = None
    offset = 10

    if contours:
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Filter out contours smaller than 5000 in area
        contours = [c for c in contours if cv2.contourArea(c) > 5000]

        if contours:
            largest_contour = contours[0]

            # Outline the red object on the RGB frame
            cv2.drawContours(color_image, [largest_contour], -1, (255, 0, 0), 2)  # Blue outline
            
            # Find the extreme points in the largest contour
            left_point = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
            right_point = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])

            left_point_offset_x = left_point[0] + offset
            right_point_offset_x = right_point[0] - offset

            # Transform coordinates so the center is (0, 0)
            left_point_transformed = (left_point_offset_x - center_x, left_point[1]- center_y)
            right_point_transformed = (right_point_offset_x - center_x, right_point[1] - center_y)

            left_point_y = (0, left_point_transformed[1])
            right_point_y = (0, right_point_transformed[1])

            # Get the depth at the leftmost and rightmost points
            left_depth = depth_frame.get_distance(left_point_offset_x, left_point[1])  # Depth at leftmost point
            right_depth = depth_frame.get_distance(right_point_offset_x, right_point[1])  # Depth at rightmost point

            return left_depth, right_depth, left_point_transformed, right_point_transformed, left_point_y, right_point_y

    return None, None, None, None, None, None  # Return None if no red object was found

def display_point_on_frame(color_image, left_point, right_point, left_point_m, right_point_m):
    """Displays the camera frame with a blue dot at the detected point, coordinates, depth, and height above the table."""

    # Reverse the transformation to get the original coordinates
    left_point_original = (left_point[0] + color_image.shape[1] // 2, left_point[1] + color_image.shape[0] // 2)
    right_point_original = (right_point[0] + color_image.shape[1] // 2, right_point[1] + color_image.shape[0] // 2)
    
    # Draw a blue circle at the center of the red object
    cv2.circle(color_image, left_point_original, 5, (0, 0, 255), -1)
    # Draw a green circle at the center of the red object
    cv2.circle(color_image, right_point_original, 5, (0, 255, 0), -1)

    # Calculate the center of the image
    center_x = color_image.shape[1] // 2
    center_y = color_image.shape[0] // 2

    # Draw a gray circle at the center of the image
    cv2.circle(color_image, (center_x, center_y), 5, (128, 128, 128), -1)

    # Display the coordinates and depth values
    cv2.putText(color_image, f"Left X: {left_point_m[0]:.3f}, Left Y: {left_point_m[1]:.3f}, Left Z: {left_point_m[2]:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(color_image, f"Right X: {right_point_m[0]:.3f}, Right Y: {right_point_m[1]:.3f}, Right Z: {right_point_m[2]:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Show the frame with the red object marked, gray center dot, and other information
    cv2.imshow("Frame with Red Object", color_image)

def get_xyz(point_depth, closest_point, closest_point_y, cam_height=0.49):
    """Calculate height above the table using the closest point's depth and new point's depth."""
    # Depth of the closest point and the new point
    closest_point_depth = closest_point['depth']
    beta = math.degrees(math.acos(cam_height / closest_point_depth))

    closest_point_y_depth = closest_point_y['depth']
    alpha = math.degrees(math.acos(cam_height / closest_point_y_depth))

    beta_rad = math.radians(beta)

    alpha_rad = math.radians(alpha)

    c = point_depth * math.sin(beta_rad)

    y = closest_point_y_depth * math.sin(alpha_rad)

    z = cam_height - point_depth * math.cos(beta_rad)

    # Calculate x, ensuring the value inside the sqrt is non-negative
    value = c**2 - (y)**2
    x = math.sqrt(abs(value))
    
    return x, y, z

def get_positions(left_depth, right_depth, closest_point_left, closest_point_right, closest_point_left_y, closest_point_right_y):
    
    left_point = get_xyz(left_depth, closest_point_left, closest_point_left_y)
    right_point = get_xyz(right_depth, closest_point_right, closest_point_right_y)

    return left_point, right_point

def get_depth_data(pipeline, align):
    # Check if chessboard depth data file exists
    if os.path.exists(DEPTH_DATA_FILE):
        print("Depth data file found, loading...")
        with open(DEPTH_DATA_FILE, "r") as f:
            depth_data = json.load(f)
    else:
        print("No depth data file found, capturing...")
        depth_data = capture_squares_depth_data(pipeline, align)

    # Convert depth data to a dictionary for fast lookup
    depth_data_dict = {(point['x'], point['y']): point for point in depth_data}

    return depth_data_dict

def get_frames(align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)  # Align depth and color frames

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())

    return depth_frame, color_image

if __name__ == "__main__":
    # Initialize camera
    pipeline, align = initialize_camera()

    # Get depth data for table
    depth_data = get_depth_data(pipeline, align)

    print("\nLooking for red objects inside the chessboard area...")

    while True:        
        # Get depth and color frame
        depth_frame, color_image = get_frames(pipeline, align)

        # Detect red objects and get their coordinates
        left_depth, right_depth, left_point, right_point, left_point_y, right_point_y = detect_red_object(color_image, depth_frame)

        closest_point_left = depth_data.get(left_point, None)
        closest_point_right = depth_data.get(right_point, None)

        closest_point_left_y = depth_data.get(left_point_y, None)
        closest_point_right_y = depth_data.get(right_point_y, None)

        if closest_point_left and closest_point_right and closest_point_left_y and closest_point_right_y != None:
            left_point_m, right_point_m = get_positions(left_depth, right_depth, closest_point_left, closest_point_right, closest_point_left_y, closest_point_right_y)
            display_point_on_frame(color_image, left_point, right_point, left_point_m, right_point_m)
        else:
            # Even if no red object is detected, still display the frame
            cv2.imshow("Frame with Red Object", color_image)

        # Exit condition (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting.")
            break

    cv2.destroyAllWindows()
