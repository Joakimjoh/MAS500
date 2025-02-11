import numpy as np
import cv2
import math
import os
import time
import json
import parameters

detected_shapes_recent = []

def get_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)  # Align depth and color frames

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())

    return depth_frame, color_image

def get_depth_data(pipeline, align):
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

def get_squares_depth_data(pipeline, align):
    """Captures depth data for the largest detected orange outline."""
    start_time = time.time()
    largest_contour = None
    max_area = 0

    while time.time() - start_time < 10:
        # Capture depth frame and color image
        depth_frame, color_image = get_frames(pipeline, align)

        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Define the threshold for gray area (you can adjust these values as needed)
        lower_gray = 100  # Lower threshold for gray
        upper_gray = 200  # Upper threshold for gray

        # Apply threshold to detect gray area
        _, gray_mask = cv2.threshold(gray, lower_gray, upper_gray, cv2.THRESH_BINARY)

        # Reduce noise with blur and morphological operations (optional)
        gray_mask = cv2.GaussianBlur(gray_mask, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour

        # Draw contours on the image for visualization
        if largest_contour is not None:
            cv2.drawContours(color_image, [largest_contour], -1, (0, 255, 0), 2)

        # Show detection process
        cv2.imshow("Largest Orange Outline Detection", color_image)
        cv2.waitKey(1)  # Refresh window

    # After 10 seconds, get depth data for the largest detected outline
    if largest_contour is not None:
        h, w = depth_frame.get_height(), depth_frame.get_width()
        mask = np.zeros((h, w), dtype=np.uint8)

        # Fill the mask with the largest contour (outline area)
        cv2.fillPoly(mask, [largest_contour], 255)

        # Get image center for depth calculation
        center_x = color_image.shape[1] // 2
        center_y = color_image.shape[0] // 2

        depth_data = []

        for y in range(h):
            for x in range(w):
                if mask[y, x] == 255:  # Only process pixels inside the outline
                    depth_value = depth_frame.get_distance(x, y)
                    if depth_value > 0:  # Only valid depth values
                        depth_data.append({
                            "x": x - center_x,  
                            "y": center_y - y,  
                            "depth": float(depth_value)
                        })

        # Save depth data to JSON
        with open("detected_depth_data.json", "w") as f:
            json.dump(depth_data, f, indent=4)

        cv2.destroyAllWindows()
        return depth_data  # Return the depth data of the largest outline
    
def draw_detected_shapes(color_image, shapes):
    """Draw outlines around the detected shapes."""
    for shape in shapes:
        # Draw the square or parallelogram outline without any additional sorting
        cv2.polylines(color_image, [shape], isClosed=True, color=(0, 255, 0), thickness=2)
    
    return color_image
        
def get_xyz(point_depth, closest_point, rel_x):
    """Calculate height above the table using the closest point's depth and new point's depth."""
    # Depth of the closest point and the new point
    closest_point_depth = closest_point['depth']

    beta = math.degrees(math.asin(parameters.CAMERA_HEIGHT / closest_point_depth))

    beta_rad = math.radians(beta)

    z = abs(point_depth - closest_point_depth) * math.sin(beta_rad)

    c = math.sqrt(point_depth**2 - (parameters.CAMERA_HEIGHT - z)**2)

    if c > rel_x:
        y = math.sqrt(c**2 - rel_x**2)
    else:
        y = math.sqrt(rel_x**2 - c**2)

    x = abs(rel_x)

    return x, y, z
