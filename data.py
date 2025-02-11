import numpy as np
from camera_detection import detect_individual_shapes
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
    """Captures depth data for detected squares and their outlines."""
    start_time = time.time()
    
    while True:
        depth_frame, color_image = get_frames(pipeline, align)

        # Detect black squares in the current frame
        found_shapes = detect_individual_shapes(color_image)

        # Add the found shapes to the list of recently detected squares
        detected_shapes_recent.extend(found_shapes)  # Now using the initialized list

        # Check if 10 seconds have passed to start detecting red objects
        elapsed_time = time.time() - start_time
        # Draw outlines around detected squares
        outlined_image = draw_detected_shapes(color_image, detected_shapes_recent)

        cv2.imshow("Detected Shapes", outlined_image)
        cv2.waitKey(1)  # Make window refresh

        # Check if 15 seconds have passed, then save the detected squares depth data
        if elapsed_time > 10:
            # After 15 seconds, process and store the depth data for each square
            top_left = top_right = bottom_left = bottom_right = (0, 0)
            for square in detected_shapes_recent:
                for point in square:
                    x, y = point[0]

                    # Update extreme points
                    if x + y < top_left[0] + top_left[1]:  # Furthest top-left
                        top_left = (x, y)
                    if x - y > top_right[0] - top_right[1]:  # Furthest top-right
                        top_right = (x, y)
                    if y - x > bottom_left[1] - bottom_left[0]:  # Furthest bottom-left
                        bottom_left = (x, y)
                    if x + y > bottom_right[0] + bottom_right[1]:  # Furthest bottom-right
                        bottom_right = (x, y)

            # Create a NumPy array for the contour
            outline_contour = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

            # Create an empty mask
            # Get image dimensions
            h, w = depth_frame.get_height(), depth_frame.get_width()
            mask = np.zeros((h, w), dtype=np.uint8)

            # Fill the outline (creates a filled region inside the contour)
            cv2.fillPoly(mask, [outline_contour], 255)
            # Get the center of the image
            center_x = color_image.shape[1] // 2
            center_y = color_image.shape[0] // 2

            depth_data = []

            for y in range(h):
                for x in range(w):
                    if mask[y, x] == 255:  # Only process pixels inside the filled contour
                        depth_value = depth_frame.get_distance(x, y)  # Get depth value
                        if depth_value > 0:  # Only store valid depth values
                            depth_data.append({
                                "x": x - center_x,  # Center the x-coordinate
                                "y": y - center_y,  # Invert y-axis to match Cartesian coordinates
                                "depth": float(depth_value)
                            })

            # Save depth data to JSON
            with open("detected_depth_data.json", "w") as f:
                json.dump(depth_data, f, indent=4)

            cv2.destroyAllWindows()
            return depth_data

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

    beta = math.degrees(math.acos(parameters.CAMERA_HEIGHT / closest_point_depth))

    beta_rad = math.radians(beta)

    z = parameters.CAMERA_HEIGHT - point_depth * math.cos(beta_rad)

    c = math.sqrt(point_depth**2 - (parameters.CAMERA_HEIGHT - z)**2)

    if c > rel_x:
        y = math.sqrt(c**2 - rel_x**2)
    else:
        y = math.sqrt(rel_x**2 - c**2)

    x = abs(rel_x)

    return x, y, z
