import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json

# Path for saving depth data
DEPTH_DATA_FILE = "detected_squares_depth_data.json"

# List to store detected shapes (initializing it properly)
detected_shapes_recent = []

def initialize_camera():
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

    return pipeline, align

def detect_individual_shapes(color_image):
    """Detects individual black squares in the image with better detection techniques."""
    # Convert to grayscale
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)  # Smoothing for noise reduction
    

    # Apply a binary inverse threshold to highlight dark regions (black squares)
    _, thresholded_image = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Use Canny edge detection for contour detection
    edges = cv2.Canny(thresholded_image, 100, 200)

    # Find contours from the edges detected
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_squares = []

    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filtering out too small contours
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:  # Only consider quadrilaterals (squares)
                pts = np.array(approx).reshape((-1, 2))
                side_lengths = [
                    np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)
                ]

                # Check if sides are approximately equal within a 60% tolerance
                avg_side_length = np.mean(side_lengths)
                side_diff = np.abs(side_lengths - avg_side_length)
                if np.all(side_diff / avg_side_length < 0.4):  # 60% tolerance for side length
                    # Check angles (approx. 90 degrees)
                    angles = [
                        np.arccos(np.clip(np.dot(pts[i] - pts[(i + 1) % 4], pts[(i + 2) % 4] - pts[(i + 1) % 4]) /
                                         (np.linalg.norm(pts[i] - pts[(i + 1) % 4]) *
                                          np.linalg.norm(pts[(i + 2) % 4] - pts[(i + 1) % 4])), -1.0, 1.0))
                        for i in range(4)
                    ]
                    if all(np.abs(angle - np.pi / 2) < np.radians(30) for angle in angles):  # 10 degrees tolerance for angles
                        found_squares.append(approx)

    return found_squares

def capture_squares_depth_data(pipeline, align):
    """Captures depth data for detected squares and their outlines."""
    start_time = time.time()
    
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

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
            center_x, center_y = color_image.shape[1] // 2, color_image.shape[0] // 2

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