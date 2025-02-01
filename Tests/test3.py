import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json

# Path for saving depth data
DEPTH_DATA_FILE = "detected_squares_depth_data.json"

# List to store detected shapes (initializing it properly)
detected_shapes_recent = []

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
                if np.all(side_diff / avg_side_length < 0.5):  # 60% tolerance for side length
                    # Check angles (approx. 90 degrees)
                    angles = [
                        np.arccos(np.clip(np.dot(pts[i] - pts[(i + 1) % 4], pts[(i + 2) % 4] - pts[(i + 1) % 4]) /
                                         (np.linalg.norm(pts[i] - pts[(i + 1) % 4]) *
                                          np.linalg.norm(pts[(i + 2) % 4] - pts[(i + 1) % 4])), -1.0, 1.0))
                        for i in range(4)
                    ]
                    if all(np.abs(angle - np.pi / 2) < np.radians(40) for angle in angles):  # 10 degrees tolerance for angles
                        found_squares.append(approx)

    return found_squares

def capture_squares_depth_data(pipeline, align):
    """Captures depth data for detected squares and their outlines."""
    start_time = time.time()
    detected_squares_data = []  # Stores depth data of squares detected within the 5-second window
    
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        cv2.imshow("Frame with Red Object", color_image)

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
        if elapsed_time > 15 and not detected_squares_data:
            # After 15 seconds, process and store the depth data for each square
            for square in detected_shapes_recent:
                square_data = []
                for point in square:
                    x, y = point[0]
                    depth_t = 0
                    depth_c = get_depth_at_point(depth_frame, x, y)
                    if depth_c != 0:  # Only add valid depth data
                        square_data.append({
                            "x": int(x),  # Convert x and y to int
                            "y": int(y),  # Convert x and y to int
                            "depth_c": float(depth_c),  # Actual depth from the camera
                            "depth_t": float(depth_t)   # Always 0 for points inside squares
                        })

                if square_data:
                    detected_squares_data.append(square_data)

            # Save depth data to JSON
            with open(DEPTH_DATA_FILE, "w") as f:
                json.dump(detected_squares_data, f, indent=4)

            break

    cv2.destroyAllWindows()
    return detected_squares_data, color_image, depth_frame

def draw_detected_shapes(color_image, shapes):
    """Draw outlines around the detected shapes."""
    for shape in shapes:
        # Draw the square or parallelogram outline without any additional sorting
        cv2.polylines(color_image, [shape], isClosed=True, color=(0, 255, 0), thickness=2)
    
    return color_image

def get_depth_at_point(depth_frame, x, y):
    """Get the depth at a given (x, y) pixel position."""
    depth = depth_frame.get_distance(x, y)
    return depth