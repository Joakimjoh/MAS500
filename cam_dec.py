import cv2
import numpy as np
import globals
import math

# Camera calibration parameters (example values, replace with your calibration data)
camera_matrix = np.array([
    [1000, 0, 320],  # fx, 0, cx
    [0, 1000, 240],  # 0, fy, cy
    [0, 0, 1]        # 0,  0,  1
], dtype=np.float32)

# Dummy distortion coefficients (replace with actual values if needed)
dist_coeffs = np.zeros(5)

def calculate_xyz(pixel_coords, depth, camera_matrix):
    """Calculate the (x, y, z) coordinates of an object in centimeters."""
    u, v = pixel_coords
    uv1 = np.array([u, v, 1], dtype=np.float32).reshape(3, 1)  # Pixel coordinates in homogeneous form
    xyz_camera = np.dot(np.linalg.inv(camera_matrix), uv1) * depth
    return xyz_camera.flatten() / 10  # Convert from mm to cm

def check_unfolded():
    # Run Ai model to check if item is unfolded
    print("Unfolded")

def scan_item():
    # Run Ai model to check what type of item and get two shoulder, hip or corner points for item
    print("Scanning object")
    globals.unfolded = True

    return

def check_line_straightness():
    """
    Function to check if the line in the current frame is straight.
    It returns True the first time the line is determined to be straight.
    Uses `globals.frame` for processing.
    """
    # Define a minimum contour area threshold
    MIN_CONTOUR_AREA = 5000  # Adjust this value based on your application

    while True:
        frame = globals.frame

        # Apply a slight blur to the frame to reduce noise
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # Get frame dimensions
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for red color
        lower_red1 = np.array([0, 120, 70])  # Adjust for your lighting conditions
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Threshold the HSV image to isolate red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply Gaussian blur to the mask to reduce noise
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_CONTOUR_AREA]

        # If any valid contours are found, process the largest one
        if contours:
            # Get the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Approximate the contour to reduce noise
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Find corners relative to the center
            corners_relative = [(pt[0][0] - center_x, center_y - pt[0][1]) for pt in largest_contour]

            # Filter points by x-coordinate (negative and positive)
            left_points = [pt for pt in corners_relative if pt[0] < 0]  # Negative x
            right_points = [pt for pt in corners_relative if pt[0] > 0]  # Positive x

            # Find the highest y-value for each group
            highest_left_image = None
            highest_right_image = None

            if left_points:
                highest_left = max(left_points, key=lambda pt: pt[1])
                highest_left_image = (highest_left[0] + center_x, center_y - highest_left[1])

            if right_points:
                highest_right = max(right_points, key=lambda pt: pt[1])
                highest_right_image = (highest_right[0] + center_x, center_y - highest_right[1])

            # Extract the contour segment connecting the red and blue points
            if highest_left_image and highest_right_image:
                # Find the indices of the points in the contour
                left_index = np.argmin(
                    [np.linalg.norm(np.array((pt[0][0], pt[0][1])) - np.array(highest_left_image)) for pt in largest_contour]
                )
                right_index = np.argmin(
                    [np.linalg.norm(np.array((pt[0][0], pt[0][1])) - np.array(highest_right_image)) for pt in largest_contour]
                )

                # Ensure proper order (left -> right)
                if left_index > right_index:
                    left_index, right_index = right_index, left_index

                # Extract the two possible segments
                segment1 = largest_contour[left_index:right_index + 1]
                segment2 = np.concatenate((largest_contour[right_index:], largest_contour[:left_index + 1]))

                # Choose the shorter segment
                if cv2.arcLength(segment1, False) < cv2.arcLength(segment2, False):
                    chosen_segment = segment1
                else:
                    chosen_segment = segment2

                # Check if the segment is straight
                x1, y1 = highest_left_image
                x2, y2 = highest_right_image

                # Calculate the line equation: y = mx + c
                if x2 != x1:
                    m = (y2 - y1) / (x2 - x1)  # Slope
                    c = y1 - m * x1            # Intercept

                    # Calculate the deviation for each point in the segment
                    deviations = []
                    for pt in chosen_segment:
                        px, py = pt[0]
                        # Distance from point (px, py) to the line y = mx + c
                        distance = abs(m * px - py + c) / math.sqrt(m**2 + 1)
                        deviations.append(distance)

                    # Maximum deviation
                    max_deviation = max(deviations)
                    
                else:
                    # Vertical line case
                    max_deviation = max(abs(pt[0][0] - x1) for pt in chosen_segment)

                threshold = 5.0  # Adjust as needed
                is_straight = max_deviation < threshold

                # Display the result
                if is_straight:
                    print("Straight")
                    return True
                else:
                    print("Not Straight")
                    return False

def right_left_corners():
    # Convert to HSV for color segmentation
    frame = globals.frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for red color
    lower_red1 = np.array([0, 120, 70])     # Lower range for red
    upper_red1 = np.array([10, 255, 255])  # Upper range for red
    lower_red2 = np.array([170, 120, 70])  # Second lower range for red
    upper_red2 = np.array([180, 255, 255]) # Second upper range for red

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours of red objects
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the frame dimensions to define the center
    frame_height, frame_width = frame.shape[:2]
    center_x = frame_width // 2
    center_y = frame_height // 2

    # Initialize variables to store the relative left-most and right-most points
    left_most_relative = None
    right_most_relative = None

    if contours:
        for contour in contours:
            if cv2.contourArea(contour) < 50:  # Ignore small contours
                continue

            # Find the extreme points in the contour
            left_point = tuple(contour[contour[:, :, 0].argmin()][0])
            right_point = tuple(contour[contour[:, :, 0].argmax()][0])

            # Convert to coordinates relative to the center of the frame
            left_point_relative = (left_point[0] - center_x, center_y - left_point[1])
            right_point_relative = (right_point[0] - center_x, center_y - right_point[1])

            # Update left-most and right-most points globally
            if (left_most_relative is None or 
                left_point_relative[0] < left_most_relative[0]):
                left_most_relative = left_point_relative
            if (right_most_relative is None or 
                right_point_relative[0] > right_most_relative[0]):
                right_most_relative = right_point_relative

    # Save the detected points
    if left_most_relative and right_most_relative:
        print(f"Left-most point relative to center: {left_most_relative}")
        print(f"Right-most point relative to center: {right_most_relative}")
    
    return left_most_relative, right_most_relative

def start_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return
    
    print("Camera opened")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame could not be read.")
            break

        frame = cv2.flip(frame, 1) # Invert the camera view (mirror effect)
        globals.frame = frame

        # Display the frame
        cv2.imshow("Camera Feed", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop camera
    cap.release()
    cv2.destroyAllWindows()
