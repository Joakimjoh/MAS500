import pyrealsense2 as rs
import numpy as np
import cv2
import math

def check_line_straight(pipeline, left_point = (200, 200), right_point = (100, 200)):
# Define a minimum contour area threshold
    MIN_CONTOUR_AREA = 5000  # Adjust this value based on your application
    while True:
        # Wait for a frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert the RealSense frame to a NumPy array
        frame = np.asanyarray(color_frame.get_data())

        # Get frame dimensions
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2

        # Draw the center (0, 0) point
        cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)  # White point for origin

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

        # Process the largest contour if it exists
            # If any valid contours are found, process the largest one
        if contours:
            # Get the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Approximate the contour to reduce noise
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            left_point = (300, 200)
            right_point = (400, 300)

            # Find the closest point to left_point in the contour
            closest_left_index = np.argmin([np.linalg.norm(np.array((pt[0][0], pt[0][1])) - np.array(left_point)) for pt in largest_contour])
            closest_left_point = largest_contour[closest_left_index][0]

            # Find the closest point to right_point in the contour
            closest_right_index = np.argmin([np.linalg.norm(np.array((pt[0][0], pt[0][1])) - np.array(right_point)) for pt in largest_contour])
            closest_right_point = largest_contour[closest_right_index][0]

            # Draw the closest points on the image
            cv2.circle(frame, tuple(closest_left_point), 5, (255, 0, 0), -1)  # Red point
            cv2.circle(frame, tuple(closest_right_point), 5, (255, 0, 0), -1)  # Blue point

            cv2.circle(frame, tuple(left_point), 5, (0, 0, 255), -1)  # Red point
            cv2.circle(frame, tuple(right_point), 5, (0, 0, 255), -1)  # Blue point

            # Ensure proper order (left -> right)
            if closest_left_index > closest_right_index:
                closest_left_index, closest_right_index = closest_right_index, closest_left_index

            # Extract the two possible segments
            segment1 = largest_contour[closest_left_index:closest_right_index + 1]
            segment2 = np.concatenate((largest_contour[closest_right_index:], largest_contour[:closest_left_index + 1]))

            # Choose the shorter segment
            if cv2.arcLength(segment1, False) < cv2.arcLength(segment2, False):
                chosen_segment = segment1
            else:
                chosen_segment = segment2

            # Draw the chosen segment
            cv2.polylines(frame, [chosen_segment], isClosed=False, color=(255, 255, 0), thickness=2)  # Cyan line

            # Check if the segment is straight
            x1, y1 = closest_left_point
            x2, y2 = closest_right_point

            # Calculate the line equation: y = mx + c
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

            # Define a threshold for straightness
            threshold = 5.0  # Adjust as needed
            is_straight = max_deviation < threshold

            if is_straight:
                return True

test = check_line_straight(pipeline)

print(test)
