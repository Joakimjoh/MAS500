import cv2
import numpy as np
import math

# Initialize the camera
cap = cv2.VideoCapture(1)  # Use 0 for the default camera

if not cap.isOpened():
    raise Exception("Could not open the camera.")

print("Press 'q' to quit.")

# Define a minimum contour area threshold
MIN_CONTOUR_AREA = 5000  # Adjust this value based on your application

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)  # Invert the camera view (mirror effect)

    # Apply a slight blur to the frame to reduce noise
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

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

    # If any valid contours are found, process the largest one
    if contours:
        # Get the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to reduce noise
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Draw the outline of the red object
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

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
            # Draw the red point
            cv2.circle(frame, highest_left_image, 5, (0, 0, 255), -1)  # Red point for highest negative x

        if right_points:
            highest_right = max(right_points, key=lambda pt: pt[1])
            highest_right_image = (highest_right[0] + center_x, center_y - highest_right[1])
            # Draw the blue point
            cv2.circle(frame, highest_right_image, 5, (255, 0, 0), -1)  # Blue point for highest positive x

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

            # Draw the chosen segment
            cv2.polylines(frame, [chosen_segment], isClosed=False, color=(255, 255, 0), thickness=2)  # Cyan line

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

                # Define a threshold for straightness
                threshold = 5.0  # Adjust as needed
                is_straight = max_deviation < threshold

                # Display the result
                if is_straight:
                    cv2.putText(frame, "Line is Straight", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, "Line is Not Straight", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            else:
                # Vertical line case
                max_deviation = max(abs(pt[0][0] - x1) for pt in chosen_segment)
                threshold = 5.0  # Adjust as needed
                is_straight = max_deviation < threshold

                # Display the result
                if is_straight:
                    cv2.putText(frame, "Line is Straight", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, "Line is Not Straight", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # Show the frame with annotations
    cv2.imshow("Red Object Detection", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
