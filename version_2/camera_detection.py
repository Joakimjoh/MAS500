import numpy as np
import math
import cv2

is_straight = False

def get_largest_contour(contours, min_contour=5000):
    # Filter by size
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [c for c in contours if cv2.contourArea(c) > min_contour]

    if contours:
        # Get the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to reduce noise
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)    

        return largest_contour
    
    return None

def detect_stretched(frame):
    global is_straight  # Ensure the function modifies the global variable

    while True:
        contours = frame.detect_red_objects()

        if contours:
            largest_contour = get_largest_contour(contours)
            
            if largest_contour is not None:
                frame.objects["Object1"] = (largest_contour, "blue")

                # Find corners relative to the center
                corners_relative = [(pt[0][0] - frame.center_x, frame.center_y - pt[0][1]) for pt in largest_contour]

                # Filter points by x-coordinate (negative and positive)
                left_points = [pt for pt in corners_relative if pt[0] < 0]  # Negative x
                right_points = [pt for pt in corners_relative if pt[0] > 0]  # Positive x

                # Find the highest y-value for each group
                highest_left_image = None
                highest_right_image = None

                if left_points:
                    highest_left = max(left_points, key=lambda pt: pt[1])
                    highest_left_image = (highest_left[0] + frame.center_x, frame.center_y - highest_left[1])
                    
                    # Draw the red point
                    frame.points["Point3"] = (highest_left_image[0], highest_left_image[1], "red")

                if right_points:
                    highest_right = max(right_points, key=lambda pt: pt[1])
                    highest_right_image = (highest_right[0] + frame.center_x, frame.center_y - highest_right[1])
                    
                    # Draw the blue point
                    frame.points["Point4"] = (highest_right_image[0], highest_right_image[1], "blue")

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
                    frame.objects["Object2"] = (chosen_segment, "cyan")

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

                            frame.text = "Line is Straight"
                        else:
                            frame.text = "Line is Not Straight"

def get_left_right_point(frame):
    contours = frame.detect_red_objects()

    if not contours:
        return None, None
    
    largest_contour = get_largest_contour(contours)

    if largest_contour is None:
        return None, None
    
    # Add contour to frame
    frame.objects["Object1"] = (largest_contour, "blue")

    # Compute centroid of the contour
    M = cv2.moments(largest_contour)
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])

    # Find extreme left and right points
    left_point = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    right_point = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])

    # Move points slightly toward centroid (10% of the way)
    def move_towards_centroid(point, centroid, factor=0.1):
        return (
            int(point[0] + (centroid[0] - point[0]) * factor),
            int(point[1] + (centroid[1] - point[1]) * factor),
        )

    left_point_inside = move_towards_centroid(left_point, (centroid_x, centroid_y))
    right_point_inside = move_towards_centroid(right_point, (centroid_x, centroid_y))

    # Add points to frame
    frame.points["Point1"] = (left_point_inside[0], left_point_inside[1], "green")
    frame.points["Point2"] = (right_point_inside[0], right_point_inside[1], "red")

    return left_point_inside, right_point_inside
        