from data import get_depth_pixel, meter_to_pixel, pixel_to_meter, transform_camera_to_world, transform_world_to_camera
from camera import get_intrinsics, get_frames
import numpy as np
import math
import cv2

is_straight = False

def detect_red_object(pipeline, align, profile, depth_data):
    """Detect red objects and get adjusted points inside the contour."""
    global prev_left_point, prev_right_point  # Use previous frame points

    depth_frame, color_frame = get_frames(pipeline, align)

    intrinsics_depth, intrinsics_color, depth_to_color_extrinsics = get_intrinsics(pipeline)

    # Convert image to HSV
    hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)

    # Define range for red color
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    mask = mask1 | mask2

    # Reduce noise with blur and morphological operations
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Filter by size
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = [c for c in contours if cv2.contourArea(c) > 5000]

        if contours:
            largest_contour = contours[0]

            # Draw full outline without simplification
            cv2.drawContours(color_frame, [largest_contour], -1, (255, 0, 0), 2)  # Blue outline

            # Compute centroid of the contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
            else:
                return None, None, None, None, None, None, None, None  # Avoid division by zero

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

            # Find the center of the frame
            cx, cy = color_frame.shape[1] // 2, color_frame.shape[0] // 2

            # Get corresponding depth pixel
            depth_pixel_center = get_depth_pixel(cx, cy, depth_frame, intrinsics_depth, intrinsics_color, depth_to_color_extrinsics, profile)
            
            if depth_pixel_center:
                cx_d, cy_d = map(int, depth_pixel_center)

                # Get real-world coordinates of the center point
                center_coords = pixel_to_meter(cx_d, cy_d, depth_frame, intrinsics_depth)

                if center_coords:
                    cx_m, cy_m, cz_m = center_coords

                    # Draw a circle at the center point
                    cv2.circle(color_frame, (cx, cy), 5, (255, 0, 0), -1)  # Blue circle
                    cv2.putText(color_frame, f"({cx_m:.3f}, {cy_m:.3f}, {cz_m:.3f})", 
                                (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


            lu, lv = left_point_inside
            ru, rv = right_point_inside

            depth_pixel_left = get_depth_pixel(lu, lv, depth_frame, intrinsics_depth, intrinsics_color, depth_to_color_extrinsics, profile)
            depth_pixel_right = get_depth_pixel(ru, rv, depth_frame, intrinsics_depth, intrinsics_color, depth_to_color_extrinsics,profile)

            lrel_x = 0
            lrel_y = 0 
            rrel_x = 0
            rrel_y = 0
            
            if depth_pixel_left and depth_pixel_right and depth_pixel_center:
                ldu, ldv = map(int, depth_pixel_left)
                rdu, rdv = map(int, depth_pixel_right)

                # Get real-world coordinates from pixel coordinates
                world_coords_left = pixel_to_meter(ldu, ldv, depth_frame, intrinsics_depth)
                world_coords_right = pixel_to_meter(rdu, rdv, depth_frame, intrinsics_depth)

                if world_coords_left and world_coords_right:
                    # Apply transformation to world coordinates
                    lx_m, ly_m, lz_m = transform_camera_to_world(world_coords_left, depth_data)
                    rx_m, ry_m, rz_m = transform_camera_to_world(world_coords_right, depth_data)

                    # Compute relative position from the center
                    lrel_x = lx_m - cx_m
                    lrel_y = ly_m - cy_m

                    rrel_x = rx_m - cx_m
                    rrel_y = ry_m - cy_m


            # Transform from meter coordinates to pixel coordinates
            lx_m = lrel_x + cx_m
            ly_m = lrel_y + cy_m

            world_coords_left = transform_world_to_camera((lx_m, ly_m, lz_m), depth_data)

            ldu, ldv = meter_to_pixel(world_coords_left, intrinsics_depth)

            depth_pixel_left = ldu, ldv

            print(depth_pixel_left)

            # Draw a circle at the center point
            cv2.circle(color_frame, depth_pixel_left, 5, (255, 0, 0), -1)  # Blue circle

            # Update previous points for next frame
            prev_left_point, prev_right_point = left_point_inside, right_point_inside

            # Transform coordinates to be centered at (0, 0)
            left_point_transformed = (left_point_inside[0] - cx, left_point_inside[1] - cy)
            right_point_transformed = (right_point_inside[0] - cx, right_point_inside[1] - cy)

            # Get depth at the adjusted points (median filter for stability)
            depth_values = [
                depth_frame.get_distance(left_point_inside[0], left_point_inside[1]),
                depth_frame.get_distance(right_point_inside[0], right_point_inside[1])
            ]
            left_depth, right_depth = np.median(depth_values), np.median(depth_values)  # Use median for stability

            if lrel_x and lrel_y and rrel_x and rrel_y: 
                return left_depth, right_depth, left_point_transformed, right_point_transformed, lrel_x, lrel_y, rrel_x, rrel_y

    return None, None, None, None, None, None, None, None   # No red object found

def detect_stretched(pipeline, align):
    global is_straight  # Ensure the function modifies the global variable

    MIN_CONTOUR_AREA = 5000  # Minimum contour area threshold

    while True:
        # Get frames
        depth_frame, frame = get_frames(pipeline, align)

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

            # Show updated frame
            cv2.imshow("Check if line is straight", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()