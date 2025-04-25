import cv2
import numpy as np
import pyrealsense2 as rs
import apriltag

def find_largest_red_object_and_get_points(color_image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Define the range for red color in HSV space
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red regions in both ranges
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the two masks to get all red areas
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Find contours of the red objects in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None  # No red objects detected

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute the centroid of the contour
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, None  # Avoid division by zero

    cx = int(M["m10"] / M["m00"])  # Centroid x
    cy = int(M["m01"] / M["m00"])  # Centroid y
    centroid = np.array([cx, cy])

    # Move contour points 5% closer to the centroid
    shrink_factor = 1  # 5% inward
    new_contour = []
    
    for point in largest_contour:
        x, y = point[0]
        original_point = np.array([x, y])
        
        # Vector from centroid to the point
        direction_vector = original_point - centroid
        new_point = centroid + direction_vector * shrink_factor  # Move towards centroid
        
        new_contour.append(new_point.astype(int))

    new_contour = np.array(new_contour, dtype=np.int32)  # Convert to int32 for OpenCV

    # Create a mask for the new smaller red object
    smallest_red_mask = np.zeros_like(red_mask)
    cv2.drawContours(smallest_red_mask, [new_contour], -1, 255, thickness=cv2.FILLED)

    # Get all (x, y) pixel coordinates for the smaller red object
    points_list = np.column_stack(np.where(smallest_red_mask == 255))
    points_list = points_list[:, [1, 0]]  # Swap columns (x, y)

    return points_list, smallest_red_mask

def red_object(color_image, min_area=5000, close_threshold=10):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Define the range for red color in HSV space
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red regions in both ranges
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the two masks to get all red areas
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Find contours of the red objects in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None  # No red objects detected

    # Filter out small contours by area
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    if not large_contours:
        return None, None  # No large enough red objects detected

    # Create a new mask to keep only the large contours
    large_red_mask = np.zeros_like(red_mask)

    # Draw the large contours on the new mask
    cv2.drawContours(large_red_mask, large_contours, -1, (255), thickness=cv2.FILLED)

    # Find the largest contour from the large ones
    largest_contour = max(large_contours, key=cv2.contourArea)

    # Draw the largest contour on the original image for visualization
    output_image = color_image.copy()
    cv2.drawContours(output_image, [largest_contour], -1, (0, 255, 0), 2)  # Green outline

    # Get all (x, y) pixel coordinates inside the largest red object
    points_list = np.column_stack(np.where(large_red_mask == 255))
    points_list = points_list[:, [1, 0]]  # Swap columns to (x, y)

    # Calculate distance from each point to the nearest point on the contour
    # Use the cv2.pointPolygonTest function to compute the distance
    filtered_points = []
    for point in points_list:
        dist = cv2.pointPolygonTest(largest_contour, (int(point[0]), int(point[1])), True)  # Signed distance
        if abs(dist) >= close_threshold:  # Exclude points too close to the contour
            filtered_points.append(point)

    # Convert filtered points to a numpy array
    filtered_points = np.array(filtered_points)

    return filtered_points, large_red_mask

def get_depth_for_multiple_points(points_list, color_image, depth_frame, detector, camera_matrix, dist_coeffs, color_intrinsics):
    # Create a dictionary to store the results
    depth_data = []

    # Convert color frame to numpy array and grayscale
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    detections = detector.detect(gray_image)

    for i, detection in enumerate(detections):
        corners = np.array(detection.corners, dtype=np.float32)

        # Compute tag center
        tag_cx = int(np.mean(corners[:, 0]))
        tag_cy = int(np.mean(corners[:, 1]))

        # Get depth at the tag center
        Z_real_tag = depth_frame.get_distance(tag_cx, tag_cy)

        if Z_real_tag <= 0:
            continue

        # Define normalized 3D object points for the tag (no assumed size)
        object_points = np.array([
            [-1, -1, 0],
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
        ], dtype=np.float32)

        # SolvePnP to get pose of the tag
        ret, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)

        if ret:
            # Scale translation vector using actual depth
            scale_factor = Z_real_tag / tvec[2]
            tvec_real = tvec * scale_factor

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # Loop over each point in the list
            for p_x, p_y in points_list:
                # Only process points inside the red mask
                if p_x < depth_frame.width - 1 and p_y < depth_frame.height - 1:
                    
                    Z_real = depth_frame.get_distance(p_x, p_y)  # Depth at (p_x, p_y)

                    if Z_real > 0:
                        fx, fy = color_intrinsics.fx, color_intrinsics.fy
                        cx, cy = color_intrinsics.ppx, color_intrinsics.ppy

                        # Compute camera coordinates for point
                        X_camera = (p_x - cx) * Z_real / fx
                        Y_camera = (p_y - cy) * Z_real / fy
                        Z_camera = Z_real
                        point_camera = np.array([[X_camera], [Y_camera], [Z_camera]])

                        # **Convert camera coordinate to AprilTag coordinate**
                        point_tag = np.dot(rotation_matrix.T, (point_camera - tvec_real))
                        point_tag[2][0] = -point_tag[2][0]
                        def model(X, a, b, c, d):
                            # Extract fitted parameters
                            x, y = X

                            return a * x + b * np.exp(c * y) + d
                        if i == 0:
                            a, b, c, d = 0.03092494, -0.03941634,  0.56732075,  0.03739654
                        elif i == 1:
                            a, b, c, d = -0.01910277, 0.01119914, 1.9034686,  -0.01428886
                            
                        h_error = model((point_tag[0][0], point_tag[1][0]), a, b, c, d)
                        h_real = point_tag[2][0] - h_error

                        # Store the result in the dictionary
                        depth_data.append((p_x, p_y, h_real))  # Store z value for the pixel (p_x, p_y)

    return depth_data

def find_mean_change_in_height(depth_data):
    
    # Sort by Z value (height)
    depth_data_sorted = sorted(depth_data, key=lambda x: x[2])  # Sort by Z value
    
    height_differences = []
    above_mean = []
    low_diff = 100000
    high_diff = -100000
    x_diff = 0
    y_diff = 0
    # Calculate the height differences between consecutive points
    for i in range(1, len(depth_data_sorted)):
        prev_x, prev_y, prev_z = depth_data_sorted[i - 1]
        curr_x, curr_y, curr_z = depth_data_sorted[i]

        # Calculate the difference in height
        diff = abs(curr_z - prev_z)

        if diff < low_diff:
            low_diff = diff

        if diff > high_diff:
            high_diff = diff
            x_diff = curr_x
            y_diff = curr_y


        height_differences.append(diff)

    print(high_diff, x_diff, y_diff)

    # Calculate the mean of the height differences
    mean_change_in_height = np.mean(height_differences) if height_differences else 0

    threshold = high_diff * 0.9  # 15% above the mean change in height

    # Check for points where the height change exceeds the threshold
    for i in range(1, len(depth_data_sorted)):
        prev_x, prev_y, prev_z = depth_data_sorted[i - 1]
        curr_x, curr_y, curr_z = depth_data_sorted[i]
        
        diff = abs(curr_z - prev_z)
        
        if diff > threshold:
            above_mean.append((prev_x, prev_y, prev_z))  # Store points with significant height change
        
    return mean_change_in_height, above_mean

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

# Get camera intrinsics
profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color)
color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                          [0, color_intrinsics.fy, color_intrinsics.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(color_intrinsics.coeffs)

depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 5)

# AprilTag detector setup
detector = apriltag.Detector()

# Initialize the variables for saving frames
color_image_saved = None
depth_frame_saved = None
print("Press F to take picture")



while True:
    # Capture frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    # Convert color frame to numpy array
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imshow("Cam", color_image)

    # Wait for 'f' key press to capture color and depth frames
    key = cv2.waitKey(1) & 0xFF
    if key == ord('f'):  # If 'F' key is pressed
        color_image_saved = color_image  # Save color image
        depth_frame_saved = depth_frame  # Save depth frame
        print("Frames captured and saved.")
        break  # Exit the loop after capturing


# Ensure that both frames are captured
if color_image_saved is not None and depth_frame_saved is not None:
    # Step 1: Find the largest red object and get its points list
    points_list, largest_red_mask = red_object(color_image_saved)
    contours_red_object, _ = cv2.findContours(largest_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color_image, contours_red_object, -1, (255, 0, 0), 2)

    cv2.imshow("Cam", color_image)

    if points_list is not None:
        # Step 2: Use the saved frames and points list to get the depth values
        # Example usage with your depth_dict:
        depth_data = get_depth_for_multiple_points(points_list, color_image_saved, depth_frame_saved, detector, camera_matrix, dist_coeffs, color_intrinsics)

        # Step 1: Find the mean change in height between consecutive points
        mean_change_in_height, above_mean_points = find_mean_change_in_height(depth_data)
        print(f"Mean change in height: {mean_change_in_height}")

        contour_mask = np.zeros(color_image.shape[:2], dtype=np.uint8)

        z_values = [point[2] for point in depth_data]
        mean_z = np.mean(z_values)
        low_z = min(z_values)
        high_z = max(z_values)

        # Draw points with smooth color transition
        for x, y, z in depth_data:
            z = min(max(z, 0), 1)  # Clamp values
            high_z = min(max(high_z, 0), 1)  # Clamp values
            low_z = min(max(low_z, 0), 1)  # Clamp values
            ratio = (z - low_z) / (high_z - low_z)  # Normalize ratio between 0 and 1
            ratio = min(max(ratio, 0), 1)  # Clamp values
            low_color=(0, 255, 0)
            high_color=(255, 0, 0)
            r = int(low_color[0] + ratio * (high_color[0] - low_color[0]))
            g = int(low_color[1] + ratio * (high_color[1] - low_color[1]))
            b = int(low_color[2] + ratio * (high_color[2] - low_color[2]))
            color = (r, g, b)
            cv2.circle(color_image_saved, (x, y), 3, color, -1)  # Draw point

        # Step 3: Show the image with contours
        cv2.imshow("Image with Height Change Contours", color_image_saved)
        cv2.imwrite("depth_frame.png", color_image_saved)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    print("No frames captured. Please press 'f' to capture frames.")
