import cv2
import numpy as np
import pyrealsense2 as rs
import apriltag
import csv

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Get depth scale for conversion
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# AprilTag detector
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)

# Set buffer margin (avoid close proximity to frame edges)
buffer_margin = 0.1  # 10% of the image width and height



while True:
    # Capture frames
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    results = detector.detect(gray)

    if len(results) >= 2:
        # Get first two detected tags
        tag1, tag2 = results[:2]

        # Use detection.center for accurate tag centers
        center_1 = np.array(tag1.center, dtype=np.int32)
        center_2 = np.array(tag2.center, dtype=np.int32)

        # Image dimensions
        img_height, img_width, _ = frame.shape
        middle_x = img_width // 2  # Vertical middle of the frame

        # Buffer area to avoid close proximity to edges
        min_x = int(img_width * buffer_margin)
        max_x = int(img_width * (1 - buffer_margin))
        min_y = int(img_height * buffer_margin)
        max_y = int(img_height * (1 - buffer_margin))

        # Define an extension depth (~2/3 down)
        extension_depth = int(img_height * 0.67)

        # Compute extended bottom points **exactly from the tag center**
        extended_center_1 = (center_1[0], extension_depth)
        extended_center_2 = (center_2[0], extension_depth)

        # Define a **straight top line** between the two AprilTag centers
        center_between = ((center_1 + center_2) // 2).astype(int)

        # Define the left and right rectangular regions, keeping them within the safe margin
        contour_left = np.array([
            (max(center_1[0], min_x), min(max(center_1[1], min_y), max_y)), 
            (max(center_between[0], min_x), min(max(center_between[1], min_y), max_y)),  
            (max(min_x, middle_x), extension_depth), 
            extended_center_1  
        ])
        
        contour_right = np.array([
            (min(center_between[0], max_x), min(max(center_between[1], min_y), max_y)), 
            (min(center_2[0], max_x), min(max(center_2[1], min_y), max_y)), 
            extended_center_2, (middle_x, extension_depth)  
        ])

        # Draw contours
        cv2.drawContours(frame, [contour_left], -1, (0, 255, 0), 2)  # Green
        cv2.drawContours(frame, [contour_right], -1, (0, 0, 255), 2)  # Red

        # Draw the **straight** top line
        cv2.line(frame, tuple(center_1), tuple(center_2), (255, 255, 0), 2)

        # Create masks for the regions
        mask_left = np.zeros_like(gray, dtype=np.uint8)
        mask_right = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillPoly(mask_left, [contour_left], 255)
        cv2.fillPoly(mask_right, [contour_right], 255)

        # Extract depth values for each region
        region_left_data = []
        region_right_data = []

        for y in range(img_height):
            for x in range(img_width):
                depth_value = depth_image[y, x] * depth_scale  # Convert to meters
                if mask_left[y, x] > 0:
                    region_left_data.append((x, y, depth_value))
                elif mask_right[y, x] > 0:
                    region_right_data.append((x, y, depth_value))

        # Save depth data to CSV files
        with open("region_left.csv", "w", newline="") as f_left:
            writer = csv.writer(f_left)
            writer.writerow(["x", "y", "depth"])
            writer.writerows(region_left_data)

        with open("region_right.csv", "w", newline="") as f_right:
            writer = csv.writer(f_right)
            writer.writerow(["x", "y", "depth"])
            writer.writerows(region_right_data)

    # Display result
    cv2.imshow("AprilTag Rectangles with Depth Data (RealSense L515)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
