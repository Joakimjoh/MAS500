import pyrealsense2 as rs
import numpy as np
import cv2
import scipy.optimize

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline_profile = pipeline.start(config)

# Configure the depth sensor for short-range preset
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
if depth_sensor.supports(rs.option.visual_preset):
    depth_sensor.set_option(rs.option.visual_preset, rs.l500_visual_preset.short_range)

# Get the depth scale from the device (meters to mm conversion)
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth Scale: {depth_scale} meters per unit")

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image_mm = (depth_image * depth_scale * 1000).astype(np.float64)  # Convert to mm
        color_image = np.asanyarray(color_frame.get_data())

        # Apply median blur to reduce noise
        depth_image_mm = cv2.medianBlur(depth_image_mm.astype(np.uint8), 5)

        # Convert the color image to HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Define the range for detecting red color
        lower_red1 = np.array([0, 120, 70])  # Lower range for red
        upper_red1 = np.array([10, 255, 255])  # Upper range for red
        lower_red2 = np.array([170, 120, 70])  # Second range for red
        upper_red2 = np.array([180, 255, 255])  # Second upper range for red

        # Create masks to detect red color
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = mask1 | mask2

        # Find contours of the red objects
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Ignore small objects
                # Draw the outline of the red object
                cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2)

                # Create a mask for the current red object
                object_mask = np.zeros_like(depth_image_mm, dtype=np.uint8)
                cv2.drawContours(object_mask, [contour], -1, 255, thickness=cv2.FILLED)

                # Mask the depth image
                depth_object = cv2.bitwise_and(depth_image_mm, depth_image_mm, mask=object_mask)

                # Fit a plane to normalize depth
                points = np.column_stack(np.nonzero(depth_object))
                depths = depth_object[points[:, 0], points[:, 1]]

                def plane(params, x, y):
                    a, b, c = params
                    return a * x + b * y + c

                def error(params, x, y, z):
                    return z - plane(params, x, y)

                x, y = points[:, 1], points[:, 0]
                params, _ = scipy.optimize.leastsq(error, [0, 0, 1], args=(x, y, depths))
                a, b, c = params

                # Generate a full-size plane for the depth image
                xx, yy = np.meshgrid(np.arange(depth_object.shape[1]), np.arange(depth_object.shape[0]))
                fitted_plane = a * xx + b * yy + c

                # Normalize the depth image using the fitted plane
                normalized_depth = np.where(depth_object > 0, depth_object - fitted_plane, 0)

                # Compute gradients for edge detection
                sobel_x = cv2.Sobel(normalized_depth, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(normalized_depth, cv2.CV_64F, 0, 1, ksize=3)
                depth_edges = np.sqrt(sobel_x**2 + sobel_y**2)
                depth_edges = (depth_edges > 5).astype(np.uint8)  # Threshold for edge detection

                # Morphological operations for cleaner edges
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                depth_edges = cv2.morphologyEx(depth_edges, cv2.MORPH_CLOSE, kernel)

                # Overlay edges on the color image
                depth_edges_visual = (depth_edges * 255).astype(np.uint8)
                depth_edges_colored = cv2.cvtColor(depth_edges_visual, cv2.COLOR_GRAY2BGR)
                depth_edges_colored[np.where((depth_edges_colored != [0, 0, 0]).all(axis=2))] = [255, 255, 255]

                color_image = cv2.addWeighted(color_image, 0.8, depth_edges_colored, 0.5, 0)

        # Display the color image with depth edges
        cv2.imshow("Depth Edges on Red Object", color_image)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
