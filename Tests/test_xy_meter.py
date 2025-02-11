import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Get depth sensor intrinsics and extrinsics
profile = pipeline.get_active_profile()
depth_stream = profile.get_stream(rs.stream.depth)
color_stream = profile.get_stream(rs.stream.color)
intrinsics_depth = depth_stream.as_video_stream_profile().get_intrinsics()
intrinsics_color = color_stream.as_video_stream_profile().get_intrinsics()
depth_to_color_extrinsics = depth_stream.get_extrinsics_to(color_stream)

def transform_camera_to_world(point, theta=45):
    """Convert RealSense (X, Y, Z) to world coordinates, correcting for tilt."""
    X, Y, Z = point
    theta_rad = np.radians(theta)  # Convert degrees to radians

    # Rotation matrix for tilt correction around X-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(theta_rad), -np.sin(theta_rad)],
        [0, np.sin(theta_rad), np.cos(theta_rad)]
    ])

    # Apply the rotation matrix
    world_coords = np.dot(rotation_matrix, np.array([X, Y, Z]))
    return tuple(world_coords)

def get_depth_pixel(u, v, depth_frame, intrinsics_depth, intrinsics_color, depth_to_color_extrinsics):
    """Convert color pixel (u, v) to corresponding depth pixel (du, dv)."""
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    depth_value = depth_frame.get_distance(u, v)  # Get depth at color pixel
    if depth_value == 0:
        return None  # No valid depth data

    # Compute color-to-depth extrinsics (inverse of depth-to-color)
    color_to_depth_extrinsics = rs.extrinsics()
    color_to_depth_extrinsics.rotation = np.linalg.inv(np.array(depth_to_color_extrinsics.rotation).reshape(3, 3)).flatten().tolist()
    color_to_depth_extrinsics.translation = (-np.array(depth_to_color_extrinsics.translation)).tolist()

    # Convert color pixel to depth pixel
    depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(
        depth_frame.as_frame().get_data(), depth_scale, 0.1, 4.0,  # Min & max depth range
        intrinsics_depth, intrinsics_color, depth_to_color_extrinsics, color_to_depth_extrinsics,
        [u, v]
    )

    return tuple(map(int, depth_pixel))  # Ensure integers for pixel indices

def pixel_to_meter(u, v, depth_frame, intrinsics):
    """Convert depth pixel (du, dv) to real-world (X, Y, Z) in meters using RealSense function."""
    depth_value = depth_frame.get_distance(u, v)  # Depth in meters
    if depth_value == 0:
        return None  # No valid depth data

    point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_value)
    return tuple(point)  # Returns (X, Y, Z) in meters

try:
    while True:
        # Capture frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Find the center of the frame
        cx, cy = color_image.shape[1] // 2, color_image.shape[0] // 2

        # Get corresponding depth pixel
        depth_pixel = get_depth_pixel(cx, cy, depth_frame, intrinsics_depth, intrinsics_color, depth_to_color_extrinsics)
        
        if depth_pixel:
            cx_d, cy_d = map(int, depth_pixel)

            # Get real-world coordinates of the center point
            center_coords = pixel_to_meter(cx_d, cy_d, depth_frame, intrinsics_depth)

            if center_coords:
                cx_m, cy_m, cz_m = center_coords
                print(f"Center Position: X={cx_m:.3f}m, Y={cy_m:.3f}m, Z={cz_m:.3f}m")

                # Draw a circle at the center point
                cv2.circle(color_image, (cx, cy), 5, (255, 0, 0), -1)  # Blue circle
                cv2.putText(color_image, f"({cx_m:.3f}, {cy_m:.3f}, {cz_m:.3f})", 
                            (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Define red color range in HSV
        lower_red1 = np.array([0, 120, 70])   # Lower range of red
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])  # Upper range of red
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2  # Combine both masks

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (assumed to be the red object)
            largest_contour = max(contours, key=cv2.contourArea)

            # Find the leftmost point (in pixels)
            leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
            u, v = leftmost  # Pixel coordinates

            # Get corresponding depth pixel
            depth_pixel = get_depth_pixel(u, v, depth_frame, intrinsics_depth, intrinsics_color, depth_to_color_extrinsics)
            
            if depth_pixel:
                du, dv = map(int, depth_pixel)

                # Get real-world coordinates from pixel coordinates
                world_coords = pixel_to_meter(du, dv, depth_frame, intrinsics_depth)

                if world_coords:
                    # Apply transformation to world coordinates
                    x_m, y_m, z_m = transform_camera_to_world(world_coords)

                    # Compute relative position from the center
                    rel_x = x_m - cx_m
                    rel_y = y_m - cy_m

                    print(f"Leftmost Red Object Relative Position: ΔX={rel_x:.3f}m, ΔY={rel_y:.3f}m")

                    # Draw a circle at the leftmost point
                    cv2.circle(color_image, (u, v), 5, (0, 255, 0), -1)
                    cv2.putText(color_image, f"({rel_x:.3f}, {rel_y:.3f})", 
                                (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the results
        cv2.imshow("Red Object Detection", color_image)
        cv2.imshow("Red Mask", mask)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop pipeline and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
