from camera import get_frames
import pyrealsense2 as rs
import numpy as np
import parameters
import time
import json
import math
import cv2

def get_squares_depth_data(pipeline, align):
    """Captures depth data for the largest detected orange outline."""
    start_time = time.time()
    largest_contour = None
    max_area = 0

    while time.time() - start_time < 10:
        # Capture depth frame and color image
        depth_frame, color_image = get_frames(pipeline, align)

        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Define the threshold for gray area (you can adjust these values as needed)
        lower_gray = 100  # Lower threshold for gray
        upper_gray = 200  # Upper threshold for gray

        # Apply threshold to detect gray area
        _, gray_mask = cv2.threshold(gray, lower_gray, upper_gray, cv2.THRESH_BINARY)

        # Reduce noise with blur and morphological operations (optional)
        gray_mask = cv2.GaussianBlur(gray_mask, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour

        # Draw contours on the image for visualization
        if largest_contour is not None:
            cv2.drawContours(color_image, [largest_contour], -1, (0, 255, 0), 2)

        # Show detection process
        cv2.imshow("Largest Orange Outline Detection", color_image)
        cv2.waitKey(1)  # Refresh window

    # After 10 seconds, get depth data for the largest detected outline
    if largest_contour is not None:
        h, w = depth_frame.get_height(), depth_frame.get_width()
        mask = np.zeros((h, w), dtype=np.uint8)

        # Fill the mask with the largest contour (outline area)
        cv2.fillPoly(mask, [largest_contour], 255)

        # Get image center for depth calculation
        center_x = color_image.shape[1] // 2
        center_y = color_image.shape[0] // 2

        depth_data = []

        for y in range(h):
            for x in range(w):
                if mask[y, x] == 255:  # Only process pixels inside the outline
                    depth_value = depth_frame.get_distance(x, y)
                    if depth_value > 0:  # Only valid depth values
                        depth_data.append({
                            "x": x - center_x,  
                            "y": center_y - y,  
                            "depth": float(depth_value)
                        })

        # Save depth data to JSON
        with open("detected_depth_data.json", "w") as f:
            json.dump(depth_data, f, indent=4)

        cv2.destroyAllWindows()
        return depth_data  # Return the depth data of the largest outline
        
def get_coordinates_meter(point_depth, closest_point, rel_x):
    """Calculate height above the table using the closest point's depth and new point's depth."""
    # Depth of the closest point and the new point
    closest_point_depth = closest_point['depth']

    beta = math.degrees(math.asin(parameters.CAMERA_HEIGHT / closest_point_depth))

    beta_rad = math.radians(beta)

    z = abs(point_depth - closest_point_depth) * math.sin(beta_rad)

    c = math.sqrt(point_depth**2 - (parameters.CAMERA_HEIGHT - z)**2)

    if c > rel_x:
        y = math.sqrt(c**2 - rel_x**2)
    else:
        y = math.sqrt(rel_x**2 - c**2)

    x = abs(rel_x)

    return x, y, z

def meter_to_pixel(world_coords, intrinsics):
    """Convert real-world coordinates (X, Y, Z) back to depth pixel (u, v) using RealSense function."""
    # Use the RealSense function to project the 3D world point to pixel coordinates
    pixel_coords = rs.rs2_project_point_to_pixel(intrinsics, world_coords)
    
    # The output is in the form of a 2D point (u, v), so return the pixel coordinates
    return tuple(map(int, pixel_coords))

def pixel_to_meter(u, v, depth_frame, intrinsics):
    """Convert depth pixel (du, dv) to real-world (X, Y, Z) in meters using RealSense function."""
    depth_value = depth_frame.get_distance(u, v)  # Depth in meters
    if depth_value == 0:
        return None  # No valid depth data

    point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_value)
    return tuple(point)  # Returns (X, Y, Z) in meters

def get_depth_pixel(x, y, depth_frame, intrinsics_depth, intrinsics_color, depth_to_color_extrinsics, profile):
    """Convert color pixel (x, y) to corresponding depth pixel (dx, dy)."""
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    depth_value = depth_frame.get_distance(x, y)  # Get depth at color pixel
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
        [x, y]
    )

    return tuple(map(int, depth_pixel))  # Ensure integers for pixel indices

def transform_camera_to_world(point, depth_data):
    """Convert RealSense (X, Y, Z) to world coordinates, correcting for tilt."""
    X, Y, Z = point

    closest_point = depth_data.get(point, None)

    if closest_point != None:
        closest_point_depth = closest_point['depth']

        theta = math.degrees(math.acos(parameters.CAMERA_HEIGHT / closest_point_depth))
        theta_rad = math.radians(theta)

        # Rotation matrix for tilt correction around X-axis
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta_rad), -np.sin(theta_rad)],
            [0, np.sin(theta_rad), np.cos(theta_rad)]
        ])

        # Apply the rotation matrix
        world_coords = np.dot(rotation_matrix, np.array([X, Y, Z]))
        return tuple(world_coords)
    return point

def transform_world_to_camera(point, depth_data):
    """Convert world coordinates (X, Y, Z) to camera coordinates, undoing tilt correction."""
    Xw, Yw, Zw = point

    closest_point = depth_data.get(point, None)

    if closest_point != None:
        closest_point_depth = closest_point['depth']

        # Re-calculate the tilt angle based on the depth value
        theta = math.degrees(math.acos(parameters.CAMERA_HEIGHT / closest_point_depth))
        theta_rad = math.radians(theta)

        # Rotation matrix for undoing the tilt correction around X-axis
        rotation_matrix_inv = np.array([
            [1, 0, 0],
            [0, np.cos(-theta_rad), -np.sin(-theta_rad)],
            [0, np.sin(-theta_rad), np.cos(-theta_rad)]
        ])

        # Apply the inverse rotation matrix
        camera_coords = np.dot(rotation_matrix_inv, np.array([Xw, Yw, Zw]))
        return tuple(camera_coords)
    
    return point

def world_to_pixel(X, Y, Z, intrinsics):
    """Convert real-world (X, Y, Z) in meters to depth pixel (u, v) using RealSense function."""
    # Project the 3D point to 2D pixel coordinates (u, v)
    point_3d = np.array([X, Y, Z])  # World coordinates (X, Y, Z)

    # Use the RealSense function to project the 3D world point to pixel coordinates
    pixel_coords = rs.rs2_project_point_to_pixel(intrinsics, point_3d)
    
    # Ensure the returned coordinates are valid and within the image bounds
    x, y = map(int, pixel_coords)

    return x, y