import cv2
import numpy as np
import pyrealsense2 as rs
import apriltag
from scipy.optimize import curve_fit
import random

def get_error(params, x, y):
    # Extract fitted parameters
    A_fit, B_fit, C_fit = params

    return A_fit * np.exp(B_fit * x + C_fit * y)

def get_error_equation(x_values, y_values, z_values):
    """
    Fits an exponential model to the given x, y, z data and returns the parameters.
    """
    def exp_model(X, A, B, C):
        x, y = X
        return A * np.exp(B * x + C * y)

    params, _ = curve_fit(exp_model, (x_values, y_values), z_values, p0=(0.01, 1, 1))
    return params

def generate_points_on_outline(corners, width, height, num_points=25, margin=10):
    """
    Generate random points along the outline of a tag's section, avoiding the image edges.
    
    Parameters:
    - corners: The corners of the tag (a list of 4 points, each with x, y coordinates).
    - width: The width of the image frame.
    - height: The height of the image frame.
    - num_points: The number of points to generate along the outline.
    - margin: The margin to avoid being too close to the image edge.
    
    Returns:
    - points: List of (x, y) tuples representing the generated points.
    """
    points = []
    
    # Ensure margin is respected by limiting the random generation range
    margin = max(margin, 0)  # Margin can't be negative

    # Loop over the edges of the quadrilateral defined by corners
    for i in range(4):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 4]

        # Get the length of the edge
        edge_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Generate random points along this edge
        for _ in range(num_points // 4):  # Divide by 4 to get equal distribution along each edge
            # Pick a random proportion along the edge (from 0 to 1)
            t = random.uniform(0, 1)

            # Interpolate between the corners (x1, y1) and (x2, y2)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)

            # Ensure points are within the frame and not too close to the edge
            x = min(max(x, margin), width - margin)
            y = min(max(y, margin), height - margin)

            points.append((x, y))

    return points

def get_tag_regions():
    """
    Detects two AprilTags, calculates their distance, finds 3D points in each tag's section,
    fits an exponential model to the (x, y, z) points, and returns the model parameters.
    """
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
    fx, fy = color_intrinsics.fx, color_intrinsics.fy
    cx, cy = color_intrinsics.ppx, color_intrinsics.ppy
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeffs = np.array(color_intrinsics.coeffs)

    # AprilTag detector
    detector = apriltag.Detector()

    # Object points for SolvePnP
    object_points_3d = np.array([
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0]
    ], dtype=np.float32)

    try:
        # Capture frame
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None  # No valid frames

        # Convert color frame to grayscale
        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        detections = detector.detect(gray_image)

        if len(detections) < 2:
            return None, None  # Need at least two tags

        # Get first two detected tags
        tag1, tag2 = detections[:2]

        # Compute tag centers
        center1 = np.mean(tag1.corners, axis=0)  # (x, y) of tag1
        center2 = np.mean(tag2.corners, axis=0)  # (x, y) of tag2

        # Compute midpoint line
        midpoint = (center1 + center2) / 2
        mid_x = int(midpoint[0])  # Vertical dividing line

        # SolvePnP for both tags
        def get_tag_pose(tag, center):
            corners = np.array(tag.corners, dtype=np.float32)
            ret, rvec, tvec = cv2.solvePnP(object_points_3d, corners, camera_matrix, dist_coeffs)
            if not ret:
                return None, None
            Z_real_tag = depth_frame.get_distance(center[0], center[1])
            scale_factor = Z_real_tag / tvec[2]  # Use depth at tag's center
            tvec_real = tvec * scale_factor
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            return rotation_matrix, tvec_real
        center1 = (int(center1[0]), int(center1[1]))
        center2 = (int(center2[0]), int(center2[1]))
        R1, t1 = get_tag_pose(tag1, center1)
        R2, t2 = get_tag_pose(tag2, center2)

        if R1 is None or R2 is None:
            return None, None  # Pose estimation failed

        # Compute real-world distance between tags
        dist_vector = t2 - t1
        tag_distance = np.linalg.norm(dist_vector)

        # Generate random points along the outline for both tags
        height, width = color_image.shape[:2]  # Get image height and width
        points_tag1 = generate_points_on_outline(tag1.corners, width, height, num_points=25, margin=10)
        points_tag2 = generate_points_on_outline(tag2.corners, width, height, num_points=25, margin=10)

        # Function to get 3D points in tag’s frame
        def get_3d_points(tag_rotation, tag_translation, points):
            x_vals, y_vals, z_vals = [], [], []
            for p_x, p_y in points:
                p_x = int(p_x)
                p_y = int(p_y)
                # Get depth at point
                depth_values = [depth_frame.get_distance(p_x + dx, p_y + dy)
                                for dx in range(-2, 3) for dy in range(-2, 3)
                                if depth_frame.get_distance(p_x + dx, p_y + dy) > 0]

                if not depth_values:
                    continue  # Skip invalid depth points

                Z_real = np.median(depth_values)  # Use median depth

                # Convert to camera coordinates
                X_camera = (p_x - cx) * Z_real / fx
                Y_camera = (p_y - cy) * Z_real / fy
                Z_camera = Z_real
                point_camera = np.array([[X_camera], [Y_camera], [Z_camera]])

                # Transform to AprilTag frame
                point_tag = np.dot(tag_rotation.T, (point_camera - tag_translation))

                # Store x, y, z separately
                x_vals.append(point_tag[0][0])
                y_vals.append(point_tag[1][0])
                z_vals.append(point_tag[2][0])

            return x_vals, y_vals, z_vals

        # Get x, y, z points for both tag regions
        x1, y1, z1 = get_3d_points(R1, t1, points_tag1)
        x2, y2, z2 = get_3d_points(R2, t2, points_tag2)

        # Fit exponential model to each tag’s points
        params_tag1 = get_error_equation(x1, y1, z1)
        params_tag2 = get_error_equation(x2, y2, z2)

        return params_tag1, params_tag2

    finally:
        pipeline.stop()  # Ensure the pipeline stops properly