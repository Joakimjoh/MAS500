import cv2
import numpy as np
import pyrealsense2 as rs

def camera_pixel_to_arm_coordsys(camera, p_x, p_y):
    """Turn pixel coordinates from camera into coordinate system of one of the robot arms"""
    # Example pixel and depth values
    depth_in_meters = depth_frame.get_distance(p_x, p_y)  # Depth value in meters

    # Draw the circle on the image (this will visualize the point on the image)
    cv2.circle(color_image, (p_x, p_y), 5, (0, 255, 0), 2)  # Green circle for the camera point

    # Get the depth stream and extrinsics
    depth_stream = profile.get_stream(rs.stream.depth)
    depth_to_color_extrinsics = depth_stream.get_extrinsics_to(color_stream)

    # Convert the extrinsics to a 4x4 matrix
    rotation_matrix = np.array(depth_to_color_extrinsics.rotation).reshape(3, 3)
    translation_vector = np.array(depth_to_color_extrinsics.translation).reshape(3, 1)
    extrinsics_matrix = np.hstack((rotation_matrix, translation_vector))
    extrinsics_matrix = np.vstack((extrinsics_matrix, np.array([0, 0, 0, 1])))

    # Get the intrinsics from depth stream
    depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
    fx = depth_intrinsics.fx  # Focal length in x (in pixels)
    fy = depth_intrinsics.fy  # Focal length in y (in pixels)
    cx = depth_intrinsics.ppx  # Principal point x (in pixels)
    cy = depth_intrinsics.ppy  # Principal point y (in pixels)

    # Convert pixel coordinates (x, y) to camera coordinates (X, Y, Z) in meters
    X_camera = (p_x - cx) * depth_in_meters / fx
    Y_camera = (p_y - cy) * depth_in_meters / fy
    Z_camera = depth_in_meters  # Z in camera coordinates is simply the depth

    # Point in depth camera's coordinate system (homogeneous coordinate)
    point_in_depth_camera = np.array([[X_camera], [Y_camera], [Z_camera], [1]])

    # Transform the point from depth camera to color camera
    point_in_color_camera = np.dot(extrinsics_matrix, point_in_depth_camera)

    # Check if Tag 1 is detected
    if 1 in tag_poses:
        rvec1, tvec1 = tag_poses[1]  # Pose of Tag 1

        # Convert rotation vector to rotation matrix (from camera to Tag 1's coordinate system)
        rotation_matrix1, _ = cv2.Rodrigues(rvec1)

        # Transform the point from color camera's coordinate system to Tag 1's coordinate system
        point_in_tag1 = np.dot(rotation_matrix1.T, (point_in_color_camera[:3] - tvec1))  # Use 3D coordinates, not homogeneous

        # Print the relative position in meters (X, Y, Z coordinates in Tag 1's system)
        print(f"Point relative to Tag 1's Coordinate System (in meters):")
        print(f"X={point_in_tag1[0][0]:.3f} m, Y={point_in_tag1[1][0]:.3f} m, Z={point_in_tag1[2][0]:.3f} m")