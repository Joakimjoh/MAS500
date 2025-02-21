import cv2
import numpy as np
import pyrealsense2 as rs
import apriltag

# RealSense pipeline setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Enable depth stream
pipeline.start(config)

tag_poses = {}

# Load camera intrinsics
profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color)
color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                          [0, color_intrinsics.fy, color_intrinsics.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(color_intrinsics.coeffs)

# Get the device
device = pipeline.get_active_profile().get_device()

depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 5)

# AprilTag detector setup
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)

# Define the 3D points of the cube in the AprilTag's coordinate system
tag_size = 0.022  # 5cm AprilTag

cube_points_3D = np.array([
    [0, 0, 0],  # Bottom-left corner of tag
    [tag_size, 0, 0],  # Bottom-right corner of tag
    [tag_size, tag_size, 0],  # Top-right corner of tag
    [0, tag_size, 0],  # Top-left corner of tag
    [0, 0, -tag_size]  # Elevated corners (top face)
], dtype=np.float32)

# RealSense alignment (depth to color)
align_to = rs.stream.color
align = rs.align(align_to)

p_x = 350
p_y = 200

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not depth_frame:
        print("No depth frame available!")
        continue

    # Convert to numpy array and grayscale
    color_image = np.asanyarray(color_frame.get_data())
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    detections = detector.detect(gray_image)

    for detection in detections:
        # Get AprilTag corner positions
        corners = np.array(detection.corners, dtype=np.float32)

        # SolvePnP to get the pose of the tag
        ret, rvec, tvec = cv2.solvePnP(cube_points_3D[:4], corners, camera_matrix, dist_coeffs)

        if ret:
            # Project 3D cube points onto 2D image
            imgpts, _ = cv2.projectPoints(cube_points_3D, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # Draw the 4 base edges (main outline in yellow)
            cv2.polylines(color_image, [imgpts[:4]], True, (255, 0, 255), 2)  # Yellow for the base

            # Draw top face in yellow
            cv2.polylines(color_image, [imgpts[4:]], True, (255, 0, 255), 2)

            # **Color three sides of the cube to represent the coordinate system**
            cv2.line(color_image, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)  # X-axis (red)
            cv2.line(color_image, tuple(imgpts[0]), tuple(imgpts[3]), (0, 255, 0), 3)  # Y-axis (green)
            cv2.line(color_image, tuple(imgpts[0]), tuple(imgpts[4]), (255, 0, 0), 3)  # Z-axis (blue)

            # Store the pose of each detected tag
            tag_poses[detection.tag_id] = (rvec, tvec)

            # Invert the transformation (rotation and translation) from camera to tag 1
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            rotation_matrix_inv = np.linalg.inv(rotation_matrix)
            tvec_inv = -rotation_matrix_inv @ tvec

            # Display the tag ID
            cv2.putText(color_image, f"ID: {detection.tag_id}", tuple(corners[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

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
    # Display result
    cv2.imshow("AprilTag 3D Cube with Coordinate System", color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
