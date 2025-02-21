import cv2
import numpy as np
import pyrealsense2 as rs
import apriltag

# RealSense pipeline setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Load camera intrinsics
profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                          [0, intrinsics.fy, intrinsics.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(intrinsics.coeffs)

# AprilTag detector setup
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)

# Define the 3D points of the cube in the AprilTag's coordinate system
tag_size = 0.022  # 5cm AprilTag

# Define 3D points for cube base (on tag) and top (elevated)
cube_points_3D = np.array([
    [0, 0, 0],  # Bottom-left corner of tag
    [tag_size, 0, 0],  # Bottom-right corner of tag
    [tag_size, tag_size, 0],  # Top-right corner of tag
    [0, tag_size, 0],  # Top-left corner of tag
    [0, 0, -tag_size],  # Elevated corners (top face)
    [tag_size, 0, -tag_size],
    [tag_size, tag_size, -tag_size],
    [0, tag_size, -tag_size]
], dtype=np.float32)

# Define 3D coordinate axes at the **top of the cube**
axes_3D = np.float32([
    [tag_size / 2, tag_size / 2, -tag_size],  # Origin at cube top center
    [tag_size / 2 + 0.03, tag_size / 2, -tag_size],  # X-axis (red)
    [tag_size / 2, tag_size / 2 + 0.03, -tag_size],  # Y-axis (green)
    [tag_size / 2, tag_size / 2, -tag_size - 0.03]   # Z-axis (blue)
])

tag_poses = {}

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
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

            # Project 3D coordinate axes onto the image
            imgpts_axes, _ = cv2.projectPoints(axes_3D, rvec, tvec, camera_matrix, dist_coeffs)

            # Draw the 4 base edges (main outline in yellow)
            cv2.polylines(color_image, [imgpts[:4]], True, (255, 0, 255), 2)  # Yellow for the base

            # Draw top face in yellow
            cv2.polylines(color_image, [imgpts[4:]], True, (255, 0, 255), 2)

            # Draw vertical lines (connecting base to top)
            for i in range(4):
                cv2.line(color_image, tuple(imgpts[i]), tuple(imgpts[i+4]), (255, 0, 255), 2)  # Magenta for verticals

            # **Color three sides of the cube to represent the coordinate system**
            # Red (X-axis): Bottom-front to bottom-right (front face)
            cv2.line(color_image, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)  # X-axis (red)
            # Green (Y-axis): Bottom-left to top-left (left face)
            cv2.line(color_image, tuple(imgpts[0]), tuple(imgpts[3]), (0, 255, 0), 3)  # Y-axis (green)
            # Blue (Z-axis): Bottom-left to top-center (backward direction)
            cv2.line(color_image, tuple(imgpts[0]), tuple(imgpts[4]), (255, 0, 0), 3)  # Z-axis (blue)

            # Store the pose of each detected tag
            tag_poses[detection.tag_id] = (rvec, tvec)

            # If Tag 1 and Tag 2 both detected, compute relative position of Tag 2 in Tag 1's coordinate system
            if 1 in tag_poses and 2 in tag_poses:
                rvec1, tvec1 = tag_poses[1]  # Pose of Tag 1
                rvec2, tvec2 = tag_poses[2]  # Pose of Tag 2

                # Convert rotation vectors to rotation matrices
                rotation_matrix1, _ = cv2.Rodrigues(rvec1)
                rotation_matrix2, _ = cv2.Rodrigues(rvec2)

                # Compute the relative transformation (Tag 2 in Tag 1's coordinate system)
                relative_rotation_matrix = np.dot(rotation_matrix1.T, rotation_matrix2)  # Inverse of Tag 1's rotation matrix
                relative_translation = np.dot(rotation_matrix1.T, (tvec2 - tvec1))  # Translate Tag 2 by the relative rotation

                print(f"\n📌 Tag 2 in Tag 1's Coordinate System:")
                print(f"Position (meters): X={relative_translation[0][0]:.3f}, Y={relative_translation[1][0]:.3f}, Z={relative_translation[2][0]:.3f}")

            # Put the tag ID text
            cv2.putText(color_image, f"ID: {detection.tag_id}", tuple(corners[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display result
    cv2.imshow("AprilTag 3D Cube with Coordinate System", color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
