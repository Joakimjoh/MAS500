import cv2
import numpy as np
import pyrealsense2 as rs
import apriltag

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

# ** Initial pixel coordinate **
p_x, p_y = 350, 200  # Example pixel coordinate

h_list = []

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    # Convert color frame to numpy array and grayscale
    color_image = np.asanyarray(color_frame.get_data())
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    detections = detector.detect(gray_image)

    for i, detection in enumerate(detections):
        corners = np.array(detection.corners, dtype=np.float32)

        # Compute tag center
        tag_cx = int(np.mean(corners[:, 0]))
        tag_cy = int(np.mean(corners[:, 1]))

        tag_size = 2

        # Define normalized 3D object points for the tag (no assumed size)
        object_points = np.array([
            [-tag_size/2, -tag_size/2, 0],
            [tag_size/2, -tag_size/2, 0],
            [tag_size/2, tag_size/2, 0],
            [-tag_size/2, tag_size/2, 0]  # Normalized tag corners
        ], dtype=np.float32)

        object_points_3d = np.array([
            [-tag_size/2, -tag_size/2, 0],
            [tag_size/2, -tag_size/2, 0],
            [tag_size/2, tag_size/2, 0],
            [-tag_size/2, tag_size/2, 0],
            [-tag_size/2, -tag_size/2, -tag_size/2]  # Normalized tag corners
        ], dtype=np.float32)

        # Detect tag pose using SolvePnP
        ret, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)

        if ret:
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # Calculate real-world depth from the tag center (use median of surrounding points as depth)
            depth_values = []
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    depth = depth_frame.get_distance(tag_cx + dx, tag_cy + dy)
                    if depth > 0:
                        depth_values.append(depth)

            if depth_values:
                Z_real_tag = np.median(depth_values)  # Median depth value

            # Compute scale factor based on depth
            scale_factor = Z_real_tag / tvec[2]  # Use depth at tag's center
            tvec_real = tvec * scale_factor  # Apply the scaling factor to the translation vector

            imgpts, _ = cv2.projectPoints(object_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            depth_values = []
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    depth = depth_frame.get_distance(p_x + dx, p_y + dy)
                    if depth > 0:
                        depth_values.append(depth)

            if depth_values:
                Z_real = np.median(depth_values)  # Use median to remove outliers

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

                # Draw the input pixel coordinate on the image
                cv2.circle(color_image, (p_x, p_y), 5, (0, 255, 0), 2)

                cv2.putText(color_image, f"X={point_tag[0][0]:.3f} Y={point_tag[1][0]:.3f} Z={point_tag[2][0]:.3f}", 
                        (10, (i+1)*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.line(color_image, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)  # X-axis (red)
            cv2.line(color_image, tuple(imgpts[0]), tuple(imgpts[3]), (0, 255, 0), 3)  # Y-axis (green)
            cv2.line(color_image, tuple(imgpts[0]), tuple(imgpts[4]), (255, 0, 0), 3)  # Z-axis (blue)

    # Show image
    cv2.imshow("AprilTag Detection", color_image)

    # **Keyboard Input for Moving p_x, p_y**
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 81:  # Left Arrow
        p_x = max(p_x - 5, 0)
    elif key == 83:  # Right Arrow
        p_x = min(p_x + 5, 639)
    elif key == 82:  # Up Arrow
        p_y = max(p_y - 5, 0)
    elif key == 84:  # Down Arrow
        p_y = min(p_y + 5, 479)

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
