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

# ** Enter a pixel coordinate manually (Replace this with real inputs if needed) **
p_x, p_y = 350, 200  # Example pixel coordinate

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

    for detection in detections:
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
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]  # Normalized tag corners
        ], dtype=np.float32)

        # SolvePnP to get pose of the tag
        ret, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)

        if ret:
            # Scale translation vector using actual depth
            scale_factor = Z_real_tag / tvec[2]
            tvec_real = tvec * scale_factor

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # **Convert input pixel (p_x, p_y) to camera coordinates**
            Z_real = depth_frame.get_distance(p_x, p_y)  # Depth at (p_x, p_y)

            if Z_real > 0:
                fx, fy = color_intrinsics.fx, color_intrinsics.fy
                cx, cy = color_intrinsics.ppx, color_intrinsics.ppy

                # Compute camera coordinates for center of tag
                X_camera_tag = (tag_cx - cx) * Z_real_tag / fx
                Y_camera_tag = (tag_cy - cy) * Z_real_tag / fy
                Z_camera_tag = Z_real_tag
                point_camera_tag = np.array([[X_camera_tag], [Y_camera_tag], [Z_camera_tag]])

                # **Convert camera coordinate to AprilTag coordinate**
                tag_point_tag = np.dot(rotation_matrix.T, (point_camera_tag - tvec_real))
                tag_point_tag[2][0] = -tag_point_tag[2][0]

                # Compute camera coordinates for point
                X_camera = (p_x - cx) * Z_real / fx
                Y_camera = (p_y - cy) * Z_real / fy
                Z_camera = Z_real
                point_camera = np.array([[X_camera], [Y_camera], [Z_camera]])

                # **Convert camera coordinate to AprilTag coordinate**
                point_tag = np.dot(rotation_matrix.T, (point_camera - tvec_real))
                point_tag[2][0] = -point_tag[2][0]
                
                point_tag -= tag_point_tag

                # Draw the input pixel coordinate on the image
                cv2.circle(color_image, (p_x, p_y), 5, (0, 255, 0), 2)

                cv2.putText(color_image, f"X={point_tag[0][0]:.3f} Y={point_tag[1][0]:.3f} Z={point_tag[2][0]:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw detected AprilTag
            for i in range(4):
                cv2.line(color_image, tuple(corners[i].astype(int)), tuple(corners[(i + 1) % 4].astype(int)), (0, 255, 0), 2)

            # Display AprilTag ID
            cv2.putText(color_image, f"ID: {detection.tag_id}",
                        (tag_cx - 20, tag_cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show image
    cv2.imshow("AprilTag Detection", color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
