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

# Define normalized 3D object points for the tag (no assumed size)
object_points_3d = np.array([
    [-1, -1, 0],
    [1, -1, 0],
    [1, 1, 0],
    [-1, 1, 0],
    [-1, -1, -1]  # Normalized tag corners
], dtype=np.float32)

# ** Initial pixel coordinate **
p_x, p_y = 250, 250  # Example pixel coordinate
history1 = []
history2 = []
while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    # Convert color frame to numpy array and grayscale
    color_image = np.asanyarray(color_frame.get_data())
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    # Detect AprilTags
    detections = detector.detect(gray_image)

    for i, detection in enumerate(detections):
        # List to store the last 5 (x, y, z, error) values
        corners = np.array(detection.corners, dtype=np.float32)

        # Compute tag center
        center = np.array(detection.center, dtype=np.int32)
        tag_cx = center[0]
        tag_cy = center[1]

        # Detect tag pose using SolvePnP
        ret, rvec, tvec = cv2.solvePnP(object_points_3d[:4], corners, camera_matrix, dist_coeffs)

        if ret:
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # Calculate real-world depth from the tag center (use median of surrounding points as depth)
            depth_values = []
            for _ in range(20):
                depth = depth_frame.get_distance(tag_cx, tag_cy) - 0.01
                if depth > 0:
                    depth_values.append(depth)

            if depth_values:
                Z_real_tag = np.median(depth_values)  # Median depth value

            # Compute scale factor based on depth
            scale_factor = Z_real_tag / tvec[2]  # Use depth at tag's center
            tvec_real = tvec * scale_factor  # Apply the scaling factor to the translation vector

            imgpts, _ = cv2.projectPoints(object_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = np.int32(imgpts).reshape(-1, 2)

            depth_values = []
            for _ in range(20):
                depth = depth_frame.get_distance(p_x, p_y) - 0.01
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
            
                def model(X, a, b, c, d):
                    # Extract fitted parameters
                    x, y = X

                    return a * x + b * np.exp(c * y) + d
                if i == 0:
                    a, b, c, d = 0.03092494, -0.03941634,  0.56732075,  0.03739654
                elif i == 1:
                    a, b, c, d = -0.01910277, 0.01119914, 1.9034686,  -0.01428886

                h_error = model((point_tag[0][0], point_tag[1][0]), a, b, c, d)
                h_real = point_tag[2][0] - h_error
                
                # Draw the input pixel coordinate on the image
                cv2.circle(color_image, (p_x, p_y), 5, (0, 255, 0), 2)


                # Add the current (x, y, z, error) values to the history list
                if i == 0:
                    history1.append((point_tag[0][0], point_tag[1][0], h_real, h_error, point_tag[2][0]))

                    # Keep only the last 5 values
                    if len(history1) > 20:
                        history1.pop(0)

                    # Calculate the average of the last 5 values
                    avg_x = np.mean([item[0] for item in history1])
                    avg_y = np.mean([item[1] for item in history1])
                    avg_z = np.mean([item[2] for item in history1])
                    avg_error = np.mean([item[3] for item in history1])
                    avg_zz = np.mean([item[4] for item in history1])

                    cv2.putText(color_image, f"X={avg_x:.3f} Y={avg_y:.3f} Z={avg_z:.3f} ZZ={avg_zz:.3f} E={avg_error:.3f}", 
                            (10, (i+1)*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                elif i == 1:
                    history2.append((point_tag[0][0], point_tag[1][0], h_real, h_error, point_tag[2][0]))

                    # Keep only the last 5 values
                    if len(history2) > 20:
                        history2.pop(0)

                    # Calculate the average of the last 5 values
                    avg_x = np.mean([item[0] for item in history2])
                    avg_y = np.mean([item[1] for item in history2])
                    avg_z = np.mean([item[2] for item in history2])
                    avg_error = np.mean([item[3] for item in history2])
                    avg_zz = np.mean([item[4] for item in history2])

                    cv2.putText(color_image, f"X={avg_x:.3f} Y={avg_y:.3f} Z={avg_z:.3f} ZZ={avg_zz:.3f} E={avg_error:.3f}", 
                            (10, (i+1)*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.line(color_image, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)  # X-axis (red)
            cv2.line(color_image, tuple(imgpts[0]), tuple(imgpts[3]), (0, 255, 0), 3)  # Y-axis (green)
            cv2.line(color_image, tuple(imgpts[0]), tuple(imgpts[4]), (255, 0, 0), 3)  # Z-axis (blue)

    # Show image
    cv2.imshow("AprilTag Detection", color_image)

    # **Keyboard Input for Moving p_x, p_y**
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        print(f"x = {point_tag[0][0]}, y = {point_tag[1][0]}, z = {point_tag[2][0]}")

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
