import cv2
import numpy as np
import apriltag
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get the depth stream profile and intrinsics
profile = pipeline.get_active_profile()
depth_stream = profile.get_stream(rs.stream.depth)
intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

# Camera intrinsics (from your calibration data)
camera_matrix = np.array([
    [745.28481482, 0, 99.85896602],
    [0, 224.6334517, 319.79026665],
    [0, 0, 1]
])
dist_coeffs = np.array([0.7428751, -0.42698043, -0.12280206, -0.26499492, 0.10712332])

# Example rotation matrix and translation vector from your calibration process
rotation_matrix = np.array([
    [0.2418689, -0.70365253, -0.66811118],
    [-0.42597434, 0.5416529, -0.72467786],
    [0.87180577, 0.45987526, -0.16872893]
])
translation_vector = np.array([0.65010343, -0.50279875, 0.95141586])  # Adjust if needed

# Camera height (49 cm from table)
CAMERA_HEIGHT = 0.49  # meters

# Configure AprilTag detector for 'tag36h11' family
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

while True:
    # Wait for the next frame
    frames = pipeline.wait_for_frames()
    aligned_frames = pipeline.wait_for_frames()

    # Get color and depth frames
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    # Convert RealSense frame to numpy array
    color_image = np.asanyarray(color_frame.get_data())
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    detections = detector.detect(gray_image)

    for detection in detections:
        center = detection.center
        corners = detection.corners

        # Get depth at tag center
        depth_value = depth_frame.get_distance(int(center[0]), int(center[1]))

        # Convert pixel coordinates + depth to real-world coordinates (in the camera's frame)
        world_coords = rs.rs2_deproject_pixel_to_point(intrinsics, [center[0], center[1]], depth_value)
        x, y, z = world_coords  # Real-world coordinates in camera frame (meters)

        # Apply rotation and translation to convert from camera frame to world frame
        world_coords_rotated = rotation_matrix.dot(np.array([x, y, z])) + translation_vector

        # Extract transformed world coordinates
        world_x, world_y, world_z = world_coords_rotated

        # Calculate elevation relative to the table (adjust based on your setup)
        elevation = CAMERA_HEIGHT - world_y

        # Draw the bounding box of the tag
        for i in range(4):
            cv2.line(color_image, tuple(map(int, corners[i])), tuple(map(int, corners[(i + 1) % 4])), (0, 255, 0), 2)

        # Display tag ID and distance
        tag_id = detection.tag_id
        print(f"Tag ID: {tag_id} | X: {world_x:.3f}m | Y: {world_y:.3f}m | Z: {world_z:.3f}m | Elevation: {elevation:.3f}m")  

    # Show the result
    cv2.imshow("AprilTag Detection - tag36h11", color_image)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
