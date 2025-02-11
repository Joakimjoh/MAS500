import cv2
import numpy as np
import pyrealsense2 as rs
import os
from datetime import datetime

# Chessboard dimensions (adjust to match your actual chessboard)
chessboard_size = (5, 6)  # 3x4 squares
square_size = 0.044  # Square size in meters (adjust as needed)

# Prepare object points (3D points in real-world space)
object_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
object_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
object_points *= square_size  # Scale points

# Arrays to store object points and image points
obj_points = []  # 3D points in real world space
img_points = []  # 2D points in image plane

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color and depth streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream

# Start streaming
pipeline.start(config)

# Wait for the camera to stabilize
print("Starting camera...")
for _ in range(10):
    pipeline.wait_for_frames()

frame_count = 0

while True:
    # Capture frames
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    # Convert to numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # Convert to grayscale
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        obj_points.append(object_points)
        img_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(color_image, chessboard_size, corners, ret)
        frame_count += 1
        print(f"Collected {frame_count} frames")

    # Show the image
    cv2.imshow("Chessboard Calibration", color_image)

    # Press 'q' to quit or collect 20 frames
    if cv2.waitKey(1) & 0xFF == ord('q') or frame_count >= 20:
        break

# Stop the camera and close windows
pipeline.stop()
cv2.destroyAllWindows()

# Perform camera calibration
print("Calibrating camera...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Save calibration data
calibration_data = {
    'camera_matrix': mtx,
    'distortion_coefficients': dist,
    'rotation_vectors': rvecs,
    'translation_vectors': tvecs
}

calibration_file = f"calibration_data.npz"
np.savez(calibration_file, **calibration_data)
print(f"Calibration data saved to {calibration_file}")

# Display results
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)
