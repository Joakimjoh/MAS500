import numpy as np
import cv2
import pyrealsense2 as rs

# Chessboard settings
CHESSBOARD_SIZE = (3, 3)  # Number of inner corners (rows, cols)
SQUARE_SIZE = 0.03  # Size of a square in meters
MIN_IMAGES = 10  # Minimum number of images for calibration

# Termination criteria for subpixel refinement
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D coordinates of chessboard corners)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Lists to store object points and image points
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("[INFO] Starting RealSense camera...")
pipeline.start(config)

captured_images = 0
while captured_images < MIN_IMAGES:
    print(captured_images)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to enhance contrast
    enhanced_gray = cv2.equalizeHist(gray)

    # Allow a 10% deviation from perfect 90-degree angles
    chessboard_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH  # Adjusts thresholding dynamically
        + cv2.CALIB_CB_FAST_CHECK  # Fast initial check to improve performance
        + cv2.CALIB_CB_FILTER_QUADS  # Allows slightly non-rectangular chessboards
    )

    # Detect chessboard corners
    ret, corners = cv2.findChessboardCorners(enhanced_gray, CHESSBOARD_SIZE, chessboard_flags)

    if ret:
        # Refine corners for better accuracy
        corners2 = cv2.cornerSubPix(enhanced_gray, corners, (11, 11), (-1, -1), CRITERIA)

        # Draw chessboard
        cv2.drawChessboardCorners(color_image, CHESSBOARD_SIZE, corners2, ret)

        cv2.imshow("Calibration", color_image)
        key = cv2.waitKey(1)

        # Wait for user confirmation ('c' to capture)
        if key == ord('c'):
            objpoints.append(objp)
            imgpoints.append(corners2)
            captured_images += 1
            print(f"[INFO] Captured {captured_images}/{MIN_IMAGES} images.")

    else:
        cv2.imshow("Calibration", color_image)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Quit if 'q' is pressed
        break

print("[INFO] Capturing complete. Calibrating camera...")

# Calibrate the camera
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n=== Calibration Results ===")
print(f"Reprojection Error: {ret}")
print(f"Camera Matrix:\n{camera_matrix}")
print(f"Distortion Coefficients:\n{distortion_coeffs}")

# Save calibration data
np.savez("camera_calibration_data.npz", camera_matrix=camera_matrix, distortion_coeffs=distortion_coeffs)
print("[INFO] Calibration data saved.")

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
