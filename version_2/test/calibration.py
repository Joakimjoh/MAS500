import cv2
import numpy as np
import pyrealsense2 as rs
import time

# Chessboard dimensions (change to match your board)
CHESSBOARD_ROWS = 8
CHESSBOARD_COLS = 6
CHESSBOARD_SIZE = 0.008  # Size of each square in meters

# Prepare 3D points of the chessboard (real-world coordinates)
obj_points = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
obj_points[:, :2] = np.indices((CHESSBOARD_ROWS, CHESSBOARD_COLS)).T.reshape(-1, 2)
obj_points *= CHESSBOARD_SIZE  # Scale to real-world size

# Lists to store object points and image points
obj_points_list = []
img_points_list = []

# RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Camera intrinsics
profile = pipeline.get_active_profile()
depth_stream = profile.get_stream(rs.stream.depth)
intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

# Adjust camera settings
device = pipeline.get_active_profile().get_device()
rgb_sensor = None
depth_sensor = device.first_depth_sensor()

for sensor in device.query_sensors():
    if sensor.get_info(rs.camera_info.name) == "RGB Camera":
        rgb_sensor = sensor
        break

if rgb_sensor:
    rgb_sensor.set_option(rs.option.saturation, 50)
    rgb_sensor.set_option(rs.option.sharpness, 100)

depth_sensor.set_option(rs.option.visual_preset, 5)  # "High Density" mode

print("Move the chessboard around the camera for calibration. Press 'q' to stop.")

frame_count = 0

while True:
    time.sleep(1)
    # Get frames from RealSense camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        continue

    # Convert to numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # Convert to grayscale with improved contrast
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_image = clahe.apply(gray_image)

    # Try different chessboard detection flags
    chessboard_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | 
                        cv2.CALIB_CB_FAST_CHECK | 
                        cv2.CALIB_CB_NORMALIZE_IMAGE | 
                        cv2.CALIB_CB_EXHAUSTIVE)

    ret, corners = cv2.findChessboardCorners(gray_image, (CHESSBOARD_COLS, CHESSBOARD_ROWS), flags=chessboard_flags)

    if ret:
        print("test1")
        # Refine corner locations with a larger search window
        corners2 = cv2.cornerSubPix(
            gray_image, corners, (15, 15), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
        )
        
        # Filter out noisy detections
        if len(corners2) == CHESSBOARD_ROWS * CHESSBOARD_COLS:
            obj_points_list.append(obj_points)
            img_points_list.append(corners2)
            frame_count += 1
            print(f"Frame {frame_count}: Chessboard detected!")

            # Draw detected corners
            cv2.drawChessboardCorners(color_image, (CHESSBOARD_COLS, CHESSBOARD_ROWS), corners2, ret)
        else:
            print("⚠️ Detected incorrect number of corners, skipping frame.")

    else:
        print("❌ Chessboard not detected.")

    # Show results
    cv2.imshow("Chessboard Detection", color_image)
    cv2.imshow("Gray Image (Processed)", gray_image)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop streaming
pipeline.stop()
cv2.destroyAllWindows()

# Perform camera calibration if enough frames were captured
if len(obj_points_list) > 10:  # Require at least 10 frames for accurate calibration
    print("Calibrating camera...")

    # Compute camera calibration parameters
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_list, img_points_list, gray_image.shape[::-1], None, None
    )

    print("\n📷 Camera Intrinsics:")
    print(camera_matrix)

    print("\n🔧 Distortion Coefficients:")
    print(dist_coeffs.ravel())

    # Extract last extrinsic parameters
    rvec = rvecs[-1]
    tvec = tvecs[-1]

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Compute Euler angles (tilt)
    pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    yaw = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Convert radians to degrees
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    roll_deg = np.degrees(roll)

    print("\n📐 Camera Tilt (Pitch): {:.2f}°".format(pitch_deg))
    print("🔄 Camera Yaw: {:.2f}°".format(yaw_deg))
    print("↺ Camera Roll: {:.2f}°".format(roll_deg))

else:
    print("⚠️ Not enough frames captured for calibration. Try again.")

