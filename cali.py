import cv2
import numpy as np
import glob
import pickle

# Checkerboard dimensions (number of inner corners per row and column)
CHECKERBOARD = (7, 6)  # Adjust this if you use a different checkerboard pattern
SQUARE_SIZE = 1.0      # Real-world size of a square in your checkerboard (e.g., in centimeters)

# Termination criteria for corner subpixel accuracy
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in the real world)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points
obj_points = []  # 3D points in real world
img_points = []  # 2D points in image plane

# Load all images from a folder
images = glob.glob('test.jpg')  # Change path to your folder

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the corners of the checkerboard
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        obj_points.append(objp)
        # Refine corner locations for more accurate calibration
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners_refined)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
        cv2.imshow('Checkerboard Detection', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

# Print the camera matrix
print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)

# Save the calibration data
with open('calibration_data.pkl', 'wb') as f:
    pickle.dump((camera_matrix, dist_coeffs), f)

print("\nCalibration complete! Data saved to 'calibration_data.pkl'.")
