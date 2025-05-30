import cv2
import numpy as np
import glob

# Define the size of the checkerboard (number of inner corners)
CHECKERBOARD_SIZE = (6, 9)

# Define the world coordinates of the checkerboard corners
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)

# Lists to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Get the paths of all the calibration images
images = glob.glob('/home/student/Documents/MAS500/test/calibration_images/*.png')

for filename in images:
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # Draw and display the corners (optional)
        cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the calibration parameters
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# Save the calibration data (optional)
np.savez('calibration_data1.npz', camera_matrix=mtx, dist_coeff=dist)