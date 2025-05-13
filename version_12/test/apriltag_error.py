import cv2
import numpy as np
import sys
import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import apriltag
import time

sys.path.append('/home/student/Documents/MAS500')

from camera import Camera
from tag import Tag

# Initialize camera and tag
camera = Camera()
tag = Tag(camera, "left")

# Get frame dimensions
height, width, _ = camera.frame.color.shape

# Compute square ROI (50% of smaller frame dimension, centered)
roi_size = min(width, height) // 2
x_start = (width - roi_size) // 2
y_start = (height - roi_size) // 2

# Generate pixel grid within ROI
x_coords, y_coords = np.meshgrid(
    np.arange(x_start, x_start + roi_size),
    np.arange(y_start, y_start + roi_size)
)

# Flatten pixel coordinates
pixel_coords = np.stack((x_coords.flatten(), y_coords.flatten()), axis=-1)

# Sample depth and convert to 3D
points = []
depth = camera.frame.depth
for x, y in pixel_coords:
    z = depth.get_distance(x, y)  # Scale if necessary (RealSense gives mm by default)
    if z == 0:
        continue
    point = camera.pixel_to_coordsystem(tag.orientation, (x, y, z))
    if point is not None:
        points.append(point)

# Convert to NumPy array for easier handling
points = np.array(points)

# Check if we have valid points to plot
if points.shape[0] == 0:
    print("No valid 3D points found.")
    sys.exit()

# Extract X, Y, Z for plotting
X = points[:, 0]
Y = points[:, 1]
Z = points[:, 2]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, s=1)

ax.set_title("3D Plot of ROI Points")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

plt.tight_layout()
plt.show()
