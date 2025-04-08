import cv2
import numpy as np
import apriltag
import pyrealsense2 as rs
import sys
import matplotlib.pyplot as plt

sys.path.append('/home/student/Documents/MAS500')
from camera import Camera

cv2.startWindowThread()
camera = Camera()

# Lists to store averaged points
x_points = []
y_points = []

tags = camera.get_tag_orientation()
print(tags)

# Loop through detected AprilTags
for i, tag in enumerate(tags):
    if i == 0:
        rvec, tvec = tag
        center = (326, 216)  # Example center; modify as needed

        # Length of scan lines
        x_length = 200
        y_length = 200
        
        # Collect points along the X-axis
        for x in range(center[0] - x_length, center[0] + x_length + 1):
            x_list = []  # Temporary list for 50 samples
            for _ in range(50):
                tag_point = camera.pixel_to_coordsystem(rvec, tvec, (x, center[1]))
                x_list.append(tag_point)
            
            # Compute mean (X, Y, Z) for this X pixel
            mean_x = np.mean([p[0] for p in x_list])
            mean_y = np.mean([p[1] for p in x_list])
            mean_z = np.mean([p[2] for p in x_list])
            
            x_points.append((mean_x, mean_y, mean_z))  # Store averaged point
        
        # Collect points along the Y-axis
        for y in range(center[1] - y_length, center[1] + y_length + 1):
            y_list = []  # Temporary list for 50 samples
            for _ in range(50):
                tag_point = camera.pixel_to_coordsystem(rvec, tvec, (center[0], y))
                y_list.append(tag_point)
            
            # Compute mean (X, Y, Z) for this Y pixel
            mean_x = np.mean([p[0] for p in y_list])
            mean_y = np.mean([p[1] for p in y_list])
            mean_z = np.mean([p[2] for p in y_list])
            
            y_points.append((mean_x, mean_y, mean_z))  # Store averaged point

# Extract values for plotting
x_vals = [p[0] for p in x_points]
x_vals.sort()  # Sorts in place (modifies x_vals)
z_vals_x = [p[2] for p in x_points]

y_vals = [p[1] for p in y_points]
z_vals_y = [p[2] for p in y_points]

# Plot Z vs X
plt.figure(figsize=(10, 6))
plt.plot(x_vals, z_vals_x, label="Z vs X", color="b")
plt.xlabel("X Coordinate")
plt.ylabel("Z Coordinate")
plt.title("Z vs X Plot")
plt.grid(True)
plt.legend()
plt.savefig('z_vs_x_plot.png')
plt.close()

# Plot Z vs Y
plt.figure(figsize=(10, 6))
plt.plot(y_vals, z_vals_y, label="Z vs Y", color="r")
plt.xlabel("Y Coordinate")
plt.ylabel("Z Coordinate")
plt.title("Z vs Y Plot")
plt.grid(True)
plt.legend()
plt.savefig('z_vs_y_plot.png')
plt.close()
