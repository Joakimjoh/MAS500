"""Third-Party Libraries"""
import cv2
import numpy as np
import csv

"""Internal Modules"""
from camera import Camera
from tag import Tag

# Start OpenCV window thread
cv2.startWindowThread()

# Initialize camera
camera = Camera()

# Wait for the first valid frame
while camera.frame is None:
    pass

# Detect right tag
tag = Tag(camera, "left")

# Get frame size
frame_h, frame_w = camera.frame.color.shape[:2]

# Define 50% central region
region_w, region_h = frame_w // 2, frame_h // 2
x1, y1 = frame_w // 4, frame_h // 4
x2, y2 = x1 + region_w, y1 + region_h

# Draw the region box (optional)
camera.frame.box["Region"] = (x1, y1, x2, y2)

# List to collect data: (px, py, x, y, z)
points_data = []

# Preload the depth frame as a NumPy array
depth_image = np.asanyarray(camera.frame.depth.get_data()).copy()

# Sample points in the region and convert to world coordinates
for py in range(y1, y2):
    for px in range(x1, x2):
        depth = camera.get_depth((px, py), depth_image=depth_image)
        depth_scale = camera.profile.get_device().first_depth_sensor().get_depth_scale()
        depth_in_meters = depth * depth_scale
        result = camera.pixel_to_coordsystem(tag, (px, py, depth_in_meters), adjust_error=False)
        if result is not None:
            X, Y, Z = result
            points_data.append((px, py, X, Y, Z))

# Save to CSV
csv_filename = "depth_data_pupil_apriltag.csv"
#csv_filename = "depth_data_apriltag.csv"
#csv_filename = "depth_data_solvepnp.csv"
with open(csv_filename, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["px", "py", "x (m)", "y (m)", "z (m)"])
    writer.writerows(points_data)

print(f"Saved {len(points_data)} points to {csv_filename}")
