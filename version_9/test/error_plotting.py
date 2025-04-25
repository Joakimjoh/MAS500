import cv2
import numpy as np
import csv
import os
import sys
import apriltag
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.append('/home/student/Documents/MAS500')
from camera import Camera
cv2.startWindowThread()
# Initialize camera
camera = Camera()
points_tag1 = []
points_tag2 = []
points_x_tag1 = []
points_x_tag2 = []
points_y_tag1 = []
points_y_tag2 = []

# Initialize AprilTag detector
detector = apriltag.Detector()

# Ensure the camera frame is valid before processing
if camera.frame is not None and camera.frame.color is not None:
    gray_image = cv2.cvtColor(camera.frame.color, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray_image)

    tags = camera.get_tag_orientation()

    for i, tag in enumerate(tags):
        # Check if the necessary CSV files exist
        if os.path.exists("region.csv"):
            # Read and process CSV files
            for filename in ["region.csv"]:
                unique_points = set()

                with open("region.csv", "r") as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip header
                    for row in reader:
                        if len(row) != 3:
                            continue  # Skip invalid rows
                        
                        x, y, z = map(float, row)
                        
                        # Check if the (x, y) combination is already processed
                        if (x, y) in unique_points:
                            continue  # Skip duplicates

                        # Add the (x, y) pair to the set
                        unique_points.add((x, y))

                        point_pixel = (x, y, z)
                        
                        # Convert to coordinate system
                        point = camera.pixel_to_coordsystem(tag[0], tag[1], point_pixel)
                        
                        # Separate points by tag
                        if i == 0:  # Assuming tag[0] corresponds to tag 1
                            points_tag1.append(point)
                            if y == 0:
                                points_x_tag1.append(point)
                            if x == 0:
                                points_y_tag1.append(point)
                        elif i == 1:  # Assuming tag[0] corresponds to tag 2
                            points_tag2.append(point)
                            if y == 0:
                                points_x_tag2.append(point)
                            if x == 0:
                                points_y_tag2.append(point)

# Check if points are available for plotting
if points_tag1 and points_tag2:
    # Convert lists to NumPy arrays for easier slicing
    points_tag1 = np.array(points_tag1)
    points_tag2 = np.array(points_tag2)

    # Extract X, Y, and Z values for both tags
    X_tag1 = points_tag1[:, 0, 0]
    Y_tag1 = points_tag1[:, 1, 0]
    Z_tag1 = points_tag1[:, 2, 0]

    X_tag2 = points_tag2[:, 0, 0]
    Y_tag2 = points_tag2[:, 1, 0]
    Z_tag2 = points_tag2[:, 2, 0]

    X_tag1_x = points_tag1[:, 0, 0]
    Y_tag1_x = points_tag1[:, 1, 0]
    Z_tag1_x = points_tag1[:, 2, 0]

    X_tag2_x = points_tag2[:, 0, 0]
    Y_tag2_x = points_tag2[:, 1, 0]
    Z_tag2_x = points_tag2[:, 2, 0]

    X_tag1_y = points_tag1[:, 0, 0]
    Y_tag1_y = points_tag1[:, 1, 0]
    Z_tag1_y = points_tag1[:, 2, 0]

    X_tag2_y = points_tag2[:, 0, 0]
    Y_tag2_y = points_tag2[:, 1, 0]
    Z_tag2_y = points_tag2[:, 2, 0]

    # Choose 1000 points evenly spaced across the range of X and Z
    num_points = 1000
    x_min, x_max = np.min(X_tag1_x), np.max(X_tag1_x)
    z_min, z_max = np.min(Z_tag1_x), np.max(Z_tag1_x)

    # Get evenly spaced indices based on the total number of points
    indices_x_1 = np.linspace(0, len(X_tag1_x) - 1, num_points, dtype=int)
    indices_x_2 = np.linspace(0, len(X_tag2_x) - 1, num_points, dtype=int)
    indices_y_1 = np.linspace(0, len(Y_tag1_y) - 1, num_points, dtype=int)
    indices_y_2 = np.linspace(0, len(Y_tag2_y) - 1, num_points, dtype=int)

    # Select the points based on the indices
    X_selected_x_tag_1 = X_tag1_x[indices_x_1]
    Z_selected_x_tag_1 = Z_tag1_x[indices_x_1]
    X_selected_x_tag_2 = X_tag2_x[indices_x_2]
    Z_selected_x_tag_2 = Z_tag2_x[indices_x_2]

    Y_selected_y_tag_1 = Y_tag1_y[indices_y_1]
    Z_selected_y_tag_1 = Z_tag1_y[indices_y_1]
    Y_selected_y_tag_2 = Y_tag2_y[indices_y_2]
    Z_selected_y_tag_2 = Z_tag2_y[indices_y_2]

    # Simple scatter plot for Tag 1 (Z vs X)
    fig, ax = plt.subplots(figsize=(15, 6))  # Increase the figure width to stretch out the plot
    ax.scatter(X_selected_x_tag_1, Z_selected_x_tag_1, c='blue', label="Tag 1", marker='o')  # Scatter plot

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Z vs X for Tag 1")
    ax.legend()

    # Format x-axis to show 3 decimal places
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3f}'))

    # Adjust x-axis limits to make the points more spread out
    ax.set_xlim([x_min - 0.1, x_max + 0.1])  # Add some padding to the x-axis
    ax.set_ylim([z_min - 0.1, z_max + 0.1])  # Add some padding to the y-axis

    # Manually adjust the z-axis to have a smaller range
    z_range = np.max(Z_selected_x_tag_1) - np.min(Z_selected_x_tag_1)
    ax.set_ylim([np.min(Z_selected_x_tag_1) - 0.1 * z_range, np.max(Z_selected_x_tag_1) + 0.1 * z_range])  # Limit the z-axis range

    fig.savefig('x1_plot.png')

    # Simple scatter plot for Tag 1 (Z vs X)
    fig, ax = plt.subplots(figsize=(15, 6))  # Increase the figure width to stretch out the plot
    ax.scatter(X_selected_x_tag_2, Z_selected_x_tag_2, c='blue', label="Tag 1", marker='o')  # Scatter plot

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Z vs X for Tag 2")
    ax.legend()

    # Format x-axis to show 3 decimal places
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3f}'))

    # Adjust x-axis limits to make the points more spread out
    ax.set_xlim([x_min - 0.1, x_max + 0.1])  # Add some padding to the x-axis
    ax.set_ylim([z_min - 0.1, z_max + 0.1])  # Add some padding to the y-axis

    # Manually adjust the z-axis to have a smaller range
    z_range = np.max(Z_selected_x_tag_2) - np.min(Z_selected_x_tag_2)
    ax.set_ylim([np.min(Z_selected_x_tag_2) - 0.1 * z_range, np.max(Z_selected_x_tag_2) + 0.1 * z_range])  # Limit the z-axis range

    fig.savefig('x2_plot.png')

    # Simple scatter plot for Tag 1 (Z vs X)
    fig, ax = plt.subplots(figsize=(15, 6))  # Increase the figure width to stretch out the plot
    ax.scatter(Y_selected_y_tag_1, Z_selected_y_tag_1, c='blue', label="Tag 1", marker='o')  # Scatter plot

    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.set_title("Z vs Y for Tag 1")
    ax.legend()

    # Format x-axis to show 3 decimal places
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3f}'))

    # Adjust x-axis limits to make the points more spread out
    ax.set_xlim([x_min - 0.1, x_max + 0.1])  # Add some padding to the x-axis
    ax.set_ylim([z_min - 0.1, z_max + 0.1])  # Add some padding to the y-axis

    # Manually adjust the z-axis to have a smaller range
    z_range = np.max(Z_selected_y_tag_1) - np.min(Z_selected_y_tag_1)
    ax.set_ylim([np.min(Z_selected_y_tag_1) - 0.1 * z_range, np.max(Z_selected_y_tag_1) + 0.1 * z_range])  # Limit the z-axis range

    fig.savefig('y1_plot.png')

    # Simple scatter plot for Tag 1 (Z vs X)
    fig, ax = plt.subplots(figsize=(15, 6))  # Increase the figure width to stretch out the plot
    ax.scatter(Y_selected_y_tag_2, Z_selected_y_tag_2, c='blue', label="Tag 1", marker='o')  # Scatter plot

    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.set_title("Z vs Y for Tag 2")
    ax.legend()

    # Format x-axis to show 3 decimal places
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3f}'))

    # Adjust x-axis limits to make the points more spread out
    ax.set_xlim([x_min - 0.1, x_max + 0.1])  # Add some padding to the x-axis
    ax.set_ylim([z_min - 0.1, z_max + 0.1])  # Add some padding to the y-axis

    # Manually adjust the z-axis to have a smaller range
    z_range = np.max(Z_selected_y_tag_2) - np.min(Z_selected_y_tag_2)
    ax.set_ylim([np.min(Z_selected_y_tag_2) - 0.1 * z_range, np.max(Z_selected_y_tag_2) + 0.1 * z_range])  # Limit the z-axis range

    fig.savefig('y2_plot.png')

else:
    print("No valid points to plot.")
