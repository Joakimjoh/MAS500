import cv2
import numpy as np
from scipy.optimize import curve_fit
import csv
import os
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import SmoothBivariateSpline
import apriltag
sys.path.append('/home/student/Documents/MAS500')

from camera import Camera

def get_error_region(camera, detections):
    color_frame = camera.frame.color
    gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

    # Image dimensions
    img_height, img_width, _ = color_frame.shape

    # Define the size of the square region (50% of the frame)
    square_size = int(min(img_height, img_width) * 0.5)
    square_half = square_size // 2

    # Create a square mask
    square_mask = np.zeros_like(gray, dtype=np.uint8)

    # Exclude points that are on the tags
    tag_mask = np.zeros_like(gray, dtype=np.uint8)
    for detection in detections:
        tag_center_x, tag_center_y = int(detection.center[0]), int(detection.center[1])

        # Calculate the top-left corner of the square (centered around the tag)
        top_left_x = max(tag_center_x - square_half, 0)
        top_left_y = max(tag_center_y - square_half, 0)

        # Calculate the bottom-right corner of the square
        bottom_right_x = min(tag_center_x + square_half, img_width)
        bottom_right_y = min(tag_center_y + square_half, img_height)

        # Update the square mask
        square_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255

        # Scale tag corners outward by 50%
        corners = np.array(detection.corners, dtype=np.float32)
        tag_center = np.mean(corners, axis=0)  # Calculate the center of the tag
        scaled_corners = tag_center + 1.50 * (corners - tag_center)  # Scale corners outward by 15%
        scaled_corners = scaled_corners.astype(np.int32)

        # Add scaled tag corners to the tag mask
        cv2.fillPoly(tag_mask, [scaled_corners], 255)

    # Combine the square mask and the inverted tag mask
    combined_mask = cv2.bitwise_and(square_mask, cv2.bitwise_not(tag_mask))

    # Extract depth values for each region
    region_data = []

    for y in range(img_height):
        for x in range(img_width):
            if combined_mask[y, x] > 0:  # Only process points outside the tag regions
                depth_values = []
                for _ in range(100):
                    depth_value = camera.frame.depth.get_distance(x, y) - 0.01
                    if depth_value > 0:
                        depth_values.append(depth_value)
                
                # Calculate the median after collecting all samples
                depth_value = np.median(depth_values) if depth_values else 0
                region_data.append((x, y, depth_value))

    # Save depth data to CSV files
    with open("region.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "depth"])
        writer.writerows(region_data)

# CUSTOM EXPONENTIAL FUNCTION
def custom_exp_error(X, A, B, C):
    x, y = X
    return A * np.exp(B * x + C * y)

# CUSTOM EXPONENTIAL FUNCTION
def custom_exp(x_values, y_values, z_values):
    """
    Fits an exponential model to the given x, y, z data and returns the parameters.
    """
    
    x_values = np.array(x_values).flatten()
    y_values = np.array(y_values).flatten()
    z_values = np.array(z_values).flatten()

    params, _ = curve_fit(custom_exp_error, (x_values, y_values), z_values, p0=(1, 1, 0.1))
    return params

# CUSTOM FUNCTION COMBINING LINEAR AND EXPONENTIAL
def custom_lin_exp_error(X, a, b, c, d):
    # Extract fitted parameters
    x, y = X

    return a * x + b * np.exp(c * y) + d

# CUSTOM FUNCTION COMBINING LINEAR AND EXPONENTIAL
def custom_lin_exp(x_values, y_values, z_values):
    # Curve fitting
    x_values = np.array(x_values).flatten()
    y_values = np.array(y_values).flatten()
    z_values = np.array(z_values).flatten()
    params, _ = curve_fit(custom_lin_exp_error, (x_values, y_values), z_values, p0=(1, 1, 1, 0.1), maxfev=10000)

    return params  # Returns (a, b, c, d)

# LINEAR REGRESSION FITTING
def linear_reg(x_values, y_values, z_values):
    # Combine x and y into a single array of shape (n_samples, n_features)
    x_values = np.array(x_values).flatten()
    y_values = np.array(y_values).flatten()
    z_values = np.array(z_values).flatten()
    X = np.column_stack((x_values, y_values))

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, z_values)

    return model

# LINEAR REGRESSION ERROR
def linear_reg_error(X, model):
    x, y = X
    input_array = np.array([x, y]).flatten().reshape(1, -1)  # Flatten and reshape to 2D
    
    # Ensure no dimensionality error
    z_pred = model.predict(input_array)  # Correct way to use predict

    return z_pred

# LINEAR SPLINE
def linear_spline(x_values, y_values, z_values):
    """
    This function takes in x, y, and z values and returns a 2D linear spline function.
    The spline interpolates z relative to x and y values.
    """
    x_values = np.array(x_values).flatten()
    y_values = np.array(y_values).flatten()
    z_values = np.array(z_values).flatten()
    # Stack x and y into a single array of coordinates
    coordinates = np.column_stack((x_values, y_values))

    # Create the 2D linear spline model
    spline_model = LinearNDInterpolator(coordinates, z_values)

    return spline_model

# LINEAR SPLINE
def linear_spline_error(X, spline_model):
    """
    Given the spline model and a new (x, y) point, predict the corresponding z value.
    """
    x, y = X
    # Use the model to predict the z value at the new (x, y) point
    z_pred = spline_model(x, y)

    return z_pred

# B-Spline Model
def b_spline(x_values, y_values, z_values, degree=3, smooth=1.0):
    """
    This function takes in x, y, and z values and returns a B-spline model.
    The B-spline interpolates z relative to x and y values.
    
    :param x_values: Array of x values (independent variable)
    :param y_values: Array of y values (independent variable)
    :param z_values: Array of z values (dependent variable)
    :param degree: Degree of the B-spline (default is 3, cubic B-spline)
    :param smooth: Smoothing factor (default is 0, no smoothing)
    
    :return: B-spline model function
    """
    x_values = np.array(x_values).flatten()
    y_values = np.array(y_values).flatten()
    z_values = np.array(z_values).flatten()
    
    # Create the B-spline model
    spline_model = SmoothBivariateSpline(x_values, y_values, z_values, kx=degree, ky=degree, s=smooth)
    
    return spline_model

# B-Spline Error
def b_spline_error(X, spline_model):
    """
    Given the spline model and a new (x, y) point, predict the corresponding z value.
    
    :param X: A tuple of (x, y) coordinates for which to predict the z value
    :param spline_model: The B-spline model
    :return: Predicted z value
    """
    x, y = X
    # Use the model to predict the z value at the new (x, y) point
    z_pred = spline_model(x, y)
    
    return z_pred

# Function to read data from the CSV files
def read_csv_data():
    region_data = []

    # Read region.csv
    with open("region.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            x, y, depth = int(row[0]), int(row[1]), float(row[2])
            region_data.append((x, y, depth))

    return region_data

# Main function to read data, calculate error, and return the errors
def calculate_errors(camera, tags):
    # Read CSV data
    region_data = read_csv_data()
    l = []

    # Lists to store point_tag data for both regions
    for i, tag in enumerate(tags):
        x_values = []
        y_values = []
        z_values = []
        rvec, tvec = tag
        # Process each region data (left and right)

        # Process the pixels in the region
        for x, y, depth in region_data:
            if depth > 0:
                point_tag = camera.pixel_to_coordsystem(rvec, tvec, (x, y, depth))
                x_values.append(point_tag[0])
                y_values.append(point_tag[1])
                z_values.append(point_tag[2])

        l.append(linear_reg(x_values, y_values, z_values))
    
    return l

cv2.startWindowThread()

camera = Camera()
points = []

detector = apriltag.Detector()
gray_image = cv2.cvtColor(camera.frame.color, cv2.COLOR_BGR2GRAY)
detections = detector.detect(gray_image)

# Ensure detections are valid and not empty
if not detections or not isinstance(detections[0], apriltag.Detection):
    raise ValueError("No valid AprilTag detections found. Please ensure the camera is capturing tags.")

if not os.path.exists("region.csv"):
    get_error_region(camera, detections)

tags = camera.get_tag_orientation()

error_list = []  # List to store the error for each iteration
z_list = []  # List to store the z values
z_real_list = []  # List to store the real z values (adjusted by error)

params = calculate_errors(camera, tags)

rvec, tvec = tags[1]  # Extract rvec and tvec for each tag

region_data = []

for i, param in enumerate(params):
    if i == 1:
        # Loop over all the x, y points in region_x and region_y
        region_x = []
        region_y = []

        # Read region_left.csv
        with open("region.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                x, y, depth = int(row[0]), int(row[1]), float(row[2])
                point = camera.pixel_to_coordsystem(rvec, tvec, (x, y, depth))  # Convert pixel to 3D coordinates
                
                region_data.append((x, y, depth))
                
                # Parameters a, b, c, d from the 'param'
                #a, b, c = param

                # Calculate the error using the model function
                #error = linear_reg_error((point[0], point[1]), a, b, c)

                error = linear_reg_error((point[0], point[1]), param)

                # Append values to the respective lists
                error_list.append(error)
                z_list.append(point[2])  # z value from the point
                z_real_list.append(point[2] - error)  # Real z value adjusted by error

# Convert x-axis and y-axis pixel coordinates to world coordinates
world_x = []
world_z_x = []  # Z values for Z vs X
world_y = []
world_z_y = []  # Z values for Z vs Y

# Calculate the center of the square region
center_x = region_data[0][0] + (region_data[-1][0] - region_data[0][0]) // 2
center_y = region_data[0][1] + (region_data[-1][1] - region_data[0][1]) // 2

for x, y, depth in region_data:
    if depth > 0:
        if y == center_y:  # Pixels along the x-axis
            point_x = camera.pixel_to_coordsystem(rvec, tvec, (x, y, depth))
            world_x.append(point_x[0])  # World X coordinate
            
            error = linear_reg_error((point_x[0], point_x[1]), params[1])

            world_z_x.append(point_x[2] - error)  # Real z value adjusted by error

        if x == center_x:  # Pixels along the y-axis
            point_y = camera.pixel_to_coordsystem(rvec, tvec, (x, y, depth))
            world_y.append(point_y[1])  # World Y coordinate

            error = linear_reg_error((point_y[0], point_y[1]), params[1])

            world_z_y.append(point_y[2])  # Real z value adjusted by error

# Plot Z vs X in world coordinates
plt.figure(figsize=(12, 8))
plt.plot(world_x, world_z_x, color='b', linewidth=2, label='Z vs X (World Coordinates)')
plt.xlabel('X (World)', fontsize=18)
plt.ylabel('Z (World)', fontsize=18)
plt.title('Depth (Z) vs X in World Coordinates (X-Axis)', fontsize=20)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('z_vs_x_world_plot.png', dpi=300)
plt.close()

# Plot Z vs Y in world coordinates
plt.figure(figsize=(12, 8))
plt.plot(world_y, world_z_y, color='r', linewidth=2, label='Z vs Y (World Coordinates)')
plt.xlabel('Y (World)', fontsize=18)
plt.ylabel('Z (World)', fontsize=18)
plt.title('Depth (Z) vs Y in World Coordinates (Y-Axis)', fontsize=20)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('z_vs_y_world_plot.png', dpi=300)
plt.close()

# Plot z_real values (adjusted by error)
plt.figure(figsize=(12, 8))
plt.plot(np.ravel(z_real_list), label='Height Above Table (m)', color='g', linewidth=2)
plt.xlabel('Pixel Points', fontsize=18)
plt.ylabel('Height', fontsize=18)
plt.title('Adjusted Height Above Table for Different Pixel Points', fontsize=20)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('z_real_plot.png', dpi=300)
plt.close()