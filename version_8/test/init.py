import cv2
import numpy as np
from scipy.optimize import curve_fit
import csv
import os
import matplotlib.pyplot as plt
import sys
import apriltag
sys.path.append('/home/student/Documents/MAS500')

from camera import Camera

def get_error_region(camera, detections, buffer_margin = 0.1):
    color_frame = camera.frame.color
    depth_frame = camera.frame.depth
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_scale = camera.depth_scale
    gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

    if len(detections) >= 2:
        # Get first two detected tags
        tag1, tag2 = detections[:2]

        # Use detection.center for accurate tag centers
        center_1 = np.array(tag1.center, dtype=np.int32)
        center_2 = np.array(tag2.center, dtype=np.int32)

        # Image dimensions
        img_height, img_width, _ = color_frame.shape
        middle_x = img_width // 2  # Vertical middle of the frame

        # Buffer area to avoid close proximity to edges
        min_x = int(img_width * buffer_margin)
        max_x = int(img_width * (1 - buffer_margin))
        min_y = int(img_height * buffer_margin)
        max_y = int(img_height * (1 - buffer_margin))

        # Define an extension depth (~2/3 down)
        extension_depth = int(img_height * 0.67)

        # Compute extended bottom points **exactly from the tag center**
        extended_center_1 = (center_1[0], extension_depth)
        extended_center_2 = (center_2[0], extension_depth)

        # Define a **straight top line** between the two AprilTag centers
        center_between = ((center_1 + center_2) // 2).astype(int)

        # Define the left and right rectangular regions, keeping them within the safe margin
        contour_left = np.array([
            (max(center_1[0], min_x), min(max(center_1[1], min_y), max_y)), 
            (max(center_between[0], min_x), min(max(center_between[1], min_y), max_y)),  
            (max(min_x, middle_x), extension_depth), 
            extended_center_1  
        ])
        
        contour_right = np.array([
            (min(center_between[0], max_x), min(max(center_between[1], min_y), max_y)), 
            (min(center_2[0], max_x), min(max(center_2[1], min_y), max_y)), 
            extended_center_2, (middle_x, extension_depth)  
        ])

        # Create masks for the regions
        mask_left = np.zeros_like(gray, dtype=np.uint8)
        mask_right = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillPoly(mask_left, [contour_left], 255)
        cv2.fillPoly(mask_right, [contour_right], 255)

        # Extract depth values for each region
        region_left_data = []
        region_right_data = []

        for y in range(img_height):
            for x in range(img_width):
                depth_value = depth_image[y, x] * depth_scale  # Convert to meters
                if mask_left[y, x] > 0:
                    region_left_data.append((x, y, depth_value))
                elif mask_right[y, x] > 0:
                    region_right_data.append((x, y, depth_value))

        # Save depth data to CSV files
        with open("region_left.csv", "w", newline="") as f_left:
            writer = csv.writer(f_left)
            writer.writerow(["x", "y", "depth"])
            writer.writerows(region_left_data)

        with open("region_right.csv", "w", newline="") as f_right:
            writer = csv.writer(f_right)
            writer.writerow(["x", "y", "depth"])
            writer.writerows(region_right_data)

# Function to read data from the CSV files
def read_csv_data():
    region_left_data = []
    region_right_data = []

    # Read region_left.csv
    with open("region_left.csv", "r") as f_left:
        reader = csv.reader(f_left)
        next(reader)  # Skip the header row
        for row in reader:
            x, y, depth = int(row[0]), int(row[1]), float(row[2])
            region_left_data.append((x, y, depth))

    # Read region_right.csv
    with open("region_right.csv", "r") as f_right:
        reader = csv.reader(f_right)
        next(reader)  # Skip the header row
        for row in reader:
            x, y, depth = int(row[0]), int(row[1]), float(row[2])
            region_right_data.append((x, y, depth))

    return region_left_data, region_right_data

def exp_model(X, A, B, C):
    x, y = X
    return A * np.exp(B * x + C * y)

# Function to fit an exponential model and return the parameters
def get_error_equation(x_values, y_values, z_values):
    """
    Fits an exponential model to the given x, y, z data and returns the parameters.
    """
    
    x_values = np.array(x_values).flatten()
    y_values = np.array(y_values).flatten()
    z_values = np.array(z_values).flatten()

    params, _ = curve_fit(exp_model, (x_values, y_values), z_values, p0=(1, 1, 0.1))
    return params

def model(X, a, b, c, d):
    # Extract fitted parameters
    x, y = X

    return a * x + b * np.exp(c * y) + d

def fit_parameters(x_values, y_values, z_values):
    # Curve fitting
    x_values = np.array(x_values).flatten()
    y_values = np.array(y_values).flatten()
    z_values = np.array(z_values).flatten()
    params, _ = curve_fit(model, (x_values, y_values), z_values, p0=(1, 1, 1, 0.1), maxfev=10000)

    return params  # Returns (a, b, c, d)

# Main function to read data, calculate error, and return the errors
def calculate_errors(camera, tags):
    # Read CSV data
    region_left_data, region_right_data = read_csv_data()
    l = []

    # Lists to store point_tag data for both regions
    for i, tag in enumerate(tags):
        x_values = []
        y_values = []
        z_values = []
        rvec, tvec = tag
        # Process each region data (left and right)
        if i == 0:
            region_data = region_left_data
        elif i == 1:
            region_data = region_right_data

        # Process the pixels in the region
        for x, y, depth in region_data:
            if depth > 0:
                point_tag = camera.pixel_to_coordsystem(rvec, tvec, (x, y))
                x_values.append(point_tag[0])
                y_values.append(point_tag[1])
                z_values.append(point_tag[2])

        l.append(fit_parameters(x_values, y_values, z_values))
    
    return l

cv2.startWindowThread()

camera = Camera()
points = []
p_x, p_y = 250, 250


detector = apriltag.Detector()
gray_image = cv2.cvtColor(camera.frame.color, cv2.COLOR_BGR2GRAY)
detections = detector.detect(gray_image)

if not (os.path.exists("region_left.csv") and os.path.exists("region_right.csv")):
    get_error_region(camera, detections)

tags = camera.get_tag_orientation()

Al = []
Bl = []
Cl = []
Dl = []
error_list = []  # List to store the error for each iteration

# Loop to collect 50 sets of parameters and errors
for j in range(10):
    print(j)
    params = calculate_errors(camera, tags)
    param = params[1]
    rvec, tvec = tags[1]  # Extract rvec and tvec for each tag
    point = camera.pixel_to_coordsystem(rvec, tvec, (p_x, p_y))  # Convert pixel to 3D coordinates
    
    # Parameters a, b, c, d from the 'param'
    a, b, c, d = param

    Al.append(a)
    Bl.append(b)
    Cl.append(c)
    Dl.append(d)
    # Calculate the error using the model function
    error = model((point[0], point[1]), a, b, c, d)
    print(error)
    print(param)
    error_list.append(error)  # Add the error to the error list

# Plot the parameters for each set of parameters across all iterations (assuming four params)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), Al, label=f'Param A')

plt.xlabel('Iterations')
plt.ylabel('Parameter A')
plt.title('Parameter Values Over 10 Iterations')
plt.legend()

# Save the plot as an image
plt.savefig('params_a_plot.png')  # Save as PNG
plt.close()  # Close the figure to avoid overlap with the next one

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), Bl, label=f'Param B')
plt.xlabel('Iterations')
plt.ylabel('Parameter B')
plt.title('Parameter Values Over 10 Iterations')
plt.legend()

# Save the plot as an image
plt.savefig('params_b_plot.png')  # Save as PNG
plt.close()  # Close the figure to avoid overlap with the next one

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), Cl, label=f'Param C')
plt.xlabel('Iterations')
plt.ylabel('Parameter C')
plt.title('Parameter Values Over 10 Iterations')
plt.legend()

# Save the plot as an image
plt.savefig('params_c_plot.png')  # Save as PNG
plt.close()  # Close the figure to avoid overlap with the next one

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), Dl, label=f'Param D')
plt.xlabel('Iterations')
plt.ylabel('Parameter D')
plt.title('Parameter Values Over 10 Iterations')
plt.legend()

# Save the plot as an image
plt.savefig('params_d_plot.png')  # Save as PNG
plt.close()  # Close the figure to avoid overlap with the next one

# Plot the error values for each error at each iteration
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), error_list, label='Error')

plt.xlabel('Iterations')
plt.ylabel('Error Value')
plt.title('Error Over 10 Iterations')
plt.legend()

# Save the error plot as an image
plt.savefig('error_plot.png')  # Save as PNG
plt.close()  # Close the figure to avoid overlap with the next one
