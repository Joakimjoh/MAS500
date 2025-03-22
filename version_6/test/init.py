import cv2
import numpy as np
from scipy.optimize import curve_fit
import csv
from pynput import keyboard
import sys
sys.path.append('/home/student/Documents/MAS500')

from camera import Camera

def get_error_region(camera, detections, buffer_margin = 0.1):
    color_frame = camera.frame.color
    depth_frame = camera.frame.depth
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
                depth_value = depth_frame[y, x] * depth_scale  # Convert to meters
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

# Function to calculate error using fitted parameters
def get_error(params, x, y):
    # Extract fitted parameters
    A_fit, B_fit, C_fit = params
    return A_fit * np.exp(B_fit * x + C_fit * y)

# Function to fit an exponential model and return the parameters
def get_error_equation(x_values, y_values, z_values):
    """
    Fits an exponential model to the given x, y, z data and returns the parameters.
    """
    def exp_model(X, A, B, C):
        x, y = X
        return A * np.exp(B * x + C * y)

    params, _ = curve_fit(exp_model, (x_values, y_values), z_values, p0=(0.01, 1, 1))
    return params

def get_world_cord(color_intrinsics, tag, x, y, depth):
    # Camera intrinsic parameters
    fx, fy = color_intrinsics.fx, color_intrinsics.fy
    cx, cy = color_intrinsics.ppx, color_intrinsics.ppy

    # Convert pixel coordinates to camera coordinates
    X_camera = (x - cx) * depth / fx
    Y_camera = (y - cy) * depth / fy
    Z_camera = depth

    # Create a 3D point in camera coordinates
    point_camera = np.array([[X_camera], [Y_camera], [Z_camera]])

    # Transform camera coordinates to AprilTag coordinates
    point_tag = np.dot(tag[1].T, (point_camera - tag[2]))
    
    return point_tag

# Main function to read data, calculate error, and return the errors
def calculate_errors(color_intrinsics, tags, points):
    # Read CSV data
    region_left_data, region_right_data = read_csv_data()

    # Lists to store point_tag data for both regions
    pos_left = []  # List for left region (x, y, z)
    pos_right = []  # List for right region (x, y, z)
    for i, tag in enumerate(tags):     
        # Process each region data (left and right)
        if i == 0:
            region_data = region_left_data
        elif i == 1:
            region_data = region_right_data
        
        x_values = []
        y_values = []
        z_values = []

        # Process the pixels in the region
        for x, y, depth in region_data:
            if depth > 0:
                point_tag = get_world_cord(color_intrinsics, tag, x, y, depth)

                # Store x, y, and depth (z) values for each region
                if i == 0:  # Left region
                    pos_left.append((point_tag[0][0], point_tag[1][0], point_tag[2][0]))
                elif i == 1:  # Right region
                    pos_right.append((point_tag[0][0], point_tag[1][0], point_tag[2][0]))

    # Combine the left and right positions into a single list: [(x, y, z), (x, y, z), ...]
    pos = [pos_left, pos_right]
    params = []

    # Fit the error model to the data
    for p in pos:
        x_values, y_values, z_values = zip(*p)  # Unzip to separate x, y, z values
        params.append(get_error_equation(x_values, y_values, z_values))

    # Now select a random point and calculate the error
    for i, param in enumerate(params):
        x, y, z = points[i]
        # Calculate error based on the model parameters and the random coordinates
        error = get_error(param, x, y)

        # Adjust z-value with the error
        z_adjusted = z - error
        print(f"Original z: {z}, Error: {error}, Adjusted z: {z_adjusted}")

# Allow streaming in thread
cv2.startWindowThread()

# Initialize camera
camera = Camera()
points = []
# Shared variables for p_x, p_y
p_x, p_y = 250, 250

# Function to handle keyboard input without blocking
def on_press(key):
    global p_x, p_y
    try:
        if key == keyboard.Key.esc:  # Escape key to quit
            return False  # Stops the listener (will stop the loop)
        elif key == keyboard.Key.left:  # Left arrow
            p_x = max(p_x - 5, 0)
        elif key == keyboard.Key.right:  # Right arrow
            p_x = min(p_x + 5, 639)
        elif key == keyboard.Key.up:  # Up arrow
            p_y = max(p_y - 5, 0)
        elif key == keyboard.Key.down:  # Down arrow
            p_y = min(p_y + 5, 479)
    except AttributeError:
        pass  # For other keys, we ignore

# Initialize camera and get tags
tags = camera.get_tag_orientation()
points = []

# Start keyboard listener without using threading
listener = keyboard.Listener(on_press=on_press)
listener.start()  # Start the listener in a non-blocking way

# Main loop
while True:
    # Capture the depth and points from the camera
    depth = camera.frame.depth.get_distance(p_x, p_y)
    for tag in tags:
        points.append(get_world_cord(camera.color_intrinsics, tag, p_x, p_y, depth))

    # Update the camera frame with the current point (for visualization purposes)
    camera.frame.points["Point1"] = (p_x, p_y, "blue")

    # Calculate errors or perform other tasks
    calculate_errors(camera.color_intrinsics, tags, points)
