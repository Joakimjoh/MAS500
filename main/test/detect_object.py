import cv2
import numpy as np
import pyrealsense2 as rs
import os
import time

# Initialize the Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Configure to use color stream

# Start the pipeline
pipeline.start(config)

# Check if background file exists
background_file = 'empty_table.jpg'

time.sleep(2)  # Allow some time for the camera to adjust

if not os.path.exists(background_file):
    # Wait until we get a valid frame before capturing
    print("Background image not found. Capturing the empty table image.")
    while True:
        frames = pipeline.wait_for_frames()  # Get the next set of frames
        color_frame = frames.get_color_frame()  # Get the color frame
        
        # Convert to numpy array
        empty_frame = np.asanyarray(color_frame.get_data())
        
        # Check if the frame is not black (i.e., it's a valid frame)
        if np.any(empty_frame):  # If there's any non-zero value, it's a valid frame
            cv2.imwrite(background_file, empty_frame)
            print("Empty table image captured and saved!")
            break

else:
    print("Background image found. Using it for comparison.")

# Capture the current scene after placing the object
print("Place the object on the table and press Enter to capture the current image...")
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    current_frame = np.asanyarray(color_frame.get_data())

    cv2.imshow('Current Scene - Place the object and press Enter to Capture', current_frame)

    key = cv2.waitKey(1)  # Capture key press
    if key == 13:  # 13 is the Enter key in OpenCV
        cv2.imwrite('current_table.jpg', current_frame)
        print("Current image with object captured!")
        break

# Close the RealSense pipeline after capturing both images
pipeline.stop()
cv2.destroyAllWindows()

# Load the background (empty) image and the current image for comparison
empty_image = cv2.imread(background_file)
current_image = cv2.imread('current_table.jpg')

# Convert images to grayscale
empty_gray = cv2.cvtColor(empty_image, cv2.COLOR_BGR2GRAY)
current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred_empty = cv2.GaussianBlur(empty_gray, (5, 5), 0)
blurred_current = cv2.GaussianBlur(current_gray, (5, 5), 0)

# Compute the absolute difference between the two images
diff = cv2.absdiff(blurred_empty, blurred_current)

# Threshold the difference to make changes more noticeable
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Apply morphological operations to clean up the image (remove small noise and fill gaps)
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Closing operation (dilate and erode)

# Find contours of the detected differences (new object)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours around the detected object(s)
output_image = current_image.copy()
for contour in contours:
    if cv2.contourArea(contour) > 500:  # Ignore small contours (noise)
        # Draw the contour (outline of the object)
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)

# Show the result
cv2.imshow('Detected Object - Outline', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
