import cv2
import numpy as np
import pyrealsense2 as rs
import os
import time

# Initialize the Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

background_file = 'empty_table.jpg'
time.sleep(2)  # Let camera stabilize

# Capture background if not found
if not os.path.exists(background_file):
    print("Capturing empty table image...")
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        empty_frame = np.asanyarray(color_frame.get_data())

        if np.any(empty_frame):
            cv2.imwrite(background_file, empty_frame)
            print("Empty background saved.")
            break
else:
    print("Using existing empty background image.")

# Load background
empty_image = cv2.imread(background_file)
empty_gray = cv2.cvtColor(empty_image, cv2.COLOR_BGR2GRAY)
blurred_empty = cv2.GaussianBlur(empty_gray, (5, 5), 0)

# Main loop — continuously detect new objects
print("Starting real-time object detection. Press 'q' to quit.")
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    current_frame = np.asanyarray(color_frame.get_data())

    # Convert to grayscale and blur
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    blurred_current = cv2.GaussianBlur(current_gray, (5, 5), 0)

    # Image difference
    diff = cv2.absdiff(blurred_empty, blurred_current)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Morphological clean-up
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find and draw contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = current_frame.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)

    # Display
    cv2.imshow('Real-Time Object Detection', output_image)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
pipeline.stop()
cv2.destroyAllWindows()
