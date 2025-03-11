import cv2
import numpy as np

# Load the image
image = cv2.imread('color_frame.jpg')
original_image = image.copy()

# Convert the image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for green and blue colors in HSV space
# Green range in HSV
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Blue range in HSV
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# Create masks for green and blue regions
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Find contours for green and blue masks
contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Function to find the largest contour
def get_largest_contour(contours):
    if len(contours) == 0:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

# Get the largest green and blue contours
largest_green_contour = get_largest_contour(contours_green)
largest_blue_contour = get_largest_contour(contours_blue)

# Draw the largest green and blue contours on the original image
if largest_green_contour is not None:
    cv2.drawContours(original_image, [largest_green_contour], -1, (0, 0, 255), 3)  # Green contour

if largest_blue_contour is not None:
    cv2.drawContours(original_image, [largest_blue_contour], -1, (0, 255, 255), 3)  # Blue contour

# Show the image with the largest green and blue contours outlined
cv2.imshow('Contours', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
