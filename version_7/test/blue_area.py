import cv2
import numpy as np

# Load the image
image = cv2.imread('depth_frame.png')

# Check if the image was loaded correctly
if image is None:
    print("Error: Could not load image. Check the file path!")
    exit()

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the blue color range in HSV space
lower_blue = np.array([100, 150, 50])  # Lower bound of blue
upper_blue = np.array([140, 255, 255])  # Upper bound of blue

# Create a mask for blue areas in the image
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Apply Gaussian blur to smooth out edges and reduce noise
blue_mask = cv2.GaussianBlur(blue_mask, (5, 5), 0)

# Find contours of the blue regions in the mask
contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("No blue contours detected!")
else:
    # Find the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the largest blue contour on the original image
    output_image = image.copy()
    cv2.drawContours(output_image, [largest_contour], -1, (0, 255, 0), 2)  # Green outline

    # Show the original and the image with the contour
    cv2.imshow("Original Image", image)
    cv2.imshow("Blue Contour", output_image)

    # Save the image with the largest contour
    cv2.imwrite("blue_contour_image.png", output_image)

    # Wait for the user to press 'q' to close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
