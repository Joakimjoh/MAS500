import cv2
import numpy as np

# Load the image
image = cv2.imread("depth_frame.png")

# Check if the image was loaded correctly
if image is None:
    print("Error: Could not load image. Check the file path!")
    exit()

# Define the target color (0, 255, 0) - pure green
target_color_min = np.array([0, 100, 0])  # Slightly lower green range
target_color_max = np.array([50, 255, 0])  # Slightly higher green range

# Create a mask that identifies all pixels in the defined green range
mask = cv2.inRange(image, target_color_min, target_color_max)

# Create an output image that starts as black
output_image = np.zeros_like(image)

# Apply the mask to keep only the target color pixels in the original image
output_image[mask == 255] = image[mask == 255]

# Show the original image and the resulting image with the expanded green range
cv2.imshow("Original Image", image)
cv2.imshow("Filtered Image", output_image)

# Save the result
cv2.imwrite("filtered_image.png", output_image)

# Press 'q' to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
