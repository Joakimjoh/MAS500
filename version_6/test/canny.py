import cv2
import numpy as np

# Load the image
image = cv2.imread("depth_frame.png")

# Check if the image was loaded correctly
if image is None:
    print("Error: Could not load image. Check the file path!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 100, 200)

# Show original and edge-detected images
cv2.imshow("Original", image)
cv2.imshow("Edges", edges)

# Press 'q' to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
