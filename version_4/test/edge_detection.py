import cv2
import numpy as np
import pyrealsense2 as rs

def generate_random_color():
    """
    Generate a random color in BGR format.
    :return: A tuple representing a random color (B, G, R).
    """
    return tuple(np.random.randint(0, 256, 3).tolist())  # Ensure it's a tuple of 3 integers

def enhance_edges(image):
    """
    Enhance the edges in the image using Sobel operator and sharpening filter.
    :param image: Input image.
    :return: Image with enhanced edges.
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of red color in HSV space
    lower_red = np.array([0, 100, 100])  # Lower bound of red
    upper_red = np.array([10, 255, 255])  # Upper bound of red

    # Create a mask to isolate the red object
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    # Apply the red mask to the original image
    red_object = cv2.bitwise_and(image, image, mask=red_mask)

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(red_object, cv2.COLOR_BGR2GRAY)

    # Apply Sobel edge detection (both X and Y directions)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the magnitude of the gradients
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)

    # Normalize to 0-255 and convert back to uint8
    sobel_edges = cv2.convertScaleAbs(sobel_edges)

    # Sharpen the original image using a kernel
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  # Sharpening filter
    sharpened = cv2.filter2D(red_object, -1, kernel)

    # Convert sobel_edges to 3-channel if needed (make it match the 3-channel sharpened image)
    sobel_edges_3channel = cv2.cvtColor(sobel_edges, cv2.COLOR_GRAY2BGR)

    # Resize sobel_edges_3channel to match the sharpened image size if necessary
    if sobel_edges_3channel.shape != sharpened.shape:
        sobel_edges_3channel = cv2.resize(sobel_edges_3channel, (sharpened.shape[1], sharpened.shape[0]))

    # Combine the sharpened image with the Sobel edges for better visibility of folds
    enhanced_image = cv2.addWeighted(sharpened, 0.7, sobel_edges_3channel, 0.3, 0)

    return enhanced_image, red_mask

def find_folds_and_contours(image, red_mask):
    """
    Find the folds in the red object and draw contours on them.
    :param image: Input image.
    :param red_mask: Mask of the red region.
    :return: Image with contours drawn on the folds.
    """
    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort contours by area (to focus on the largest ones, which could represent folds)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Draw contours on the original image
    for i, contour in enumerate(contours[:5]):  # Consider top 5 largest contours
        if cv2.contourArea(contour) > 500:  # Minimum area to avoid noise
            color = generate_random_color()  # Generate a random color
            cv2.drawContours(image, [contour], -1, color, 2)  # Draw contour with random color

    return image

# Set up RealSense camera pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Adjust resolution as needed
pipeline.start(config)

while True:
    # Capture a frame from the RealSense camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise ValueError("No color frame received from the camera")

    # Convert the color frame to a numpy array
    image_np = np.asanyarray(color_frame.get_data())

    # Enhance the edges and get the red mask
    enhanced_image, red_mask = enhance_edges(image_np)

    # Find the folds and contours and draw them on the enhanced image
    result_image = find_folds_and_contours(enhanced_image, red_mask)

    # Display the result
    cv2.imshow('Detected Folds and Edges', result_image)

    # Exit condition: press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
pipeline.stop()
cv2.destroyAllWindows()
