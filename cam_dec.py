import cv2
import numpy as np

# Camera calibration parameters (example values, replace with your calibration data)
camera_matrix = np.array([
    [1000, 0, 320],  # fx, 0, cx
    [0, 1000, 240],  # 0, fy, cy
    [0, 0, 1]        # 0,  0,  1
], dtype=np.float32)

# Dummy distortion coefficients (replace with actual values if needed)
dist_coeffs = np.zeros(5)

def calculate_xyz(pixel_coords, depth, camera_matrix):
    """Calculate the (x, y, z) coordinates of an object in centimeters."""
    u, v = pixel_coords
    uv1 = np.array([u, v, 1], dtype=np.float32).reshape(3, 1)  # Pixel coordinates in homogeneous form
    xyz_camera = np.dot(np.linalg.inv(camera_matrix), uv1) * depth
    return xyz_camera.flatten() / 10  # Convert from mm to cm

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for red color (adjust based on lighting conditions)
    lower_red1 = np.array([0, 120, 70])     # Lower range for red
    upper_red1 = np.array([10, 255, 255])  # Upper range for red
    lower_red2 = np.array([170, 120, 70])  # Second lower range for red (due to HSV wrap-around)
    upper_red2 = np.array([180, 255, 255]) # Second upper range for red

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours of red objects
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Process each detected contour
        for contour in contours:
            # Filter small contours to avoid noise
            if cv2.contourArea(contour) < 50:
                continue

            # Get bounding box and center of the object
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            # Assume known depth in millimeters (e.g., 1000 mm for this example)
            depth = 1000  # Replace with actual depth measurement or sensor data

            # Calculate 3D coordinates in centimeters
            xyz = calculate_xyz((center_x, center_y), depth, camera_matrix)

            # Print and display the result
            print(f"Red Object Detected at 3D Coordinates (cm): {xyz}")

            # Draw the detection and label it
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"3D (cm): {xyz.round(2)}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the video feed with detections
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
