import cv2
import numpy as np
import time
import threading

# Global variable to control red object detection
detect = False
lock = threading.Lock()

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

def start_camera(cap):
    """Start the camera and handle red object detection based on the detect flag."""
    global detect

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    start_time = None
    final_positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Invert the camera view
        frame = cv2.flip(frame, 1)

        with lock:
            if detect:
                # Start red object detection
                if start_time is None:
                    start_time = time.time()
                    final_positions.clear()

                # Convert to HSV for color segmentation
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Define HSV range for red color
                lower_red1 = np.array([0, 120, 70])     # Lower range for red
                upper_red1 = np.array([10, 255, 255])  # Upper range for red
                lower_red2 = np.array([170, 120, 70])  # Second lower range for red
                upper_red2 = np.array([180, 255, 255]) # Second upper range for red

                # Create masks for red color
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask = cv2.bitwise_or(mask1, mask2)

                # Find contours of red objects
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Track the last detected positions of red objects
                current_positions = []
                if contours:
                    for contour in contours:
                        if cv2.contourArea(contour) < 50:
                            continue

                        # Get bounding box and center of the object
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x + w // 2
                        center_y = y + h // 2

                        # Draw the detection
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, "Red Object", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Calculate the 3D coordinates
                        depth = 1000  # Example depth value in mm
                        xyz = calculate_xyz((center_x, center_y), depth, camera_matrix)
                        current_positions.append((center_x, center_y, xyz.tolist()))

                # Update the final detected positions
                final_positions = current_positions

                # Stop detection after 5 seconds and log positions
                if time.time() - start_time > 5:
                    detect = False
                    start_time = None

                    # Log the final detected positions
                    print("Final Red Object Positions at 5-Second Mark:")
                    for pos in final_positions:
                        pixel_coords, xyz_coords = pos[:2], pos[2]
                        print(f"Pixel Coordinates: {pixel_coords}, 3D Coordinates (cm): {xyz_coords}")

            else:
                # Normal camera feed
                cv2.putText(frame, "Normal Mode", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the video feed
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

def toggle_detection():
    """Toggle the detect flag to start red object detection."""
    global detect
    while True:
        input("Press Enter to start red object detection for 5 seconds...")
        with lock:
            detect = True

# Main program
cap = cv2.VideoCapture(0)

# Start camera feed in a separate thread
camera_thread = threading.Thread(target=start_camera, args=(cap,))
camera_thread.start()

# Start the detection toggle handler
toggle_detection()
