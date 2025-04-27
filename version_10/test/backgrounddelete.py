import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)

# A variable to store the background image (first picture)
background_image = None
background_saved = False  # Flag to check if the background has been saved

# List to store contours (outlines) of detected objects
saved_contours = []

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    if not background_saved:
        # Capture the first frame as the background
        background_image = frame.copy()
        background_saved = True
        print("Background image captured. Now, press 'd' to detect new objects.")

    # Apply background subtractor (subtract the background)
    fgmask = fgbg.apply(frame)

    # Perform morphological operations to clean up the mask
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Find contours of the moving objects
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If a new object is detected, draw an outline around it
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            # Draw the outline of the detected object (new object) on the frame
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)  # Green color, thickness 3

            # Save the contour to persist the outline when key is pressed
            saved_contours.append(contour)

    # Display the result with the drawn outline
    cv2.imshow('Frame', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('d'):  # Press 'd' to detect new objects and draw outline
        print("Detecting new objects...")
        # Reset the background subtractor to detect new changes
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)  # Reset
    elif key == ord('s'):  # Press 's' to save the detected outline
        print("Outline saved.")
        # Optionally, save the frame with the drawn contours to an image
        cv2.imwrite("image_with_outline.png", frame)

# Release resources
cap.release()
cv2.destroyAllWindows()
