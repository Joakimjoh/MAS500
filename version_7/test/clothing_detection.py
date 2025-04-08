import cv2
import numpy as np
import pyrealsense2 as rs
from google.cloud import vision
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/student/Documents/detect-clothing-f1ca351e7c85.json"

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Initialize Google Cloud Vision client
client = vision.ImageAnnotatorClient()

def detect_shirt_in_image(image):
    # Convert the image to bytes
    _, encoded_image = cv2.imencode('.jpg', image)
    content = encoded_image.tobytes()

    # Create an Image object
    vision_image = vision.Image(content=content)

    # Perform label detection on the image
    response = client.label_detection(image=vision_image)

    # Check if "shirt" is in the labels
    labels = response.label_annotations
    for label in labels:
        print(f"Detected label: {label.description}, Confidence: {label.score}")
        if 'shirt' in label.description.lower():
            return True
    return False

while True:
    # Wait for a frame from the RealSense camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    # Convert to numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # Display the image
    cv2.imshow("RealSense Image", color_image)

    # Wait for 'q' to be pressed to trigger detection
    if cv2.waitKey(1) & 0xFF == ord('f'):
        if detect_shirt_in_image(color_image):
            print("Shirt detected in the image!")
            # Optionally, you can highlight or annotate the shirt in the image here
            cv2.putText(color_image, "Shirt Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print("Not detected")

    # If 'q' is pressed again, exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
