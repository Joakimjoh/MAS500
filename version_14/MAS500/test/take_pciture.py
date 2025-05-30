import pyrealsense2 as rs
import numpy as np
import cv2

# === Initialize RealSense pipeline ===
pipeline = rs.pipeline()
config = rs.config()

# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Warm-up
for _ in range(10):
    pipeline.wait_for_frames()

print("📷 Press Enter to take a photo. Press Ctrl+C to quit.")
image_counter = 29

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert to numpy array and display
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("Live View (press Enter to capture)", color_image)

        key = cv2.waitKey(1)
        if key == 13:  # Enter key
            filename = f"/home/student/Documents/MAS500/test/image_{image_counter}.png"
            cv2.imwrite(filename, color_image)
            print(f"✅ Saved {filename}")
            image_counter += 1

except KeyboardInterrupt:
    print("\n👋 Exiting...")

# Cleanup
cv2.destroyAllWindows()
pipeline.stop()
