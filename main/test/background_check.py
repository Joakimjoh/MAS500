import pyrealsense2 as rs
import numpy as np
import cv2
import os

BACKGROUND_PATH = 'empty_workspace.png'

def capture_background_frame(pipeline):
    print("[INFO] Background image not found.")
    print("Please ensure the workspace is empty.")
    print("Press 's' to save the current frame as background...")

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("Capture Empty Workspace", color_image)

        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(BACKGROUND_PATH, color_image)
            print("[INFO] Background image saved.")
            cv2.destroyWindow("Capture Empty Workspace")
            return

def detect_largest_object(current_frame, background_path=BACKGROUND_PATH, min_area=2000):
    background_img = cv2.imread(background_path)

    if background_img is None or current_frame is None:
        print("[Error] Missing background or current frame.")
        return None, None, None, None

    # Convert to grayscale
    bg_gray = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    bg_blur = cv2.GaussianBlur(bg_gray, (3, 3), 0)
    cur_blur = cv2.GaussianBlur(cur_gray, (3, 3), 0)

    # Difference image
    diff = cv2.absdiff(bg_blur, cur_blur)

    # Threshold
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 30 or h < 30:
            continue

        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        temp_mask = np.zeros_like(bg_gray)
        cv2.drawContours(temp_mask, [contour], -1, 255, -1)
        mean_bg = cv2.mean(bg_gray, mask=temp_mask)[0]
        mean_cur = cv2.mean(cur_gray, mask=temp_mask)[0]
        if abs(mean_bg - mean_cur) < 30:
            continue

        if area > max_area:
            largest_contour = contour
            max_area = area

    return largest_contour, clean, cur_blur, current_frame

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        for _ in range(30):  # Warm-up frames
            pipeline.wait_for_frames()

        if not os.path.exists(BACKGROUND_PATH):
            capture_background_frame(pipeline)

        print("[INFO] Capturing frame with textile...")
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("No color frame received.")
            return

        color_image = np.asanyarray(color_frame.get_data())
        largest_contour, mask, blurred, original = detect_largest_object(color_image)

        if largest_contour is not None:
            output_image = original.copy()
            cv2.drawContours(output_image, [largest_contour], -1, (0, 255, 0), 3)
        else:
            print("[INFO] No textile detected.")
            output_image = original

        # Show result windows
        cv2.imshow("Original Frame", original)
        cv2.imshow("Blurred Grayscale", blurred)
        cv2.imshow("Detected Textile", output_image)

        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
