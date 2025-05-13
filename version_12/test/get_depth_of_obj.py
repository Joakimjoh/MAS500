from sklearn.linear_model import LinearRegression
import pyrealsense2 as rs
import numpy as np
import apriltag
import cv2
import csv
import os
import joblib

def red_object(color_image, min_area=1000):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Define the range for red color in HSV space
    lower_red1 = np.array([0, 50, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 30])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red regions in both ranges
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the two masks
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Filter by area
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if not large_contours:
        return None, None

    # Create mask from valid contours
    object_mask = np.zeros_like(red_mask)
    for cnt in large_contours:
        cv2.drawContours(object_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return object_mask

def get_point_mask(object_mask):
        # Get all (x, y) pixel coordinates of red regions
    points_list = np.column_stack(np.where(object_mask == 255))
    points_list = points_list[:, [1, 0]]  # Convert to (x, y)
    return points_list

def get_depth_for_multiple_points(model, points_list, orientation, rmat, depth_frame, intrinsics):
    # Create a dictionary to store the results
    depth_data = []
    _, tvec = orientation

    # Loop over each point in the list
    for p_x, p_y in points_list:
        # Only process points inside the red mask
        if p_x < depth_frame.width - 1 and p_y < depth_frame.height - 1:
            
            Z_real = depth_frame.get_distance(p_x, p_y)  # Depth at (p_x, p_y)

            if Z_real > 0:
                fx, fy, cx, cy = intrinsics

                # Compute camera coordinates for point
                X_camera = (p_x - cx) * Z_real / fx
                Y_camera = (p_y - cy) * Z_real / fy
                Z_camera = Z_real
                point_camera = np.array([[X_camera], [Y_camera], [Z_camera]])

                # **Convert camera coordinate to AprilTag coordinate**
                point_tag = np.dot(rmat.T, (point_camera - tvec))

                point_tag = adjust_error(model, point_tag)

                # Store the result in the dictionary
                depth_data.append((p_x, p_y, point_tag[2]))  # Store z value for the pixel (p_x, p_y)

    return depth_data

def create_sample_region(color_frame, depth_frame):
    """Get pixel and depth points of a region 50% the size of the frame"""
    img_height, img_width, _ = color_frame.shape
    square_size = int(min(img_height, img_width) * 0.5)
    square_half = square_size // 2
    center_x, center_y = img_width // 2, img_height // 2

    top_left_x = center_x - square_half
    top_left_y = center_y - square_half
    bottom_right_x = center_x + square_half
    bottom_right_y = center_y + square_half

    region_data = []

    for y in range(top_left_y, bottom_right_y):
        for x in range(top_left_x, bottom_right_x):
            depth = depth_frame.get_distance(x, y)
            region_data.append((x, y, depth))

    with open("region.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "depth"])
        writer.writerows(region_data)


def get_orientation(color_frame, depth_frame, camera_matrix, dist_coeffs):
    """Estimate AprilTag orientation in camera frame"""
    object_points = np.array([
        [-1, 1, 0],
        [1, 1, 0],
        [1, -1, 0],
        [-1, -1, 0],
        [-1, -1, 1]
    ], dtype=np.float32)

    detector = apriltag.Detector()

    gray_image = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray_image)

    if not detections:
        return None

    detection = detections[0]

    corners = np.array(detection.corners, dtype=np.float32)
    tag_cx, tag_cy = np.array(detection.center, dtype=np.int32)

    _, rvec, tvec = cv2.solvePnP(object_points[:4], corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rvec, tvec = cv2.solvePnPRefineLM(object_points[:4], corners, camera_matrix, dist_coeffs, rvec, tvec)

    depth_values = []
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            depth = depth_frame.get_distance(tag_cx + dx, tag_cy + dy)
            if depth > 0:
                depth_values.append(depth)

    if not depth_values:
        return None

    Z_real_tag = np.median(depth_values)
    scale_factor = Z_real_tag / tvec[2]
    tvec_real = tvec * scale_factor

    imgpts, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    return (rvec, tvec_real), imgpts


def pixel_to_coordsystem(depth_frame, intrinsics, orientation, rmat, point_pixel):
    if point_pixel is None:
        print("[Error] pixel_to_coordsystem: Received None as point_pixel")
        return None

    _, tvec = orientation

    if len(point_pixel) == 2:
        x, y = point_pixel
        z = depth_frame.get_distance(x, y)
    elif len(point_pixel) == 3:
        x, y, z = point_pixel
    else:
        print("[Error] pixel_to_coordsystem: Invalid point_pixel format:", point_pixel)
        return None

    fx, fy, cx, cy = intrinsics

    X_camera = (x - cx) * z / fx
    Y_camera = (y - cy) * z / fy
    Z_camera = z

    point_camera = np.array([[X_camera], [Y_camera], [Z_camera]])
    point = np.dot(rmat.T, (point_camera - tvec))
    return point.flatten()

def get_region_data(color_frame, depth_frame):
    if not os.path.exists("region.csv"):
        create_sample_region(color_frame, depth_frame)

    region_data = []
    with open("region.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            x, y, depth = int(row[0]), int(row[1]), float(row[2])
            region_data.append((x, y, depth))

    return region_data


def linear_reg(x_values, y_values, z_values):
    X = np.column_stack((x_values, y_values))
    model = LinearRegression()
    model.fit(X, z_values)
    return model


def get_error_model(color_frame, depth_frame, orientation, rmat, intrinsics):
    region_data = get_region_data(color_frame, depth_frame)

    x_values, y_values, z_values = [], [], []

    for x, y, depth in region_data:
        if depth > 0:
            point_tag = pixel_to_coordsystem(depth_frame, intrinsics, orientation, rmat, (x, y, depth))
            x_values.append(point_tag[0])
            y_values.append(point_tag[1])
            z_values.append(point_tag[2])

    return linear_reg(np.array(x_values), np.array(y_values), np.array(z_values))


def get_linear_reg_error(model, point):
    x, y = point
    input_array = np.array([x, y]).reshape(1, -1)
    z_pred = model.predict(input_array)
    return z_pred[0]


def adjust_error(model, point):
    error = get_linear_reg_error(model, (point[0], point[1]))
    point[2] -= error
    point[2] = max(point[2], 0.000)
    return point

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

# Get camera intrinsics
profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color)
color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                          [0, color_intrinsics.fy, color_intrinsics.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(color_intrinsics.coeffs)

depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 5)

intrinsics = (color_intrinsics.fx, color_intrinsics.fy,
              color_intrinsics.ppx, color_intrinsics.ppy)

# Create folder if it doesn't exist
os.makedirs("images", exist_ok=True)

frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)
color_frame = aligned_frames.get_color_frame()
depth_frame = aligned_frames.get_depth_frame()

color_image = np.asanyarray(color_frame.get_data())
cv2.imshow("Cam", color_image)

# Compute orientation, model, and red object only once
orientation, imgpts = get_orientation(color_image, depth_frame, camera_matrix, dist_coeffs)
if orientation is None:
    print("No AprilTag detected.")
    pipeline.stop()
    exit(1)

rmat, _ = cv2.Rodrigues(orientation[0])

model_path = "error_model.joblib"
if os.path.exists(model_path):
    print("Loading existing error model...")
    model = joblib.load(model_path)
else:
    print("Creating new error model...")
    model = get_error_model(color_image, depth_frame, orientation, rmat, intrinsics)
    joblib.dump(model, model_path)
    print("Model saved to:", model_path)

# Start loop for capturing and saving visualized depth




# TILLA HER!!!! KAN DU ENDRE HVORDAN BILDE REKKE DU ER PÅ!!!!!!!!!!!!!!!!!1

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# HVIS DU HAR TATT 100 BILDER OG SKAL STARTE PÅ NYTT ENDRE VERDI UNDER TIL 101!!!!!!!!!!!!!!!!!!!!!!!!

image_index = 383











print("Press 'f' to save visualized depth image. Press 'q' to quit.")

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())

    # Get the red mask (interior + outline points if you modified the function)
    red_mask = red_object(color_image)

    # Draw object outline if mask is valid
    if red_mask is not None and isinstance(red_mask, np.ndarray):
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                cv2.drawContours(color_image, [cnt], -1, (0, 255, 0), 2)

    # Draw AprilTag axes (make sure imgpts is valid)
    if 'imgpts' in locals() and imgpts is not None and np.isfinite(imgpts).all():
        imgpts = imgpts.reshape(-1, 2).astype(int)
        cv2.line(color_image, tuple(imgpts[3]), tuple(imgpts[2]), (0, 0, 255), 3)  # X-axis
        cv2.line(color_image, tuple(imgpts[3]), tuple(imgpts[0]), (0, 255, 0), 3)  # Y-axis
        cv2.line(color_image, tuple(imgpts[3]), tuple(imgpts[4]), (255, 0, 0), 3)  # Z-axis

    cv2.imshow("Cam", color_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('f'):
        points_list = get_point_mask(red_mask)
        if points_list is None:
            print("No red object detected.")
            pipeline.stop()
            exit(1)
        print(f"[{image_index}] Capturing.")

        depth_data = get_depth_for_multiple_points(model, points_list, orientation, rmat, depth_frame, intrinsics)
        z_values = np.array([point[2] for point in depth_data])
        z_values = z_values[z_values > 0]

        if len(z_values) == 0:
            print("No valid depth data.")
            continue

        low_z = np.min(z_values)
        high_z = np.max(z_values)

        for x, y, z in depth_data:
            ratio = (z - low_z) / (high_z - low_z)
            ratio = ratio.item()  # Fix for NumPy 1.25+

            r = int(0 + ratio * (255 - 0))
            g = int(255 - ratio * 255)
            b = 0
            cv2.circle(color_image, (x, y), 2, (r, g, b), -1)
        cv2.putText(color_image, f"Image {image_index}", (10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(255, 255, 255),  # White text
            thickness=2,
            lineType=cv2.LINE_AA)
        
        filename = f"images/images{image_index}.png"
        cv2.imwrite(filename, color_image)
        print(f"Saved: {filename}")
        cv2.imshow("Depth Visualization", color_image)
        print(f"[{image_index}] Captured.")
        image_index += 1

    elif key == ord('q'):
        print("Exiting.")
        break

cv2.destroyAllWindows()
pipeline.stop()
