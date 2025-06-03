"""Standard Library"""
import threading

"""Third-Party Libraries"""
from scipy.ndimage import label
import pyrealsense2 as rs
import numpy as np
import apriltag
from pupil_apriltags import Detector
import cv2
import csv

"""Internal Modules"""
from frame import Frame

class Camera:
    """Handles RealSense camera initialization and continuous frame fetching."""
    def __init__(self):
        """
        Initialize the RealSense camera, configure stream settings,
        compute camera intrinsics/extrinsics, and start a thread for real-time frame updates.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Camera intrinsics
        self.profile = self.pipeline.get_active_profile()
        depth_stream = self.profile.get_stream(rs.stream.depth)
        color_stream = self.profile.get_stream(rs.stream.color)
        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # Matrix and coefficient of color frame
        self.camera_matrix = np.array([[self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
                          [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                          [0, 0, 1]])
        self.dist_coeffs = np.array(self.color_intrinsics.coeffs)

        # Use manually calibrated intrinsics
        # self.camera_matrix = np.array([
        #     [575.85, 0.0, 329.62],
        #     [0.0, 574.11, 255.56],
        #     [0.0, 0.0, 1.0]
        # ])
        # self.dist_coeffs = np.array([[0.05279, -0.0198, 0.00563, 0.0039, -0.50307]])

        # Compute extrinsics (depth-to-color)
        self.depth_to_color_extrinsics = depth_stream.get_extrinsics_to(color_stream)
        self.extrinsics_rotation = np.array(self.depth_to_color_extrinsics.rotation).reshape(3, 3)
        self.extrinsics_translation = np.array(self.depth_to_color_extrinsics.translation).reshape(3, 1)
        self.extrinsics_matrix = np.hstack((self.extrinsics_rotation, self.extrinsics_translation))

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        # Get the device
        device = self.pipeline.get_active_profile().get_device()
        self.rgb_sensor = None

        for sensor in device.query_sensors():
            if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                self.rgb_sensor = sensor
                break

        if self.rgb_sensor:
            self.rgb_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
            # Set RGB sensor options
            self.rgb_sensor.set_option(rs.option.exposure, 166.0)     # Only used if auto-exposure is OFF
            self.rgb_sensor.set_option(rs.option.brightness, 5.0)
            self.rgb_sensor.set_option(rs.option.contrast, 50.0)
            self.rgb_sensor.set_option(rs.option.saturation, 30.0)
            self.rgb_sensor.set_option(rs.option.sharpness, 100.0)

        # Set depth sensor preset
        self.depth_sensor.set_option(rs.option.visual_preset, 5)
        
        # ---- Variable Initialization ----
        self.manual_mode = True
        self.key = None
        self.frame = Frame()  # Initialize a frame object

        # Start the frame-fetching thread
        self.running = True
        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

        self.thread.join(timeout=2)

    def update_frames(self):
        """
        Continuously fetch aligned color and depth frames,
        populate overlays if in manual mode, and display the frame with user interaction.
        """
        while self.running:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            self.frame.color = aligned_frames.get_color_frame()
            self.frame.depth = aligned_frames.get_depth_frame()

            if not self.frame.color or not self.frame.depth:
                continue

            # Convert RealSense frame to numpy array
            self.frame.color = np.asanyarray(self.frame.color.get_data())
            self.frame.color_standard = self.frame.color.copy()

            if self.frame.color is not None and not self.frame.center_x and not self.frame.center_y:
                height, width, _ = self.frame.color.shape
                self.frame.center_x, self.frame.center_y = width // 2, height // 2

            mode = "Manual" if self.manual_mode else "Auto"
            self.frame.text_mode = f"{mode} - Press 'r' to change mode"
            
            # Add attributes to frame
            if self.manual_mode:
                self.frame.populate()

            # Display frame
            self.frame.display()

            self.key = cv2.waitKey(1) & 0xFF

            # Change operating mode
            if self.key == ord('r'):
                self.manual_mode = not self.manual_mode

            # Close streaming frames
            if self.key == 27:  # ESC key
                self.stop()
                break

    def stop(self):
        """
        Stop the RealSense pipeline and close any display windows.
        """
        self.running = False
        self.frame.close()
        self.pipeline.stop()

    def mouse_callback(self, event, x, y, flags, param):
        """
        Internal OpenCV mouse callback to capture click coordinates.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)

    def wait_for_click(self):
        """
        Block execution until a mouse click is detected on the frame window,
        then return the clicked (x, y) pixel location.
        """
        self.clicked_point = None
        cv2.setMouseCallback(self.frame.title, self.mouse_callback)
        while self.clicked_point is None:
            pass
        
        return self.clicked_point

    def get_depth(self, point, depth_image=None):
        """
        Retrieve depth at a pixel location using either the provided depth image
        or the live depth stream. Applies a correction offset.

        :param point: (x, y) pixel coordinate
        :param depth_image: Optional numpy depth array
        :return: Depth in meters, or 0 if unavailable
        """
        
        px, py = point
        # Use depth_image if given, else fallback to live API
        if depth_image is not None:
            if 0 <= px < depth_image.shape[1] and 0 <= py < depth_image.shape[0]:
                d = depth_image[py, px]
            else:
                d = 0
        else:
            d = self.frame.depth.get_distance(px, py)

        d -= 0.01  # Adjust for camera offset
        if d > 0:
            return d
        
        return 0
                    

    def pixel_to_coordsystem(self, tag, point_pixel, adjust_error = False):
        """
        Convert a 2D pixel point to a 3D point in the AprilTag coordinate system.

        :param tag: AprilTag object containing pose
        :param point_pixel: (x, y) or (x, y, z) point in pixel space
        :param adjust_error: Whether to apply learned error correction
        :return: 3D point in tag frame or None
        """
        if point_pixel is None:
            return None

        rvec, tvec = tag.orientation
        rmat, _ = cv2.Rodrigues(rvec)

        # Handle the case where point_pixel contains 2 or 3 values
        if len(point_pixel) == 2:  # Only x, y provided
            x, y = point_pixel
            z = self.get_depth(point_pixel)

        elif len(point_pixel) == 3:  # x, y, z provided
            x, y, z = point_pixel

        else:
            return None

        # Camera intrinsic parameters
        fx, fy = self.color_intrinsics.fx, self.color_intrinsics.fy
        cx, cy = self.color_intrinsics.ppx, self.color_intrinsics.ppy

        # Convert pixel coordinates to camera coordinates
        X_camera = (x - cx) * z / fx
        Y_camera = (y - cy) * z / fy
        Z_camera = z

        point_camera = np.array([[X_camera], [Y_camera], [Z_camera]])

        # Transform camera coordinates to AprilTag coordinates
        point = np.dot(rmat.T, (point_camera - tvec))

        if adjust_error:
            point = tag.adjust_error(point)

        return point

    def coordsystem_to_pixel(self, tag, point_tag):
        """
        Project a 3D point in AprilTag frame to pixel coordinates in the camera image.

        :param tag: AprilTag object
        :param point_tag: (x, y, z) point in tag frame
        :return: 2D pixel coordinate or None
        """

        if point_tag is None or len(point_tag) != 3:
            return None

        # Ensure point is a NumPy array and undo any depth correction
        point_tag = np.array(point_tag)
        point_tag_corrected = tag.reverse_adjust_error(point_tag.copy())

        # Get tag pose
        rvec, tvec = tag.orientation
        rmat, _ = cv2.Rodrigues(rvec)

        # Transform point from tag frame to camera frame
        point_tag_reshaped = point_tag_corrected.reshape((3, 1))
        point_camera = rmat @ point_tag_reshaped + tvec
        Xc, Yc, Zc = point_camera.flatten()

        if Zc <= 0:
            return None

        # Camera intrinsics
        fx, fy = self.color_intrinsics.fx, self.color_intrinsics.fy
        cx, cy = self.color_intrinsics.ppx, self.color_intrinsics.ppy

        # Project to 2D pixel
        u = Xc * fx / Zc + cx
        v = Yc * fy / Zc + cy

        return np.array([int(round(u)), int(round(v))])
    
    def get_orientation(self, side=None):
        """
        Detect AprilTags in the image and estimate pose of a specified tag.
        Applies optional corrections for tag position and visualizes axes.

        :param side: 'left', 'right', or None to auto-select
        :return: Tuple (rotation_vector, translation_vector) or None if not found
        """

        tag_size = 1 # set as unitless

        # Create detector (do this once in __init__ in real usage)
        detector = Detector(families="tag36h11")

        if self.frame.color is None:
            return None

        # Convert to grayscale
        gray_image = cv2.cvtColor(self.frame.color, cv2.COLOR_BGR2GRAY)

        # Get intrinsics
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Detect tags and estimate pose
        detections = detector.detect(
            gray_image,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=tag_size
        )

        if not detections:
            return None

        # Choose tag based on side
        if side == "left":
            detection = min(detections, key=lambda d: d.center[0])
        elif side == "right":
            detection = max(detections, key=lambda d: d.center[0])
        else:
            detection = detections[0]

        # Get pose
        rmat = detection.pose_R  # 3x3 rotation matrix
        tvec = detection.pose_t.reshape(3, 1)  # 3x1 translation vector

        # Check if Z-axis points away from camera (i.e., tag is flipped)
        z_axis = rmat[:, 2]
        if z_axis[2] > 0:
            rmat[:, 1] *= -1  # flip Y
            rmat[:, 2] *= -1  # flip Z

        # Convert rotation matrix to rotation vector
        rvec, _ = cv2.Rodrigues(rmat)

        # Optional: visualize coordinate axes
        axis_points = np.array([
            [0, 0, 0],
            [1, 0, 0],  # X (red)
            [0, 1, 0],  # Y (green)
            [0, 0, 1]   # Z (blue)
        ], dtype=np.float32)

        imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        self.frame.axes[detection.tag_id] = imgpts

        # Get tag corners from detection (already in pixel coordinates)
        corners = detection.corners  # shape: (4, 2)

        # Create a mask to cover the polygon area of the tag
        mask = np.zeros_like(gray_image, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(corners), 255)

        # Get coordinates of all pixels inside the tag region
        ys, xs = np.where(mask == 255)

        # Collect depth values at those pixels
        depth_values = [
            self.get_depth((int(x), int(y)))
            for x, y in zip(xs, ys)
            if self.get_depth((int(x), int(y))) > 0
        ]

        if not depth_values:
            return None  # Can't compute real-world depth

        # Use median to reduce noise
        Z_real_tag = np.median(depth_values)

        # Compute scale factor and apply to tvec
        scale_factor = Z_real_tag / tvec[2]
        tvec_real = tvec * scale_factor

        R_tag_to_cam, _ = cv2.Rodrigues(rvec)
        tag_y_axis_in_camera = R_tag_to_cam[:, 1].reshape(3, 1)

        if side == "left":
            tvec_real += tag_y_axis_in_camera * 0.11
        elif side == "right":
            tvec_real -= tag_y_axis_in_camera * 0.11
        
        tag_z_axis_in_camera = R_tag_to_cam[:, 2].reshape(3, 1)
        tvec_real += tag_z_axis_in_camera * -0.01

        return rvec, tvec_real
    
    def create_sample_region(self):
        """
        Sample a central region of the color+depth image (50% size),
        retrieve depth for each pixel, and save data to a CSV file.
        """
        # Image dimensions
        img_height, img_width, _ = self.frame.color.shape

        # Define the size and position of the central square (50% of frame)
        square_size = int(min(img_height, img_width) * 0.5)
        square_half = square_size // 2
        center_x, center_y = img_width // 2, img_height // 2

        top_left_x = center_x - square_half
        top_left_y = center_y - square_half
        bottom_right_x = center_x + square_half
        bottom_right_y = center_y + square_half

        region_data = []

        depth_image = np.asanyarray(self.frame.depth.get_data()).copy()

        for y in range(top_left_y, bottom_right_y):
            for x in range(top_left_x, bottom_right_x):
                raw_depth = self.get_depth((x, y), depth_image=depth_image)
                depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
                depth_in_meters = raw_depth * depth_scale
                region_data.append((x, y, depth_in_meters))

        # Save depth data to CSV files
        with open("region.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "depth"])
            writer.writerows(region_data)

    def get_depth_map_object(self, bot_left, bot_right):
        """
        Create a color-coded depth map of the largest detected object
        using 3D triangulation from both robot arms’ camera perspectives.

        :param bot_left: Left robot arm object
        :param bot_right: Right robot arm object
        :return: Color image with depth encoding or original image on failure
        """
        _, mask = self.frame.detect_largest_object()
        color_image = self.frame.color_standard.copy()
        depth_image = np.asanyarray(self.frame.depth.get_data())
        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        mask_indices = np.column_stack(np.where(mask == 255))
        if mask_indices.size == 0:
            return color_image
        mask_indices = mask_indices[:, [1, 0]]  # (x, y)

        # === Collect 3D points (left and right)
        px_py_list, pt_l_list, pt_r_list = [], [], []

        for px, py in mask_indices:
            raw_depth = self.get_depth((px, py), depth_image=depth_image)
            if raw_depth == 0:
                continue
            depth_m = raw_depth * depth_scale
            pt_l = self.pixel_to_coordsystem(bot_left.tag, (px, py, depth_m), adjust_error=False)
            pt_r = self.pixel_to_coordsystem(bot_right.tag, (px, py, depth_m), adjust_error=False)

            if pt_l is not None and pt_r is not None and np.isfinite(pt_l[2]) and np.isfinite(pt_r[2]):
                px_py_list.append((px, py))
                pt_l_list.append(pt_l)
                pt_r_list.append(pt_r)

        if not pt_l_list:
            return color_image
        
        pt_l_array = np.array(pt_l_list, dtype=np.float32).reshape(-1, 3)
        pt_r_array = np.array(pt_r_list, dtype=np.float32).reshape(-1, 3)
        adj_l = bot_left.tag.batch_adjust_error(pt_l_array)
        adj_r = bot_right.tag.batch_adjust_error(pt_r_array)

        depth_data = []
        for (px, py), pl, pr in zip(px_py_list, adj_l, adj_r):
            z_avg = (pl[2] + pr[2]) / 2
            depth_data.append((px, py, z_avg))

        if not depth_data:
            return color_image

        depth_data = np.array(depth_data)
        px = depth_data[:, 0].astype(int)
        py = depth_data[:, 1].astype(int)
        z = depth_data[:, 2]

        z_min, z_max = z.min(), z.max()
        z_range = z_max - z_min if z_max > z_min else 1.0
        ratios = ((z - z_min) / z_range).clip(0, 1)

        r = (ratios * 255).astype(np.uint8)
        g = ((1 - ratios) * 255).astype(np.uint8)
        b = np.zeros_like(r)

        # === Apply colors safely
        h, w = color_image.shape[:2]
        valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        px = px[valid]
        py = py[valid]
        r = r[valid]
        g = g[valid]

        color_image[py, px, 0] = r  # Blue
        color_image[py, px, 1] = g
        color_image[py, px, 2] = 0

        cv2.imwrite("/home/student/Documents/MAS500/depth_map.png", color_image)
        return color_image