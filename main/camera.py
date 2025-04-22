"""Standard Library"""
import threading

"""Third-Party Libraries"""
import pyrealsense2 as rs
import numpy as np
import apriltag
import cv2
import csv

"""Internal Modules"""
from frame import Frame

class Camera:
    """Handles RealSense camera initialization and continuous frame fetching."""
    def __init__(self):
        # RealSense pipeline and configuration
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
            # Set RGB sensor options
            self.rgb_sensor.set_option(rs.option.saturation, 30)
            self.rgb_sensor.set_option(rs.option.sharpness, 100)

        # Set depth sensor preset
        self.depth_sensor.set_option(rs.option.visual_preset, 5)

        self.manual_mode = False

        self.frame = Frame()  # Initialize a frame object

        # Start the frame-fetching thread
        self.running = True
        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

        self.thread.join(timeout=2)

    def update_frames(self):
        """Continuously fetch frames and display them with outlines and points."""
        while self.running:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            self.frame.color = aligned_frames.get_color_frame()
            self.frame.depth = aligned_frames.get_depth_frame()

            if not self.frame.color or not self.frame.depth:
                continue

            # Convert RealSense frame to numpy array
            self.frame.color = np.asanyarray(self.frame.color.get_data())

            if self.frame.color is not None and not self.frame.center_x and not self.frame.center_y:
                height, width, _ = self.frame.color.shape
                self.frame.center_x, self.frame.center_y = width // 2, height // 2

            mode = "Manual" if self.manual_mode else "Auto"
            text = f"{mode} Mode - Press 'r' to change mode"

            cv2.putText(self.frame.color, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add attributes to frame
            if self.manual_mode:
                self.frame.populate()

            # Display frame
            self.frame.display()

            # Change operating mode
            if cv2.waitKey(1) & 0xFF == ord('r'):
                self.manual_mode = not self.manual_mode

            # Close streaming frames
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    def stop(self):
        """Stops the camera stream and thread."""
        self.running = False
        self.frame.close()
        self.pipeline.stop()

    def get_depth(self, point):
        """Get depth at pixel coordiante with a filter for more stable depth reading"""
        x, y = point
        depth_values = []
        for _ in range(10):
            depth_value = self.frame.depth.get_distance(x, y) - 0.01 # Remove 1cm becuase of camera error
            if depth_value > 0:
                depth_values.append(depth_value)

        return np.median(depth_values) if depth_value else 0

    def pixel_to_coordsystem(self, orientation, point_pixel):
        """Convert pixel coordinates from the camera into the coordinate system of one of the robot arms."""
        rvec, tvec = orientation

        rmat, _ = cv2.Rodrigues(rvec)
        
        # Handle the case where point_pixel contains 2 or 3 values
        if len(point_pixel) == 2:  # Only x, y provided
            z = self.get_depth(point_pixel)

        elif len(point_pixel) == 3:  # x, y, z provided
            x, y, z = point_pixel

        # Camera intrinsic parameters
        fx, fy = self.color_intrinsics.fx, self.color_intrinsics.fy
        cx, cy = self.color_intrinsics.ppx, self.color_intrinsics.ppy

        # Convert pixel coordinates to camera coordinates
        X_camera = (x - cx) * z / fx
        Y_camera = (y - cy) * z / fy
        Z_camera = z  # Already given or retrieved

        point_camera = np.array([[X_camera], [Y_camera], [Z_camera]])

        # Transform camera coordinates to AprilTag coordinates
        point = np.dot(rmat.T, (point_camera - tvec))

        return point

    def coordsystem_to_pixel(self, orientation, point):
        """Convert a point from the robot arm's coordinate system to pixel coordinates in the camera's image"""
        rvec, tvec = orientation
        # Convert rotation vector to rotation matrix (robot to camera)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Translate the point from robot to camera coordinates
        point_in_camera = np.dot(rotation_matrix, point) + tvec
        
        # Convert from camera coordinates to pixel coordinates
        X_camera, Y_camera, Z_camera = point_in_camera
        
        # Ensure that Z_camera is not zero to avoid division by zero
        if Z_camera == 0:
            raise ValueError("Z coordinate in camera system is zero, cannot project to 2D.")

        # Intrinsic parameters
        fx, fy = self.color_intrinsics.fx, self.color_intrinsics.fy
        cx, cy = self.color_intrinsics.ppx, self.color_intrinsics.ppy

        # Project the point from camera coordinates to pixel coordinates using the camera's intrinsic matrix
        pixel_x = (fx * X_camera / Z_camera) + cx
        pixel_y = (fy * Y_camera / Z_camera) + cy
        
        # Return the pixel coordinates
        return int(pixel_x), int(pixel_y)
    
    def get_orientation(self, side=None):
        """Get orientation and translation relative to camera"""

        # Define normalized 3D object points for the tag (including a fifth point for the Z-axis)
        object_points = np.array([
            [-1, -1, 0],
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, -1]
        ], dtype=np.float32)

        # Detect AprilTags
        detector = apriltag.Detector()

        if self.frame.color is not None:
            gray_image = cv2.cvtColor(self.frame.color, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray_image)
        else:
            return None  # No frame to process

        if not detections:
            return None  # No tags found

        # Select the detection based on the given id
        if side == "left":
            detection = min(detections, key=lambda d: d.center[0])
        elif side == "right":
            detection = max(detections, key=lambda d: d.center[0])
        else:
            detection = detections[0]

        # Get AprilTag corner positions and center
        corners = np.array(detection.corners, dtype=np.float32)
        tag_cx, tag_cy = np.array(detection.center, dtype=np.int32)

        # Initial pose estimation using SOLVEPNP_ITERATIVE
        _, rvec, tvec = cv2.solvePnP(object_points[:4], corners, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # Refine pose estimation
        rvec, tvec = cv2.solvePnPRefineLM(object_points[:4], corners, self.camera_matrix, self.dist_coeffs, rvec, tvec)

        # Calculate real-world depth from the tag center
        depth_values = []
        for dx in range(-3, 4):  # Use a 7x7 neighborhood
            for dy in range(-3, 4):
                depth = self.get_depth((tag_cx + dx, tag_cy + dy))
                if depth > 0:
                    depth_values.append(depth)

        if len(depth_values) == 0:
            return None  # Can't compute orientation without depth info

        Z_real_tag = np.median(depth_values)

        # Compute scale factor based on depth
        scale_factor = Z_real_tag / tvec[2]
        tvec_real = tvec * scale_factor

        # Visualize tag axes
        imgpts, _ = cv2.projectPoints(object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1, 2)
        self.frame.axes[detection.tag_id] = imgpts

        # Return the pose (rotation and real-world translation)
        orientation = (rvec, tvec_real)

        return orientation
    
    def create_sample_region(self):
        """Get pixel and depth points of a region 50% the size of the frame"""
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

        for y in range(top_left_y, bottom_right_y):
            for x in range(top_left_x, bottom_right_x):
                depth = self.get_depth((x, y))
                region_data.append((x, y, depth))

        # Save depth data to CSV files
        with open("region.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "depth"])
            writer.writerows(region_data)