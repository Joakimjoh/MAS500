"""Standard Library"""
import threading

"""Third-Party Libraries"""
import pyrealsense2 as rs
import numpy as np
import apriltag
import cv2

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
        
        # Start streaming
        self.pipeline.start(config)

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

        self.frame = Frame()  # Initialize a frame object

        # Start the frame-fetching thread
        self.running = True
        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

    def update_frames(self):
        """Continuously fetch frames and display them with outlines and points."""
        while self.running:
            frames = self.pipeline.wait_for_frames()
            self.frame.depth = frames.get_depth_frame()
            self.frame.color = frames.get_color_frame()

            if not self.frame.color or not self.frame.depth:
                continue

            # Convert RealSense frame to numpy array
            self.frame.color = np.asanyarray(self.frame.color.get_data())

            if self.frame.color is not None and not self.frame.center_x and not self.frame.center_y:
                height, width, _ = self.frame.color.shape
                self.frame.center_x, self.frame.center_y = width // 2, height // 2

            # Add attributes to frame
            self.frame.populate()

            # Display frame
            self.frame.display()

            # Close streaming frames
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.frame.close()
                break

            # Stop camera stream
            if cv2.waitKey(1) & 0xFF == ord('f'):
                self.stop()
                break

    def stop(self):
        """Stops the camera stream and thread."""
        self.running = False
        self.pipeline.stop()

    def pixel_to_coordsystem(self, rvec, tvec, point_pixel):
        """Turn pixel coordinates from camera into coordinate system of one of the robot arms"""
        pixel_x, pixel_y = point_pixel
        # Pixel and depth values
        depth_in_meters = self.frame.depth.get_distance(pixel_x, pixel_y)  # Depth value in meters

        fx = self.depth_intrinsics.fx  # Focal length in x (in pixels)
        fy = self.depth_intrinsics.fy  # Focal length in y (in pixels)
        cx = self.depth_intrinsics.ppx  # Principal point x (in pixels)
        cy = self.depth_intrinsics.ppy  # Principal point y (in pixels)

        # Convert pixel coordinates (x, y) to camera coordinates (X, Y, Z) in meters
        X_camera = (pixel_x - cx) * depth_in_meters / fx
        Y_camera = (pixel_y - cy) * depth_in_meters / fy
        Z_camera = depth_in_meters  # Z in camera coordinates is simply the depth

        # Point in depth camera's coordinate system (homogeneous coordinate)
        point_in_depth_camera = [[X_camera], [Y_camera], [Z_camera], [1]]

        # Transform the point from depth camera to color camera
        point_in_color_camera = np.dot(self.extrinsics_matrix, point_in_depth_camera)

        # Convert rotation vector to rotation matrix (from camera to Tag 1's coordinate system)
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Transform the point from color camera's coordinate system to Tag 1's coordinate system
        point_in_tag = np.dot(rotation_matrix.T, (point_in_color_camera[:3] - tvec))  # Use 3D coordinates, not homogeneous

        return point_in_tag.reshape(1, 3)
    
    def coordsystem_to_pixel(self, rvec, tvec, point_robot):
        """Convert a point from the robot arm's coordinate system to pixel coordinates in the camera's image"""
        
        # Step 1: Convert the point from robot coordinate system to the camera coordinate system
        # Convert rotation vector to rotation matrix (robot to camera)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Translate the point from robot to camera coordinates
        point_in_camera = np.dot(rotation_matrix, point_robot) + tvec
        
        # Step 2: Convert from camera coordinates to pixel coordinates
        X_camera, Y_camera, Z_camera = point_in_camera
        
        # Ensure that Z_camera is not zero to avoid division by zero
        if Z_camera == 0:
            raise ValueError("Z coordinate in camera system is zero, cannot project to 2D.")

        # Intrinsic parameters
        fx = self.depth_intrinsics.fx  # Focal length in x (in pixels)
        fy = self.depth_intrinsics.fy  # Focal length in y (in pixels)
        cx = self.depth_intrinsics.ppx  # Principal point x (in pixels)
        cy = self.depth_intrinsics.ppy  # Principal point y (in pixels)

        # Project the point from camera coordinates to pixel coordinates using the camera's intrinsic matrix
        pixel_x = (fx * X_camera / Z_camera) + cx
        pixel_y = (fy * Y_camera / Z_camera) + cy
        
        # Return the pixel coordinates
        return int(pixel_x), int(pixel_y)
    
    def get_tag_orientation(self):
        """Get orientation and translation relative to camera"""

        # Define normalized 3D object points for the tag (no assumed size)
        object_points = np.array([
            [-1, -1, 0],
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, -1]  # Normalized tag corners
        ], dtype=np.float32)

        # Detect AprilTags
        # Valid AprilTags
        detector = apriltag.Detector()

        while True:  # Keep looping until at least two tags are detected
            if self.frame.color is not None:
                gray_image = cv2.cvtColor(self.frame.color, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(gray_image)

                if len(detections) >= 2:  # Only return when 2 or more are found
                    break

        # **Sort by the leftmost corner's x-coordinate** (smallest x first)
        detections.sort(key=lambda tag: min(tag.corners[:, 0]))  

        tags = []

        for i, detection in enumerate(detections):
            # Get AprilTag corner positions
            corners = np.array(detection.corners, dtype=np.float32)
            tag_cx, tag_cy = np.array(detection.center, dtype=np.int32)
            # SolvePnP to get the pose of the tag
            ret, rvec, tvec = cv2.solvePnP(object_points[:4], corners, self.camera_matrix, self.dist_coeffs)
            
            tvec[0] -= 0.1 # ROBOT BASE OFFSET FROM APRILTAG
            
            # Calculate real-world depth from the tag center (use median of surrounding points as depth)
            depth_values = []
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    depth = self.frame.depth.get_distance(tag_cx + dx, tag_cy + dy)
                    if depth > 0:
                        depth_values.append(depth)

            if depth_values:
                Z_real_tag = np.median(depth_values)  # Median depth value

            # Compute scale factor based on depth
            scale_factor = Z_real_tag / tvec[2]  # Use depth at tag's center
            tvec_real = tvec * scale_factor  # Apply the scaling factor to the translation vector


            if ret:
                # Project 3D cube points onto 2D image
                imgpts, _ = cv2.projectPoints(object_points, rvec, tvec_real, self.camera_matrix, self.dist_coeffs)
                imgpts = np.int32(imgpts).reshape(-1, 2)

                # Display axis of tag
                self.frame.axes[detection.tag_id] = imgpts

                rmatrix, _ = cv2.Rodrigues(rvec)

                # Store the pose of each detected tag
                tags.append((rvec, rmatrix, tvec_real))

        return tags
