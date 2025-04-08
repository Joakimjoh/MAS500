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
        """Convert pixel coordinates from the camera into the coordinate system of one of the robot arms."""
        # Camera intrinsic parameters
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Handle the case where point_pixel contains 2 or 3 values
        if len(point_pixel) == 2:  # Only x, y provided
            x, y = point_pixel
            depth_values = []
            for _ in range(100):
                depth_value = self.frame.depth.get_distance(x, y) - 0.01
                if depth_value > 0:
                    depth_values.append(depth_value)
            
            # Calculate the median after collecting all samples
            depth_value = np.median(depth_values) if depth_values else 0

            if depth_values:
                z = np.median(depth_values)  # Median depth value
        elif len(point_pixel) == 3:  # x, y, z provided
            x, y, z = point_pixel
        else:
            raise ValueError("Invalid point_pixel, must contain either 2 (x, y) or 3 (x, y, z) values")

        # Camera intrinsic parameters
        fx, fy = self.color_intrinsics.fx, self.color_intrinsics.fy
        cx, cy = self.color_intrinsics.ppx, self.color_intrinsics.ppy

        # Convert pixel coordinates to camera coordinates
        X_camera = (x - cx) * z / fx
        Y_camera = (y - cy) * z / fy
        Z_camera = z  # Already given or retrieved

        point_camera = np.array([[X_camera], [Y_camera], [Z_camera]])

        # Transform camera coordinates to AprilTag coordinates
        point_tag = np.dot(rmat.T, (point_camera - tvec))

        return point_tag
    
    def coordsystem_to_pixel(self, rvec, tvec, point_robot):
        """Convert a point from the robot arm's coordinate system to pixel coordinates in the camera's image"""
        
        # Convert rotation vector to rotation matrix (robot to camera)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Translate the point from robot to camera coordinates
        point_in_camera = np.dot(rotation_matrix, point_robot) + tvec
        
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
    
    def get_tag_orientation(self):
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

        while True:  # Keep looping until at least two tags are detected
            if self.frame.color is not None:
                gray_image = cv2.cvtColor(self.frame.color, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(gray_image)

                if len(detections) >= 2:  # Only return when 2 or more are found
                    break

        tags = []

        for detection in detections:
            # Get AprilTag corner positions
            corners = np.array(detection.corners, dtype=np.float32)

            # Compute tag center
            tag_cx, tag_cy = np.array(detection.center, dtype=np.int32)

            # Initial pose estimation using SOLVEPNP_ITERATIVE
            ret, rvec, tvec = cv2.solvePnP(
                object_points[:4], corners, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not ret:
                continue  # Skip this tag if pose estimation fails

            # Refine pose estimation
            rvec, tvec = cv2.solvePnPRefineLM(object_points[:4], corners, self.camera_matrix, self.dist_coeffs, rvec, tvec)

            # Calculate real-world depth from the tag center (use robust outlier rejection)
            depth_values = []
            for dx in range(-3, 4):  # Use a 7x7 neighborhood around the center
                for dy in range(-3, 4):
                    depth = self.frame.depth.get_distance(tag_cx + dx, tag_cy + dy) - 0.01
                    if depth > 0:
                        depth_values.append(depth)

            if depth_values:
                # Reject outliers using interquartile range (IQR)
                depth_values = np.array(depth_values)
                q1, q3 = np.percentile(depth_values, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_depth_values = depth_values[(depth_values >= lower_bound) & (depth_values <= upper_bound)]

                if len(filtered_depth_values) > 0:
                    Z_real_tag = np.mean(filtered_depth_values)  # Use mean of filtered values
                else:
                    continue  # Skip this tag if no valid depth values remain
            else:
                continue  # Skip this tag if depth estimation fails

            # Compute scale factor based on depth
            scale_factor = Z_real_tag / tvec[2]  # Use depth at tag's center
            tvec_real = tvec * scale_factor  # Apply the scaling factor to the translation vector

            # Validate orientation (ensure it is not overly tilted)
            rmat, _ = cv2.Rodrigues(rvec)
            z_axis = rmat[:, 2]  # Extract the Z-axis of the rotation matrix
            if abs(z_axis[2]) < 0.5:  # If the Z-axis is too tilted, skip this tag
                continue

            # Debugging: Visualize the tag axes
            imgpts, _ = cv2.projectPoints(object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            # Display axis of tag
            self.frame.axes[detection.tag_id] = imgpts

            # Store the pose of each detected tag
            tags.append((rvec, tvec_real))

        return tags
