import numpy as np
import math
import cv2

class Frame:
    def __init__(self):
        """
        Initialize the frame with default attributes for display, overlays,
        geometric annotations, and internal image buffers.
        """
        self.title = "RealSense Camera"  # Default window title
        self.text = ""  # List of text to display in the top-left corner
        self.text_mode = ""
        self.objects = {}  # Dictionary {label: (outline, color)}
        self.polys = {}
        self.points = {}  # Dictionary {label: (x, y, color)}
        self.axes = {} # Dictionary {label: (x, y)}
        self.box = {}
        

        # Attributes for frames
        self.color = None
        self.depth = None
        self.color_standard = None
        self.center_x = None
        self.center_y = None
        self.tag_points = None

    def display(self):
        """Display the frame with the title"""
        cv2.imshow(self.title, self.color)

    def close():
        """Close all displayed frames"""
        cv2.destroyAllWindows()

    def populate(self):
        """
        Draw all visual elements onto the frame including:
        contours, polylines, labeled points, 3D axes, status text, and tag coordinates.
        """
        # Draw all outlines with their colors
        if self.objects:
            for label, (object, color_code) in self.objects.items():
                cv2.drawContours(self.color, [np.array(object)], -1, color_code, 2)

        if self.polys:
            for label, (poly, color_code) in self.polys.items():
                cv2.polylines(self.color, [np.array(poly)], isClosed=False, color=color_code, thickness=2)
   
        # Draw all points with their colors and labels
        if self.points:
            for label, (x, y, size, color_code) in self.points.items():
                cv2.circle(self.color, (x, y), size, color_code, -1)  # Draw point
                cv2.putText(self.color, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw all axes
        if self.axes:
            for label, points in self.axes.items():
                # Red (X-axis): Bottom-front to bottom-right (front face)
                cv2.line(self.color, tuple(points[0]), tuple(points[1]), (0, 0, 255), 3)  # X-axis (red)
                # Green (Y-axis): Bottom-left to top-left (left face)
                cv2.line(self.color, tuple(points[0]), tuple(points[2]), (0, 255, 0), 3)  # Y-axis (green)
                # Blue (Z-axis): Bottom-left to top-center (backward direction)
                cv2.line(self.color, tuple(points[0]), tuple(points[3]), (255, 0, 0), 3)  # Z-axis (blue)

        # Draw text in the top-left corner
        if self.text:
            cv2.putText(self.color, self.text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        if self.text_mode:
            cv2.putText(self.color, self.text_mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        if self.box:
            for label, (x1, y1, x2, y2) in self.box.items():
                cv2.rectangle(self.color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # Draw text in the top-left corner
        if self.tag_points:
            height, width = self.color.shape[:2]
            font_scale = 0.5
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX

            num_tags = len(self.tag_points)

            for i, p in enumerate(self.tag_points):
                if p is None or any(v is None for v in p):
                    continue  # Skip this point completely!

                # Convert numpy ndarray to a list if needed and format the values as floats
                label = f"{'Lp' if i == 0 else 'Rp'}: {float(p[0]):.2f}, {float(p[1]):.2f}, {float(p[2]):.3f}"

                # Get text size to position properly
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                x = width - text_width - 10
                y = height - 10 - (num_tags - 1 - i) * (text_height + 10)
                cv2.putText(self.color, label, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def save_image(self, filename: str = "empty_workspace.png"):
        """
        Save the current frame to a file.

        :param filename: Name of the file to save the image
        """
        if self.color is not None:
            cv2.imwrite(filename, self.color_standard)
            print(f"Image saved as {filename}")
        else:
            print("No image to save.")
    
    def detect_largest_object(self, min_area: int = 2000):
        """
        Detect the largest changed object in the scene by subtracting
        the background image. Applies thresholding and contour filtering
        based on area and intensity difference.

        :param min_area: Minimum area in pixels to consider as valid object
        :return: The largest contour and binary mask, or (None, None) if not found
        """
        background_path = 'empty_workspace.png'
        background_img = cv2.imread(background_path)

        if background_img is None or self.color_standard is None:
            print("[Error] Missing background or frame.")
            return None, None

        # Convert to grayscale
        bg_gray = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(self.color_standard, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        bg_blur = cv2.GaussianBlur(bg_gray, (9, 9), 0)
        cur_blur = cv2.GaussianBlur(cur_gray, (9, 9), 0)

        # Difference image
        diff = cv2.absdiff(bg_blur, cur_blur)

        # Threshold
        _, thresh = cv2.threshold(diff, 70, 255, cv2.THRESH_BINARY)

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

            # Check intensity difference inside contour
            temp_mask = np.zeros_like(bg_gray)
            cv2.drawContours(temp_mask, [contour], -1, 255, -1)
            mean_bg = cv2.mean(bg_gray, mask=temp_mask)[0]
            mean_cur = cv2.mean(cur_gray, mask=temp_mask)[0]
            if abs(mean_bg - mean_cur) < 30:
                continue

            if area > max_area:
                largest_contour = contour
                max_area = area
        mask = np.zeros_like(clean)
        for cnt in contours:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

        if largest_contour is not None:
            return largest_contour, mask

        return None, None

    def get_left_right_point(self, percentage: int = 100):
        """
        Find and return the inner-left and inner-right points of the top part
        of the largest contour, based on a percentage of its height.

        :param percentage: How much of the contour height to consider (default 100%)
        :return: List of two inner points [left_point, right_point] or None if not found
        """
        detected_contour, _ = self.detect_largest_object()
        if detected_contour is None:
            return None, None

        # Compute bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(detected_contour)

        # Determine the height of the portion to check based on the percentage
        height_to_check = int(h * (percentage / 100))
        
        # Crop the contour to the top portion based on the given percentage
        cropped_contour = detected_contour[detected_contour[:, :, 1] <= (y + height_to_check)]

        if cropped_contour.size == 0:
            return None, None

        # Find extreme left and right points of the cropped contour
        left_point = tuple(cropped_contour[cropped_contour[:, 0].argmin()])
        right_point = tuple(cropped_contour[cropped_contour[:, 0].argmax()])

        # Compute centroid of the cropped contour
        M = cv2.moments(cropped_contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            # fallback if invalid
            centroid_x = (left_point[0] + right_point[0]) // 2
            centroid_y = (left_point[1] + right_point[1]) // 2

        # Calculate the vector from the points to the centroid
        delta_left = (centroid_x - left_point[0], centroid_y - left_point[1])
        delta_right = (centroid_x - right_point[0], centroid_y - right_point[1])

        # Move the points towards the centroid (e.g., 10% towards the centroid)
        left_point_inside = (int(left_point[0] + 0.1 * delta_left[0]), int(left_point[1] + 0.1 * delta_left[1]))
        right_point_inside = (int(right_point[0] + 0.1 * delta_right[0]), int(right_point[1] + 0.1 * delta_right[1]))

        self.points["Left Point"] = (left_point_inside[0], left_point_inside[1], 5, (0, 0, 255))
        self.points["Right Point"] = (right_point_inside[0], right_point_inside[1], 5, (255, 0, 0))

        return [left_point_inside, right_point_inside]
