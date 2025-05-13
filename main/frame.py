import numpy as np
import cv2

class Frame:
    """Represents a frame with multiple outlines, points, title, and text."""
    def __init__(self):
        self.title = "RealSense Camera"  # Default window title
        self.text = ""  # List of text to display in the top-left corner
        self.objects = {}  # Dictionary {label: (outline, color)}
        self.points = {}  # Dictionary {label: (x, y, color)}
        self.axes = {} # Dictionary {label: (x, y)}
        

        # Attributes for frames
        self.color = None
        self.depth = None
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
        # Draw all outlines with their colors
        if self.objects:
            for label, (object, color_code) in self.objects.items():
                cv2.drawContours(self.color, [np.array(object)], -1, color_code, 2)
   
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
                cv2.line(self.color, tuple(points[3]), tuple(points[2]), (0, 0, 255), 3)  # X-axis (red)
                # Green (Y-axis): Bottom-left to top-left (left face)
                cv2.line(self.color, tuple(points[3]), tuple(points[0]), (0, 255, 0), 3)  # Y-axis (green)
                # Blue (Z-axis): Bottom-left to top-center (backward direction)
                cv2.line(self.color, tuple(points[3]), tuple(points[4]), (255, 0, 0), 3)  # Z-axis (blue)

        # Draw text in the top-left corner
        if self.text:
            cv2.putText(self.color, self.text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
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
    
    def detect_red_objects(self):
            """Detect red objects and get adjusted points inside the contour."""
            # Convert image to HSV
            hsv = cv2.cvtColor(self.color, cv2.COLOR_BGR2HSV)

            # Define range for red color
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

            # Combine masks
            mask = mask1 | mask2

            # Reduce noise with blur and morphological operations
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            return contours, mask
    
    def get_largest_contour(self, contours, min_contour=5000):
        # Filter by size
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = [c for c in contours if cv2.contourArea(c) > min_contour]

        if contours:
            # Get the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Approximate the contour to reduce noise
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)    

            return largest_contour
        
        return None

    def get_left_right_point(self, percentage: int = 100):
        contours, _ = self.detect_red_objects()

        if not contours:
            return None, None
        
        largest_contour = self.get_largest_contour(contours)

        if largest_contour is None:
            return None, None

        # Compute bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Determine the height of the portion to check based on the percentage
        height_to_check = int(h * (percentage / 100))
        
        # Crop the contour to the top portion based on the given percentage
        cropped_contour = largest_contour[largest_contour[:, :, 1] <= (y + height_to_check)]

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

