from dual_arm_xs import InterbotixManipulatorXS
from scipy.spatial import Delaunay
from ultralytics import YOLO
from camera import Camera
from enum import Enum
import numpy as np
import threading
import math
import time
import cv2

# Define the UnfoldState enum class
class UnfoldState(Enum):
    GET_POINTS = 1
    PICK_UP = 2
    STRETCH = 3
    LAY_FLAT = 4
    DETECT = 5
    GET_POINTS_UPPER = 6
    DONE = 7

class Process():
    def __init__(self, 
        camera: Camera = None,
        bot_left: InterbotixManipulatorXS = None,
        bot_right: InterbotixManipulatorXS = None,
    ) -> None:

        self.state = UnfoldState.GET_POINTS
        self.previous_state = None
        self.flag_straight = False
        self.pixel_points = [[None, None], [None, None]]
        self.tag_points = [[None, None, None], [None, None, None]]

        self.barrier = threading.Barrier(2)
        self.pick_up_height = 0.25

        self.model = YOLO('/home/student/Documents/MAS500/best.pt')

        if camera is not None:
            self.camera = camera

        if bot_left is not None and bot_right is not None:
            self.bot_left, self.bot_right = bot_left, bot_right

        # Get 2D XY outline of max reach
        self.reach_left = self.bot_left.arm.max_reach_outline()
        self.reach_right = self.bot_right.arm.max_reach_outline()

        self.hull_left = Delaunay(np.array(self.reach_left))
        self.hull_right = Delaunay(np.array(self.reach_right))

        # Clear previous pixel points
        self.reach_pixels_left = []
        self.reach_pixels_right = []

        self.get_robot_reach()

        thread = threading.Thread(target=self.update_points, daemon=True)
        thread.start()

        thread.join(timeout=2)

    def get_robot_reach(self):
        # Number of points to generate around the circle
        num_points = 16
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        # Get reach radius for each arm
        radius_left = self.bot_left.arm.max_reach_radius()
        radius_right = self.bot_right.arm.max_reach_radius()

        # Generate circle points (z = 0)
        for angle in angles:
            # Create 3D point on the circle (z=0)
            point_left = [radius_left * np.cos(angle), radius_left * np.sin(angle), 0]
            point_right = [radius_right * np.cos(angle), radius_right * np.sin(angle), 0]

            # Convert to pixel coordinates using each bot's tag as the reference frame
            pixel_left = self.camera.coordsystem_to_pixel(self.bot_left.tag, point_left)
            if pixel_left is not None:
                self.reach_pixels_left.append(pixel_left)

            pixel_right = self.camera.coordsystem_to_pixel(self.bot_right.tag, point_right)
            if pixel_right is not None:
                self.reach_pixels_right.append(pixel_right)

    def update_points(self):
        while True:
            self.camera.frame.tag_points = self.tag_points
            # Format for cv2.drawContours: shape (n_points, 1, 2)
            reach_contour_left = np.array(self.reach_pixels_left, dtype=np.int32).reshape((-1, 1, 2))
            reach_contour_right = np.array(self.reach_pixels_right, dtype=np.int32).reshape((-1, 1, 2))

            # Assign to camera frame objects with custom colors
            self.camera.frame.objects["Reach Left"] = (reach_contour_left, (0, 0, 255))     # Red
            self.camera.frame.objects["Reach Right"] = (reach_contour_right, (255, 0, 0))   # Blue
            time.sleep(0.1)

    def set_bot_sleep(self, bot):
        bot.gripper.release()
        bot.arm.set_ee_pose_components(0.2, 0, 0.3)
        bot.gripper.grasp(0.1)
        bot.arm.go_to_sleep_pose()

    def unfold_detected(self, fold_point, unfold_point, bot):
        fx, fy, fz = fold_point

        fx = float(fx)
        fy = float(fy)
        fz = float(fz) + 0.01

        ux, uy, uz = unfold_point

        ux = float(ux)
        uy = float(uy)
        uz = float(uz)

        bot.gripper.release()

        bot.arm.set_ee_pose_components(0.2, 0, 0.3)

        bot.arm.set_ee_pose_components(fx, fy, 0.2, pitch=1)

        bot.arm.set_ee_pose_components(fx, fy, 0.05, pitch=1)

        bot.gripper.grasp(0.01)

        bot.arm.set_ee_pose_components(fx, fy, fz, pitch=1)
        time.sleep(1)

        bot.arm.set_ee_pose_components(fx, fy, uz + 0.1, pitch=1)

        bot.arm.set_ee_pose_components(ux, uy, uz + 0.1, pitch=1)

        self.set_bot_sleep(bot)

        self.camera.frame.points.pop("Edge", None)
        self.camera.frame.points.pop("Target", None)
        self.camera.frame.points.pop("Center", None)
        self.camera.frame.box.pop("Fold", None)

    def unfold(self):
        while self.state != UnfoldState.DONE:
            self.previous_state = self.state

            if self.state == UnfoldState.GET_POINTS:
                self.pixel_points = self.camera.frame.get_left_right_point()

                if self.pixel_points is None:
                    self.camera.frame.text = "[Error] No points detected, please try again"
                    self.state = UnfoldState.GET_POINTS
                else:
                    # Proceed only if pixel_points is valid
                    for id, bot in enumerate([self.bot_left, self.bot_right]):
                        if id < len(self.pixel_points):  # Ensure there's a valid point for each bot
                            self.tag_points[id] = self.camera.pixel_to_coordsystem(bot.tag, self.pixel_points[id], adjust_error=True)

                            if self.tag_points[id] is not None:
                                self.state = UnfoldState.PICK_UP
                            else:
                                self.camera.frame.text = f"[Error] Invalid point for bot {bot}"
                                self.state = UnfoldState.GET_POINTS

            elif self.state == UnfoldState.PICK_UP:
                threads = []
                for id, bot in enumerate([self.bot_left, self.bot_right]):
                    self.tag_points[id] = self.camera.pixel_to_coordsystem(bot.tag, self.pixel_points[id], adjust_error=True)
                    thread = threading.Thread(target=self.pick_up_object, args=(bot, id))
                    thread.start()
                    threads.append(thread)

                for t in threads:
                    t.join()
                self.state = UnfoldState.STRETCH

            elif self.state == UnfoldState.STRETCH:
                self.event_stretch = threading.Event()
                threads = []
                for id, bot in enumerate([self.bot_left, self.bot_right]):
                    thread = threading.Thread(target=self.stretch, args=(bot, id))
                    thread.start()
                    threads.append(thread)
                self.event_stretch.wait()
                self.detect_stretched()

                for t in threads:
                    t.join()

                self.flag_straight = False
                self.state = UnfoldState.LAY_FLAT

            elif self.state == UnfoldState.LAY_FLAT:
                threads = []
                for id, bot in enumerate([self.bot_left, self.bot_right]):
                    thread = threading.Thread(target=self.lay_flat_object, args=(bot, id))
                    thread.start()
                    threads.append(thread)
                for t in threads:
                    t.join()
                self.state = UnfoldState.DETECT

            elif self.state == UnfoldState.DETECT:
                fold_point, unfold_point, bot = self.detect_fold()

                if fold_point is None:
                    return UnfoldState.DONE
                
                self.unfold_detected(fold_point, unfold_point, bot)
                
                self.state = UnfoldState.GET_POINTS_UPPER

            elif self.state == UnfoldState.GET_POINTS_UPPER:
                self.pixel_points = self.camera.frame.get_left_right_point(percentage=20)

                if self.pixel_points is None:
                    self.camera.frame.text = "[Error] No points detected, please try again"
                    self.state = UnfoldState.GET_POINTS_UPPER
                else:
                    # Proceed only if pixel_points is valid
                    for id, bot in enumerate([self.bot_left, self.bot_right]):
                        if id < len(self.pixel_points):  # Ensure there's a valid point for each bot
                            self.tag_points[id] = self.camera.pixel_to_coordsystem(bot.tag, self.pixel_points[id], adjust_error=True)

                            if self.tag_points[id] is not None:
                                self.state = UnfoldState.PICK_UP
                            else:
                                self.camera.frame.text = f"[Error] Invalid point for bot {bot}"
                                self.state = UnfoldState.GET_POINTS_UPPER

            self.validate_points()

            # If manual mode is on, wait for input after each step
            if self.camera.manual_mode:
                self.state = self.await_manual_control()

    def detect_fold(self):
        """
        Detect fold by running inference and analyzing depth along the bisector
        of the two best Z-directions. Returns:
        - center coordinate in 3D
        - doubled target coordinate in 3D
        """
        color_image = self.camera.get_depth_map_object(self.bot_left, self.bot_right)
        results = self.model(color_image)
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None, None, None

        boxes = result.boxes
        confidences = boxes.conf
        xywh = boxes.xywh

        top_idx = int(confidences.argmax())
        center_x, center_y, w, h = map(int, xywh[top_idx])
        center = (center_x, center_y)

        # Select tag based on object position in image
        height, width = color_image.shape[:2]
        bot = self.bot_left if center_x < width // 2 else self.bot_right

        # Define directions
        directions = {
            "north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0),
            "northwest": (-1, -1), "northeast": (1, -1),
            "southwest": (-1, 1), "southeast": (1, 1)
        }

        step_size = 1
        direction_scores = {}

        # Get bounding box edges (+/- margin for safety)
        x_min = (center_x - w // 2) - 5
        x_max = (center_x + w // 2) + 5
        y_min = (center_y - h // 2) - 5
        y_max = (center_y + h // 2) + 5

        self.camera.frame.box["Fold"] = (x_min, y_min, x_max, y_max)

        # Compute average Z value per direction within bounding box
        for name, (dx, dy) in directions.items():
            px, py = center
            z_values = []

            while x_min <= int(px) <= x_max and y_min <= int(py) <= y_max:
                pixel_point = (int(px), int(py))
                tag_p = self.camera.pixel_to_coordsystem(bot.tag, pixel_point, adjust_error=True)

                if tag_p is not None and not np.isnan(tag_p[2]):
                    z_values.append(tag_p[2])

                px += dx * step_size
                py += dy * step_size

            direction_scores[name] = np.mean(z_values) if z_values else -np.inf

        # Sort directions by descending average Z
        sorted_dirs = sorted(direction_scores.items(), key=lambda x: x[1], reverse=True)
        if not sorted_dirs or sorted_dirs[0][1] == -np.inf:
            return None, None, None

        best_direction_name = sorted_dirs[0][0]

        # Determine target direction based on refinement rules
        target_direction = best_direction_name

        if best_direction_name == "north":
            east_score = direction_scores.get("east", -np.inf)
            west_score = direction_scores.get("west", -np.inf)
            target_direction = "northwest" if west_score > east_score else "northeast"

        elif best_direction_name == "south":
            east_score = direction_scores.get("east", -np.inf)
            west_score = direction_scores.get("west", -np.inf)
            target_direction = "southwest" if west_score > east_score else "southeast"

        elif best_direction_name == "east":
            north_score = direction_scores.get("north", -np.inf)
            south_score = direction_scores.get("south", -np.inf)
            target_direction = "northeast" if north_score > south_score else "southeast"

        elif best_direction_name == "west":
            north_score = direction_scores.get("north", -np.inf)
            south_score = direction_scores.get("south", -np.inf)
            target_direction = "northwest" if north_score > south_score else "southwest"

        elif best_direction_name == "northeast":
            northwest_score = direction_scores.get("northwest", -np.inf)
            southeast_score = direction_scores.get("southeast", -np.inf)
            target_direction = "north" if northwest_score > southeast_score else "east"

        elif best_direction_name == "southwest":
            southeast_score = direction_scores.get("southeast", -np.inf)
            northwest_score = direction_scores.get("northwest", -np.inf)
            target_direction = "south" if southeast_score > northwest_score else "west"

        elif best_direction_name == "southeast":
            southwest_score = direction_scores.get("southwest", -np.inf)
            northeast_score = direction_scores.get("northeast", -np.inf)
            target_direction = "south" if southwest_score > northeast_score else "east"

        # elif best_direction_name == "nortwesteast":
        #     north_score = direction_scores.get("north", -np.inf)
        #     southeast_score = direction_scores.get("southeast", -np.inf)
        #     target_direction = "north" if north_score > southeast_score else "east"

        # directions = {
        #     "north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0),
        #     "northwest": (-1, -1), "northeast": (1, -1),
        #     "southwest": (-1, 1), "southeast": (1, 1),
        #     "northwesteast": (-0.5, -1), "northwestwest": (-1.5, -1),
        #     "northeasteast": (1.5, -1), "northeastwest": (0.5, -1),
        #     "southeasteast": (1.5, 1), "southeastwest": (0.5, 1),
        #     "southwesteast": (-0.5, 1), "southwestwest": (-1.5, 1)
        # }

        # Get (dx, dy) of target direction
        target_dx, target_dy = directions[target_direction]

        # Get object contour
        contour, _ = self.camera.frame.detect_largest_object()
        if contour is None:
            print("test4")
            return None, None, None

        # Trace along selected vector to contour edge
        px, py = center
        edge_point = None
        max_steps = max(width, height)

        for _ in range(max_steps):
            px += target_dx * step_size
            py += target_dy * step_size
            ipx, ipy = int(round(px)), int(round(py))

            if not (0 <= ipx < width and 0 <= ipy < height):
                break

            dist = cv2.pointPolygonTest(contour, (ipx, ipy), False)
            if dist <= 0:
                edge_point = (ipx, ipy)
                break

        if edge_point is None:
            print("test5")
            return None, None, None

        # Compute doubled point
        cx, cy = center
        ex, ey = edge_point
        dx = ex - cx
        dy = ey - cy
        doubled_point = (int(cx + 2 * dx), int(cy + 2 * dy))

        # Visual debug points
        self.camera.frame.points.pop("Edge", None)
        self.camera.frame.points.pop("Target", None)
        self.camera.frame.points.pop("Center", None)
        self.camera.frame.points["Edge"] = (ex, ey, 5, (0, 255, 255))
        self.camera.frame.points["Target"] = (doubled_point[0], doubled_point[1], 5, (0, 255, 255))
        self.camera.frame.points["Center"] = (cx, cy, 5, (255, 0, 255))

        # Convert pixel to 3D coordinate system
        center_coord = self.camera.pixel_to_coordsystem(bot.tag, center, adjust_error=True)
        edge_coord = self.camera.pixel_to_coordsystem(bot.tag, edge_point, adjust_error=True)

        if center_coord is None or edge_coord is None:
            return None, None, None

        # Compute doubled 3D target coordinate
        target_coord = (edge_coord - center_coord) + edge_coord

        return center_coord, target_coord, bot


    def validate_points(self):
        if self.tag_points[0] is None or self.tag_points[1] is None:
            self.camera.frame.text = "[Error] Tag points not defined"
            self.state = UnfoldState.GET_POINTS
            return

        point_left = np.array(self.tag_points[0]).reshape(1, -1)
        point_right = np.array(self.tag_points[1]).reshape(1, -1)

        if self.hull_left.find_simplex(point_left) < 0:
            self.camera.frame.text = f"[Error] Point {self.tag_points[0]} is out of reach for left arm"
            self.state = UnfoldState.GET_POINTS
            return

        if self.hull_right.find_simplex(point_right) < 0:
            self.camera.frame.text = f"[Error] Point {self.tag_points[1]} is out of reach for right arm"
            self.state = UnfoldState.GET_POINTS
            return

    def await_manual_control(self):
        self.camera.frame.text = "q = back, w = next, e = manual pixel input, s = sleep, d = detect folds"
        while True:
            key = self.camera.key  # Assuming you are updating key somewhere else

            if key == ord('q'):
                self.camera.frame.text = "Going back to previous state"
                return self.previous_state
            
            elif key == ord('d'):
                self.camera.frame.text = "Detecting folds"
                return UnfoldState.DETECT

            elif key == ord('w'):
                self.camera.frame.text = "Continuing to next state"
                return self.state

            elif key == ord('e'):
                self.camera.frame.text = "Manual pixel input mode"
                # Trigger the click function from the Camera to get points manually
                self.camera.frame.text = "Click point for left arm"
                left = self.camera.wait_for_click()
                self.camera.frame.points["Left Point"] = (left[0], left[1], 5, (0, 0, 255))

                self.camera.frame.text = "Click point for right arm"
                right = self.camera.wait_for_click()
                self.camera.frame.points["Right Point"] = (right[0], right[1], 5, (255, 0, 0))

                # Correct order if needed
                if left[0] > right[0]:
                    left, right = right, left
                    self.camera.frame.points["Left Point"] = (left[0], left[1], 5, (0, 0, 255))
                    self.camera.frame.points["Right Point"] = (right[0], right[1], 5, (255, 0, 0))

                self.pixel_points = [left, right]
                self.camera.frame.text = "Points set, returning to pick up state"
                return UnfoldState.PICK_UP
            
            elif key == ord('s'):
                # Create threads for both bots
                bot_left_thread = threading.Thread(target=self.set_bot_sleep, args=(self.bot_left,))
                bot_right_thread = threading.Thread(target=self.set_bot_sleep, args=(self.bot_right,))

                # Start both threads
                bot_left_thread.start()
                bot_right_thread.start()

                # Wait for both threads to finish
                bot_left_thread.join()
                bot_right_thread.join()

                return UnfoldState.GET_POINTS

    def detect_stretched(self):
        while True:
            detected_contour, _ = self.camera.frame.detect_largest_object()
            if detected_contour is None or len(detected_contour) < 2:
                return None, None
            
            # Approximate the contour to reduce noise
            epsilon = 0.01 * cv2.arcLength(detected_contour, True)
            detected_contour = cv2.approxPolyDP(detected_contour, epsilon, True)

            # Reference points
            px1, py1 = self.pixel_points[0]
            px2, py2 = self.pixel_points[1]

            px1 += 75
            px2 -= 75

            # Find the closest point to left_point in the contour
            closest_left_index = np.argmin([np.linalg.norm(np.array((pt[0][0], pt[0][1])) - np.array((px1, py1))) for pt in detected_contour])
            closest_left_point = detected_contour[closest_left_index][0]
            x1, y1 = closest_left_point

            # Find the closest point to right_point in the contour
            closest_right_index = np.argmin([np.linalg.norm(np.array((pt[0][0], pt[0][1])) - np.array((px2, py2))) for pt in detected_contour])
            closest_right_point = detected_contour[closest_right_index][0]
            x2, y2 = closest_right_point

            self.camera.frame.points["Left Point"] = (x1, y1, 5, (0, 255, 0))
            self.camera.frame.points["Right Point"] = (x2, y2, 5, (255, 0, 0))

            # Ensure proper order (left -> right)
            if closest_left_index > closest_right_index:
                closest_left_index, closest_right_index = closest_right_index, closest_left_index

            # Extract the two possible segments
            segment1 = detected_contour[closest_left_index:closest_right_index + 1]
            segment2 = np.concatenate((detected_contour[closest_right_index:], detected_contour[:closest_left_index + 1]))

            # Choose the shorter segment
            if cv2.arcLength(segment1, False) < cv2.arcLength(segment2, False):
                chosen_segment = segment1
            else:
                chosen_segment = segment2

            # Visualize points and selected contour segment
            self.camera.frame.polys["Poly"] = (chosen_segment, (0, 255, 0))

            # Check if the segment is straight
            x1, y1 = closest_left_point
            x2, y2 = closest_right_point

            # Calculate the line equation: y = mx + c
            m = (y2 - y1) / (x2 - x1)  # Slope
            c = y1 - m * x1            # Intercept

            # Calculate the deviation for each point in the segment
            deviations = []
            for pt in chosen_segment:
                px, py = pt[0]
                # Distance from point (px, py) to the line y = mx + c
                distance = abs(m * px - py + c) / math.sqrt(m**2 + 1)
                deviations.append(distance)

            # Maximum deviation
            max_deviation = max(deviations)

            # Define a threshold for straightness
            threshold = 5.0  # Adjust as needed
            is_straight = max_deviation < threshold

            if is_straight or self.flag_straight:
                self.camera.frame.text = "Line is Straight"
                self.flag_straight = True
                self.camera.frame.polys["Poly"] = ([], (0, 0, 0))
                break
            else:
                self.camera.frame.text = "Line is Not Straight"

    def pick_up_object(self, bot, id, pitch=1.35):
        bot.arm.set_ee_pose_components(0.2, 0, 0.3)
        x, y, z = self.tag_points[id]

        x = float(x)
        y = float(y)
        z = float(z)
        
        bot.gripper.release()

        bot.arm.set_ee_pose_components(x, y, z + 0.15, pitch=pitch)

        bot.arm.set_ee_pose_components(x, y, z + 0.1, pitch=pitch)

        bot.arm.set_ee_pose_components(x, y, z + 0.05, pitch=pitch)
        self.barrier.wait()
        bot.gripper.grasp(0.01)

        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, y, z, pitch=pitch)

        time.sleep(1)

        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, y, z + 0.1, pitch=pitch)

        pitch = 1

        self.barrier.wait()

        if id == 0:
            y = -self.pick_up_height
        elif id == 1:
            y = self.pick_up_height

        bot.arm.set_ee_pose_components(x, y, self.pick_up_height, pitch=pitch)

        self.barrier.wait()
        
        self.tag_points[id] = (x, y, self.pick_up_height)

        self.barrier.wait()

        x1, _, _ = self.tag_points[0]
        x2, _, _ = self.tag_points[1]
        avg_x = max((x1 + x2) / 2.0, 0.1)

        self.barrier.wait()
        bot.arm.set_ee_pose_components(avg_x, y, self.pick_up_height, pitch=pitch)

        self.tag_points[id] = (avg_x, y, self.pick_up_height)

    def stretch(self, bot, id, stretch_rate=0.01):
        x, _, _ = self.tag_points[id]
        x = float(x)
        
        if id == 0:
            y = -self.pick_up_height
        elif id == 1:
            y = self.pick_up_height

        self.barrier.wait()

        self.event_stretch.set()
        while not self.flag_straight:
            x -= stretch_rate
            self.pixel_points[id] = self.camera.coordsystem_to_pixel(bot.tag, (x, y, self.pick_up_height))
            bot.arm.set_ee_pose_components(x, y, self.pick_up_height, pitch=1)

            if x < 0.15:
                break
        
        self.barrier.wait()
        self.flag_straight = True
        self.tag_points[id] = (x, y, self.pick_up_height)


    def lay_flat_object(self, bot, id, pitch=1):
        x, _, z = self.tag_points[id]
        x = float(x)
        z = float(z)

        if id == 0:
            y = -0.1
        elif id == 1:
            y = 0.1

        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, y, 0.2, pitch=pitch)

        if id == 0:
            y = 0.2
        elif id == 1:
            y = -0.2

        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, y, 0.2, pitch=pitch)

        if id == 0:
            y = 0.4
        elif id == 1:
            y = -0.4

        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, y, 0.2, pitch=pitch)

        if id == 0:
            y = 0.2
        elif id == 1:
            y = -0.2

        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, y, 0.2, pitch=pitch)

        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, 0, 0.2, pitch=pitch)

        if id == 0:
            y = -0.1
        elif id == 1:
            y = 0.1

        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, y, 0.1, pitch=pitch)

        if id == 0:
            y = -0.4
        elif id == 1:
            y = 0.4

        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, y, 0.1, pitch=pitch)
        
        self.set_bot_sleep(bot)
