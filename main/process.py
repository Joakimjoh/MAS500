from camera import Camera
from dual_arm_xs import InterbotixManipulatorXS
import cv2
import numpy as np
import threading
import math
import time
from enum import Enum

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

        if camera is not None:
            self.camera = camera

        if bot_left is not None and bot_right is not None:
            self.bot_left, self.bot_right = bot_left, bot_right

        thread = threading.Thread(target=self.update_points, daemon=True)
        thread.start()

        thread.join(timeout=2)

    def update_points(self):
        while True:
            self.camera.frame.tag_points = self.tag_points
            time.sleep(0.1)

    def set_bot_sleep(self, bot):
        bot.gripper.release()
        bot.arm.go_to_sleep_pose()

    def unfold(self):
        while self.state != UnfoldState.DONE:
            self.previous_state = self.state

            if self.state == UnfoldState.GET_POINTS:
                self.pixel_points = self.camera.frame.get_left_right_point()

                if self.pixel_points is None:
                    print("[Error] No points detected, please try again.")
                    self.state = UnfoldState.GET_POINTS
                else:
                    # Proceed only if pixel_points is valid
                    for id, bot in enumerate([self.bot_left, self.bot_right]):
                        if id < len(self.pixel_points):  # Ensure there's a valid point for each bot
                            point = self.camera.pixel_to_coordsystem(
                                bot.tag.orientation, self.pixel_points[id]
                            )

                            if point is not None:
                                self.tag_points[id] = bot.tag.adjust_error(point)
                                self.state = UnfoldState.PICK_UP
                            else:
                                print(f"[Error] Invalid point for bot {bot}.")
                                self.state = UnfoldState.GET_POINTS

            elif self.state == UnfoldState.PICK_UP:
                threads = []
                for id, bot in enumerate([self.bot_left, self.bot_right]):
                    point = self.camera.pixel_to_coordsystem(
                        bot.tag.orientation, self.pixel_points[id]
                    )
                    self.tag_points[id] = bot.tag.adjust_error(point)
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
                print("test")
                for t in threads:
                    t.join()
                self.state = UnfoldState.GET_POINTS_UPPER

            # elif self.state == UnfoldState.DETECT:
            #     # Run AI model to detect if the object is flat
            #     # If flat, unfoldstate.DONE
            #     # If not flat, unfoldstate.GET_POINTS_UPPER
            #     self.state = UnfoldState.GET_POINTS_UPPER

            elif self.state == UnfoldState.GET_POINTS_UPPER:
                self.pixel_points = self.camera.frame.get_left_right_point(percentage=20)

                if self.pixel_points is None:
                    print("[Error] No points detected, please try again.")
                    self.state = UnfoldState.GET_POINTS_UPPER
                else:
                    # Proceed only if pixel_points is valid
                    for id, bot in enumerate([self.bot_left, self.bot_right]):
                        if id < len(self.pixel_points):  # Ensure there's a valid point for each bot
                            point = self.camera.pixel_to_coordsystem(
                                bot.tag.orientation, self.pixel_points[id]
                            )

                            if point is not None:
                                self.tag_points[id] = bot.tag.adjust_error(point)
                                self.state = UnfoldState.PICK_UP
                            else:
                                print(f"[Error] Invalid point for bot {bot}.")
                                self.state = UnfoldState.GET_POINTS_UPPER

            print(self.state)
            # If manual mode is on, wait for input after each step
            if self.camera.manual_mode:
                self.state = self.await_manual_control()

    def await_manual_control(self):
        print("[Manual] Waiting for input (q = back, w = next, e = manual pixel input then pick up, s = sleep)")
        while True:
            key = self.camera.key  # Assuming you are updating key somewhere else

            if key == ord('q'):
                print("[Manual] Going back to previous state.")
                return self.previous_state

            elif key == ord('w'):
                print("[Manual] Continuing to next state.")
                return self.state

            elif key == ord('e'):
                print("[Manual] Manual pixel input mode.")
                # Trigger the click function from the Camera to get points manually
                print("[Manual] Click point for left arm")
                left = self.camera.wait_for_click()
                self.camera.frame.points["Left Point"] = (left[0], left[1], "red")

                print("[Manual] Click point for right arm")
                right = self.camera.wait_for_click()
                self.camera.frame.points["Right Point"] = (right[0], right[1], "blue")

                # Correct order if needed
                if left[0] > right[0]:
                    print("[Manual] Points were reversed, swapping them.")
                    left, right = right, left
                    self.camera.frame.points["Left Point"] = (left[0], left[1], "red")
                    self.camera.frame.points["Right Point"] = (right[0], right[1], "blue")

                self.pixel_points = [left, right]
                print("[Manual] Custom pixel points set, returning to PICK_UP.")
                return UnfoldState.PICK_UP
            
            elif key == ord('s'):
                print("Set robots to sleep")
                
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
        prev_pixel_points = [[0, 0], [0, 0]]
        while True:
            contours = self.camera.frame.detect_red_objects()

            if contours:
                largest_contour = self.camera.frame.get_largest_contour(contours)
                
                if largest_contour is not None:
                    # Find the nearest points on the contour to the shifted points
                    diff_x1 = int((self.pixel_points[0][0] - prev_pixel_points[0][0]) * 1.5)
                    diff_x2 = int((self.pixel_points[1][0] - prev_pixel_points[1][0]) * 0.5)
                    x1, y1 = (diff_x1 + self.pixel_points[0][0]) + 90, self.pixel_points[0][1] + 100
                    x2, y2 = (diff_x2 + self.pixel_points[1][0]) - 70, self.pixel_points[1][1] + 100
                    
                    prev_pixel_points = self.pixel_points.copy()
                    left_index = np.argmin(
                        [np.linalg.norm(np.array((pt[0][0], pt[0][1])) - np.array((x1, y1))) for pt in largest_contour]
                    )
                    right_index = np.argmin(
                        [np.linalg.norm(np.array((pt[0][0], pt[0][1])) - np.array((x2, y2))) for pt in largest_contour]
                    )

                    # Ensure proper order (left -> right)
                    if left_index > right_index:
                        left_index, right_index = right_index, left_index

                    # Extract the two possible segments
                    segment1 = largest_contour[left_index:right_index + 1]
                    segment2 = np.concatenate((largest_contour[right_index:], largest_contour[:left_index + 1]))

                    # Choose the shorter segment
                    if cv2.arcLength(segment1, False) < cv2.arcLength(segment2, False):
                        chosen_segment = segment1
                    else:
                        chosen_segment = segment2

                    # Draw the chosen segment for visualization
                    self.camera.frame.objects["Object"] = (chosen_segment, "cyan")

                    self.camera.frame.points["Left Point"] = (x1, y1, "red")
                    self.camera.frame.points["Right Point"] = (x2, y2, "blue")

                    # Calculate the line equation: y = mx + c
                    if x2 != x1:
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
                        threshold = 25.0  # Adjust as needed
                        is_straight = max_deviation < threshold

                        # Display the result
                        if is_straight or self.flag_straight:
                            self.camera.frame.text = "Line is Straight"
                            self.flag_straight = True
                            self.camera.frame.objects["Object"] = ([], "cyan")
                            break
                        else:
                            self.camera.frame.text = "Line is Not Straight"

    def pick_up_object(self, bot, id, pitch=1.35):
        x, y, z = self.tag_points[id]

        x = float(x)
        y = float(y)
        z = float(z)
        
        bot.gripper.release()

        bot.arm.set_ee_pose_components(x, 0, z + 0.15, pitch=pitch)

        bot.arm.set_ee_pose_components(x, y, z + 0.15, pitch=pitch)

        bot.arm.set_ee_pose_components(x, y, z + 0.1, pitch=pitch)

        bot.arm.set_ee_pose_components(x, y, z + 0.05, pitch=pitch)
        self.barrier.wait()
        bot.gripper.grasp(0.01)
        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, y, z, pitch=pitch)
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
            self.pixel_points[id] = self.camera.coordsystem_to_pixel(bot.tag.orientation, (x, y, self.pick_up_height))
            bot.arm.set_ee_pose_components(x, y, self.pick_up_height, pitch=1)

            if (id == 0 and self.pixel_points[id][0] < -100) or (id == 1 and self.pixel_points[id][0] > 650):
                self.flag_straight = True
                break

        self.barrier.wait()
        self.tag_points[id] = (x, y, self.pick_up_height)

    def lay_flat_object(self, bot, id, pitch=1):
        x, _, z = self.tag_points[id]
        x = float(x)
        z = float(z)
        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, 0, z, pitch=pitch)
        if id == 0:
            y = 0.3
        elif id == 1:
            y = -0.3
        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, y, z, pitch=pitch)

        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, y, 0.1, pitch=pitch)

        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, 0, 0.1, pitch=pitch)

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
