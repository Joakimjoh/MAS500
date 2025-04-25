from camera import Camera
from dual_arm_xs import InterbotixManipulatorXS
import cv2
import numpy as np
import threading
import math
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
        self.pixel_points = {}
        self.tag_points = {}

        self.barrier = threading.Barrier(2)
        self.pick_up_height = 0.25

        if camera is not None:
            self.camera = camera

        if bot_left is not None and bot_right is not None:
            self.bot_left, self.bot_right = bot_left, bot_right

    def unfold(self):
        while self.state != UnfoldState.DONE:
            self.previous_state = self.state

            if self.state == UnfoldState.GET_POINTS:
                self.pixel_points = self.camera.frame.get_left_right_point()
                if self.pixel_points is not None:
                    self.state = UnfoldState.PICK_UP

            elif self.state == UnfoldState.PICK_UP:
                threads = []
                for id, bot in enumerate([self.bot_left, self.bot_right]):
                    self.tag_points[id] = self.camera.pixel_to_coordsystem(
                        bot.tag.orientation, self.pixel_points[id]
                    )
                    thread = threading.Thread(target=self.pick_up_object, args=(bot, id))
                    thread.start()
                    threads.append(thread)

                for t in threads:
                    t.join()
                self.state = UnfoldState.STRETCH

            elif self.state == UnfoldState.STRETCH:
                threads = []
                for id, bot in enumerate([self.bot_left, self.bot_right]):
                    thread = threading.Thread(target=self.stretch, args=(bot, id))
                    thread.start()
                    threads.append(thread)
                
                self.detect_stretched()

                for t in threads:
                    t.join()
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
                # Run AI model to detect if the object is flat
                # If flat, unfoldstate.DONE
                # If not flat, unfoldstate.GET_POINTS_UPPER
                self.state = UnfoldState.GET_POINTS_UPPER

            elif self.state == UnfoldState.GET_POINTS_UPPER:
                self.pixel_points = self.camera.frame.get_left_right_point(20)
                if self.pixel_points is not None:
                    self.state = UnfoldState.PICK_UP

            # If manual mode is on, wait for input after each step
            if self.camera.manual_mode:
                self.state = self.await_manual_control()

    def await_manual_control(self):
        print("[Manual] Waiting for input (q = back, w = next, e = manual pixel input then pick up)...")
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
                self.pixel_points[0] = self.camera.wait_for_click()  # Assuming wait_for_click waits for a click and returns the coordinates

                print("[Manual] Click point for right arm")
                self.pixel_points[1] = self.camera.wait_for_click()  # Get second point from user
                print("[Manual] Custom pixel points set, returning to PICK_UP.")
                return UnfoldState.PICK_UP

    def detect_stretched(self):
        while True:
            contours = self.camera.frame.detect_red_objects()

            if contours:
                largest_contour = self.camera.frame.get_largest_contour(contours)
                
                if largest_contour is not None:
                    # Find the nearest points on the contour to self.point_left and self.point_right
                    left_index = np.argmin(
                        [np.linalg.norm(np.array((pt[0][0], pt[0][1])) - np.array(self.pixel_points[0])) for pt in largest_contour]
                    )
                    right_index = np.argmin(
                        [np.linalg.norm(np.array((pt[0][0], pt[0][1])) - np.array(self.pixel_points[1])) for pt in largest_contour]
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

                    # Check if the segment is straight
                    x1, y1 = self.pixel_points[0]
                    x2, y2 = self.pixel_points[1]

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
                        threshold = 5.0  # Adjust as needed
                        is_straight = max_deviation < threshold

                        # Display the result
                        if is_straight:
                            self.camera.frame.text = "Line is Straight"
                            self.flag_straight = True
                            break
                        else:
                            self.camera.frame.text = "Line is Not Straight"

    def stretch(self, bot, id, stretch_rate=0.005):
        x, _, _ = self.tag_points[id]
        if id == 0:
            y = -self.pick_up_height
        elif id == 1:
            y = self.pick_up_height

        while not self.flag_straight:
            x += stretch_rate
            bot.arm.set_ee_pose_components(x, y, self.pick_up_height, pitch=1)
            self.barrier.wait()

            self.pixel_points[id] = self.camera.coordsystem_to_pixel(bot.tag.orientation, (x, y, self.pick_up_height))

        self.tag_points[id] = (x, y, self.pick_up_height)
        self.barrier.wait()
        self.lay_flat_object(bot, x)

    def pick_up_object(self, bot, id, pitch=1):
        x, y, z = self.pixel_points[id]

        bot.gripper.release()

        bot.arm.set_ee_pose_components(x, y, z + 0.1, pitch=pitch)

        bot.arm.set_ee_pose_components(x, y, z + 0.05, pitch=pitch)

        bot.gripper.grasp(0.1)

        bot.arm.set_ee_pose_components(x, y, z, pitch=pitch)

        self.barrier.wait()

        if id == 0:
            y = -self.pick_up_height
        elif id == 1:
            y = self.pick_up_height

        bot.arm.set_ee_pose_components(x, y, self.pick_up_height, pitch=pitch)

        self.barrier.wait()

        self.tag_points[id] = (x, y, self.pick_up_height)

    def lay_flat_object(self, bot, id, pitch=1):
        x, _, _ = self.tag_points[id]
        self.barrier.wait()
        bot.arm.set_ee_pose_components(x, 0, 0.1, pitch=pitch)
        self.barrier.wait()

        if id == 0:
            bot.arm.set_ee_pose_components(x, -self.pick_up_height, 0.1, pitch=pitch)
        elif id == 1:
            bot.arm.set_ee_pose_components(x, self.pick_up_height, 0.1, pitch=pitch)
        
        bot.gripper.release()
        bot.arm.go_to_sleep_pose()
