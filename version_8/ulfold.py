import cv2
import numpy as np
import math
import threading

pick_up_event = threading.Event()

pixel_points = {}

flag_corner, flag_straight, flag3 = False

def start(left_bot, right_bot):
    pixel_points = get_left_right_point()

    if pixel_points is not None:
        threads = []
        barrier = threading.Barrier(2)
        for i, bot in enumerate([left_bot, right_bot]):
            thread = threading.Thread(target=stretch, args=(bot, pixel_points[i], barrier))
            thread.start()
            threads.append(thread)

        pick_up_event.wait()

        # thread_detect = threading.Thread(target=self.detect_stretched)
        # thread_detect.start()

        # for thread in threads:
        #     thread.join()

def get_largest_contour(contours, min_contour=5000):
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

def get_left_right_point(frame):
    contours = frame.detect_red_objects()

    if not contours:
        return None, None
    
    largest_contour = get_largest_contour(contours)

    if largest_contour is None:
        return None, None
    
    # Add contour to frame
    frame.objects["Object1"] = (largest_contour, "blue")

    # Compute centroid of the contour
    M = cv2.moments(largest_contour)
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])

    # Find extreme left and right points
    left_point = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    right_point = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])

    # Calculate the vector from the points to the centroid
    delta_left = (centroid_x - left_point[0], centroid_y - left_point[1])
    delta_right = (centroid_x - right_point[0], centroid_y - right_point[1])

    # Move the points 10% towards the centroid
    left_point_inside = (int(left_point[0] + 0.1 * delta_left[0]), int(left_point[1] + 0.1 * delta_left[1]))
    right_point_inside = (int(right_point[0] + 0.1 * delta_right[0]), int(right_point[1] + 0.1 * delta_right[1]))
            
    # Add points to frame
    frame.points["Point1"] = (left_point_inside[0], left_point_inside[1], "green")
    frame.points["Point2"] = (right_point_inside[0], right_point_inside[1], "red")

    return left_point_inside, right_point_inside

def detect_stretched(frame, point_left, point_right):
    while True:
        contours = frame.detect_red_objects()

        if contours:
            largest_contour = get_largest_contour(contours)
            
            if largest_contour is not None:
                frame.objects["Object1"] = (largest_contour, "blue")

                # Find the nearest points on the contour to self.point_left and self.point_right
                left_index = np.argmin(
                    [np.linalg.norm(np.array((pt[0][0], pt[0][1])) - np.array(point_left)) for pt in largest_contour]
                )
                right_index = np.argmin(
                    [np.linalg.norm(np.array((pt[0][0], pt[0][1])) - np.array(point_right)) for pt in largest_contour]
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
                frame.objects["Object2"] = (chosen_segment, "cyan")

                # Check if the segment is straight
                x1, y1 = point_left
                x2, y2 = point_right

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
                        frame.text = "Line is Straight"
                        flag_straight = True
                        break
                    else:
                        frame.text = "Line is Not Straight"

def test(bot, point):
    point = bot.tag.camera.pixel_to_coordsystem(bot.tag.orientation, pixel_points[id])
    x, y, z = bot.tag.adjust_error(point)

    stretch(bot)

    print(x, y, z)

def stretch(bot, point, barrier, stretch_rate=0.005):
    x, y, z = point

    pick_up_object(bot, barrier, point)

    pick_up_event.set()

    while True:
        stretch_state = detect_stretched()
        x += stretch_rate
        bot.arm.set_ee_pose_components(x, y, z, pitch=1)
        barrier.wait()

        point = bot.tag.camera.coordsystem_to_pixel(bot.tag.orientation, point)

    lay_flat_object(bot, x, y)

def pick_up_object(bot, barrier, x, y, z, pitch=1):
    bot.gripper.release()

    bot.arm.set_ee_pose_components(x, y, z + 0.1, pitch=pitch)

    barrier.wait()

    bot.arm.set_ee_pose_components(x, y, z + 0.05, pitch=pitch)

    bot.gripper.grasp(0.1)

    bot.arm.set_ee_pose_components(x, y, z, pitch=pitch)

    barrier.wait()

    bot.arm.set_ee_pose_components(x, 0.25, 0.25, pitch=pitch)

def lay_flat_object(bot, x, y, pitch=1):
    bot.arm.set_ee_pose_components(x, 0, 0.1, pitch=pitch)

    if y > 0:
        bot.arm.set_ee_pose_components(x, 0.25, 0.1, pitch=pitch)
    else:
        bot.arm.set_ee_pose_components(x, -0.25, 0.1, pitch=pitch)

    bot.gripper.release()
    bot.arm.go_to_sleep_pose()
