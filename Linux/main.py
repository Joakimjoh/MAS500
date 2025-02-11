from initialize import initialize_camera
from move_arm import start_arm
from camera_detection import detect_red_object, check_line_straight
import camera_detection
from data import get_depth_data, get_frames, get_xyz
from camera import display_point_on_frame
import pyrealsense2 as rs
import cv2
import threading
from move_arm import InterbotixManipulatorXS
import time
import rclpy

def get_x_coordinate(u, v, depth_frame, intrinsics_depth):
    """
    Convert pixel (u, v) in the depth image to real-world (X, Y, Z) coordinates.
    Returns the X coordinate in meters.
    """
    depth_value = depth_frame.get_distance(u, v)  # Get depth value in meters
    if depth_value == 0:
        return None  # No valid depth data

    # Convert (u, v, depth) to real-world coordinates
    point = rs.rs2_deproject_pixel_to_point(intrinsics_depth, [u, v], depth_value)
    x_m = point[0]  # X coordinate in meters
    return x_m

def move_arm_to_home_pose(bot, barrier, x, y, z):

    bot.gripper.release()

    bot.arm.set_ee_pose_components(x, y, z + 0.1, pitch=1)

    barrier.wait()

    bot.arm.set_ee_pose_components(x, y, z + 0.05, pitch=1)
    
    bot.gripper.grasp(0.1)

    bot.arm.set_ee_pose_components(x, y, z, pitch=1)

    barrier.wait()

    if y > 0:
        bot.arm.set_ee_pose_components(x, 0.25, 0.25, pitch=1)
    elif y < 0:
        bot.arm.set_ee_pose_components(x, -0.25, 0.25, pitch=1)

def move_arm(bot, x, y, z, pitch=1):
    bot.arm.set_ee_pose_components(x, y, z, pitch=pitch)


def start_arm():
    rclpy.init()

    # Create the Interbotix manipulators for the two robots
    bot1 = InterbotixManipulatorXS(
        robot_model='wx250s',
        group_name='arm',
        robot_name='arm1',
        gripper_name='gripper',
        node_name='node1'
    )
    bot2 = InterbotixManipulatorXS(
        robot_model='wx250s',
        group_name='arm',
        robot_name='arm2',
        gripper_name='gripper',
        node_name='node2'
    )

    return bot1, bot2

def lay_object(bot, x, y):
    bot.arm.set_ee_pose_components(x, 0, 0.1, pitch=1)

    if y > 0:
        bot.arm.set_ee_pose_components(x, 0.25, 0.1, pitch=1)
    else:
        bot.arm.set_ee_pose_components(x, -0.25, 0.1, pitch=1)

    bot.gripper.release()

    bot.arm.go_to_sleep_pose()

    
# Initialize camera
pipeline, align, intrinsics_depth = initialize_camera()

# Get depth data for table
depth_data = get_depth_data(pipeline, align)

while True:        
    # Get depth and color frame
    depth_frame, color_image = get_frames(pipeline, align)

    # Detect red objects and get their coordinates
    left_depth, right_depth, left_point, right_point, left_point_y, right_point_y, lrel_x, lrel_y, rrel_x, rrel_y = detect_red_object(color_image, depth_frame, pipeline, depth_data)

    closest_point_left = depth_data.get(left_point, None)
    closest_point_right = depth_data.get(right_point, None)

    closest_point_left_y = depth_data.get(left_point_y, None)
    closest_point_right_y = depth_data.get(right_point_y, None)

    if closest_point_left and closest_point_right and closest_point_left_y and closest_point_right_y and lrel_x and rrel_x != None:
        left_point_m = get_xyz(left_depth, closest_point_left, lrel_x)
        right_point_m = get_xyz(right_depth, closest_point_right, rrel_x)
        
        display_point_on_frame(color_image, left_point, right_point, left_point_m, right_point_m)

    if cv2.waitKey(1) & 0xFF == ord('f'):
            # Input XYZ positions for both arms
            x_l = 0.55 - left_point_m[0]
            y_l = left_point_m[1] - 0.77
            z_l = 0.015

            x_r = 0.50 - right_point_m[0]
            y_r = 0.80 - right_point_m[1]
            z_r = 0.015

            pitch = 1

            l = (x_l, y_l, z_l)
            r = (x_r, y_r, z_r)

            bot1, bot2 = start_arm()

            barrier = threading.Barrier(2)

            thread1 = threading.Thread(target=move_arm_to_home_pose, args=(bot1, barrier, x_l, y_l, z_l))
            thread2 = threading.Thread(target=move_arm_to_home_pose, args=(bot2, barrier, x_r, y_r, z_r))

            thread1.start()
            thread2.start()

            thread1.join()
            thread2.join()

            thread5 = threading.Thread(target=check_line_straight, args=(pipeline, align))
            thread5.daemon = True

            thread5.start()

            while True:

                depth_frame, color_image = get_frames(pipeline, align)

                x_l -= 0.005
                x_r -= 0.005

                left_point = (x_l, 0)
                right_point = (x_r, 0)

                thread3 = threading.Thread(target=move_arm, args=(bot1, x_l, -0.25, 0.25, pitch))
                thread4 = threading.Thread(target=move_arm, args=(bot2, x_r, 0.25, 0.25, pitch))


                thread3.start()
                thread4.start()

                thread3.join()
                thread4.join()

                # Capture key press
                key = cv2.waitKey(1) & 0xFF  # Ensures only the last 8 bits are considered

                if key == ord('f') or camera_detection.is_straight:
                    break

            thread6 = threading.Thread(target=lay_object, args=(bot1, x_l, y_l))
            thread7 = threading.Thread(target=lay_object, args=(bot2, x_r, y_r))


            thread6.start()
            thread7.start()

            thread6.join()
            thread7.join()

       
    # Show the frame with the red object marked, gray center dot, and other information
    cv2.imshow("Frame with Red Object", color_image)

    # Exit condition (press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting.")
        break

cv2.destroyAllWindows()