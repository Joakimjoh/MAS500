from initialize import initialize_camera, initialize_robots, initialize_depth_data
from camera_detection import detect_red_object, detect_stretched
from camera import display_point_on_frame, get_frames
from data import get_coordinates_meter
from arm_control import step1
import threading
import cv2

# Initialize camera
pipeline, align, profile = initialize_camera()

# Initialize depth data
depth_data = initialize_depth_data(pipeline, align)

while True:
    # Get depth and color frame
    depth_frame, color_image = get_frames(pipeline, align)

    # Detect red objects and get their coordinates
    left_depth, right_depth, left_point, right_point, lrel_x, lrel_y, rrel_x, rrel_y = detect_red_object(pipeline, align, profile, depth_data)

    closest_point_left = depth_data.get(left_point, None)
    closest_point_right = depth_data.get(right_point, None)

    if closest_point_left and closest_point_right and lrel_x and rrel_x != None:
        left_point_m = get_coordinates_meter(left_depth, closest_point_left, lrel_x)
        right_point_m = get_coordinates_meter(right_depth, closest_point_right, rrel_x)
        
        display_point_on_frame(color_image, left_point, right_point, left_point_m, right_point_m)

    # Step 1 grap points and lay object flat
    if cv2.waitKey(1) & 0xFF == ord('f'):
            # Input XYZ positions for both arms
            x_l = 0.55 - left_point_m[0]
            y_l = left_point_m[1] - 0.77
            z_l = 0.015

            x_r = 0.50 - right_point_m[0]
            y_r = 0.80 - right_point_m[1]
            z_r = 0.015

            bot1, bot2 = initialize_robots()

            thread5 = threading.Thread(target=detect_stretched, args=(pipeline, align))
            thread5.daemon = True

            thread5.start()

            thread1 = threading.Thread(target=step1, args=(bot1, x_l, y_l, z_l))
            thread2 = threading.Thread(target=step1, args=(bot2, x_r, y_r, z_r))

            thread1.start()
            thread2.start()

            thread1.join()
            thread2.join()
  
    # Show the frame with the red object marked, gray center dot, and other information
    cv2.imshow("Frame with Red Object", color_image)

    # Exit condition (press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting.")
        break

cv2.destroyAllWindows()