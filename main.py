from initialize import initialize_camera
from camera_detection import detect_red_object
from data import get_depth_data, get_frames, get_xyz
from camera import display_point_on_frame
import pyrealsense2 as rs
import cv2

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
    else:
        # Even if no red object is detected, still display the frame
        cv2.imshow("Frame with Red Object", color_image)

    # Exit condition (press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting.")
        break

cv2.destroyAllWindows()