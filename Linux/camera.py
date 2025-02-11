import cv2

def display_point_on_frame(color_image, left_point, right_point, left_point_m, right_point_m):
    """Displays the camera frame with a blue dot at the detected point, coordinates, depth, and height above the table."""

    # Reverse the transformation to get the original coordinates
    left_point_original = (left_point[0] + color_image.shape[1] // 2, left_point[1] + color_image.shape[0] // 2)
    right_point_original = (right_point[0] + color_image.shape[1] // 2, right_point[1] + color_image.shape[0] // 2)
    
    if 0 <= left_point_original[0] < color_image.shape[1] and 0 <= left_point_original[1] < color_image.shape[0]:
        cv2.circle(color_image, left_point_original, 5, (0, 0, 255), -1)

    if 0 <= right_point_original[0] < color_image.shape[1] and 0 <= right_point_original[1] < color_image.shape[0]:
        cv2.circle(color_image, right_point_original, 5, (0, 0, 255), -1)

    # Calculate the center of the image
    center_x = color_image.shape[1] // 2
    center_y = color_image.shape[0] // 2

    # Draw a gray circle at the center of the image
    cv2.circle(color_image, (center_x, center_y), 5, (128, 128, 128), -1)

    # Display the coordinates and depth values
    cv2.putText(color_image, f"Left X: {left_point_m[0]:.3f}, Left Y: {left_point_m[1]:.3f}, Left Z: {left_point_m[2]:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(color_image, f"Right X: {right_point_m[0]:.3f}, Right Y: {right_point_m[1]:.3f}, Right Z: {right_point_m[2]:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
