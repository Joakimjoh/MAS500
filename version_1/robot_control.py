import parameters as parameters
from interbotix_xs_modules.arm import InterbotixArm

def robot_pathing(positions):
    print("Making path for {position}")

def robot_control(positions):
    robot_pathing(positions)

    # Move robot arms to pos and pickup

    print("Moving robot arms to {position}")

def pickup_stretch(left_pos, right_pos):
    # Pick up by each corner

    while True:
        # Stretch out each corner till line between is straight
        
        # Check if line is straight
        if get_straightness():
            print("Line Straight")
            break
        else:
            print("Line not straight") 
            # Move robots back a distance
    
    # Lay item flat


def move_to_xyz_with_offset(camera_xyz, robot_model="wx250s", offset=(0.3, -0.2, 0.0), gripper_open_width=0.05, gripper_close_width=0.01):
    """
    Moves the Interbotix WidowX-250 to a position adjusted by an offset and operates the gripper.

    Parameters:
        robot_model (str): The model of the robot arm (e.g., "wx250s").
        camera_xyz (tuple): The XYZ coordinates detected by the camera in meters (x, y, z).
        offset (tuple): The offset to apply to the camera coordinates (x_offset, y_offset, z_offset) in meters.
        gripper_open_width (float): The width to open the gripper (in meters).
        gripper_close_width (float): The width to close the gripper (in meters).

    Returns:
        None
    """
    # Calculate the adjusted target position
    adjusted_xyz = (
        camera_xyz[0] + offset[0],  # Adjust X by the offset
        camera_xyz[1] + offset[1],  # Adjust Y by the offset
        camera_xyz[2] + offset[2],  # Adjust Z by the offset
    )

    # Initialize the robot arm
    bot = InterbotixArm(robot_model=robot_model, robot_name="arm")

    try:
        # Open the gripper
        print("Opening the gripper...")
        bot.gripper.open(gripper_open_width)
        
        # Move the arm to the adjusted XYZ position
        print(f"Moving the arm to adjusted XYZ: {adjusted_xyz}...")
        bot.arm.set_ee_pose_components(x=adjusted_xyz[0], y=adjusted_xyz[1], z=adjusted_xyz[2])

        # Close the gripper
        print("Closing the gripper...")
        bot.gripper.close(gripper_close_width)
    finally:
        # Safely shutdown the robot
        print("Shutting down the robot...")
        bot.shutdown()
