from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

def initialize_robots():
    """
    Initializes two Interbotix VX250 6-DOF robotic arms and returns their objects.

    Returns:
        tuple: A tuple containing two initialized robot objects (robot1, robot2).
    """
    try:
        # Initialize the first robot
        robot1 = InterbotixManipulatorXS(
            robot_model="vx250",
            robot_name="vx250_robot1",
            use_gripper=True
        )
        print("Robot 1 initialized successfully.")

        # Initialize the second robot
        robot2 = InterbotixManipulatorXS(
            robot_model="vx250",
            robot_name="vx250_robot2",
            use_gripper=True
        )
        print("Robot 2 initialized successfully.")

        if robot1 and robot2:
            # Move robots to initial positions
            robot1.arm.go_to_home_pose()
            robot2.arm.go_to_home_pose()

            print("Both robots are ready for commands!")
        else:
            print("Failed to initialize one or both robots.")
                
        return robot1, robot2

    except Exception as e:
        print(f"An error occurred while initializing robots: {e}")
        return None, None
