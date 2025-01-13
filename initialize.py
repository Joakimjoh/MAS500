import time
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

def initialize_robots():
    """
    Initializes two Interbotix VX250 6-DOF robotic arms and keeps trying until both are ready.

    Returns:
        tuple: A tuple containing two initialized robot objects (robot1, robot2).
    """
    while True:
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
                print("Both robots are ready for commands!")
                return robot1, robot2  # Return both initialized robots

        except Exception as e:
            print(f"An error occurred while initializing robots: {e}")

        print("Retrying initialization in 5 seconds...")
        time.sleep(5)  # Wait for 5 seconds before retrying
