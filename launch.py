import subprocess
import threading

def launch_both_robots():
    """
    Launches two Interbotix robotic arms with unique namespaces and opens a separate terminal window
    for stopping commands.
    """
    def launch_robot(robot_name, model_name):
        """
        Launches a single robot using its ROS 2 launch file.

        Args:
            robot_name (str): Unique namespace for the robot (e.g., 'vx250_robot1').
            model_name (str): Model name of the robot (e.g., 'vx250').

        Returns:
            subprocess.Popen: The subprocess running the launch file.
        """
        try:
            # ROS 2 launch command
            command = [
                "ros2", "launch", "interbotix_xsarm_control", "xsarm_control.launch.py",
                f"robot_name:={robot_name}",
                f"robot_model:={model_name}"
            ]
            print(f"Launching {robot_name}...")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return process
        except Exception as e:
            print(f"Failed to launch {robot_name}: {e}")
            return None

    def command_listener(process1, process2):
        """
        Listens for stop commands in the terminal to terminate the processes.
        """
        print("Type 'q1' to stop Robot 1, 'q2' to stop Robot 2, or 'exit' to stop both.")
        while True:
            command = input("Enter command: ").strip().lower()
            if command == "q1":
                print("Stopping Robot 1...")
                if process1.poll() is None:  # Check if process1 is still running
                    process1.terminate()
                    process1.wait()
                    print("Robot 1 stopped.")
                else:
                    print("Robot 1 is already stopped.")
            elif command == "q2":
                print("Stopping Robot 2...")
                if process2.poll() is None:  # Check if process2 is still running
                    process2.terminate()
                    process2.wait()
                    print("Robot 2 stopped.")
                else:
                    print("Robot 2 is already stopped.")
            elif command == "exit":
                print("Stopping both robots...")
                if process1.poll() is None:
                    process1.terminate()
                    process1.wait()
                    print("Robot 1 stopped.")
                if process2.poll() is None:
                    process2.terminate()
                    process2.wait()
                    print("Robot 2 stopped.")
                break
            else:
                print("Unknown command. Use 'q1', 'q2', or 'exit'.")

    # Define robot names and models
    robot1_name = "vx250_robot1"
    robot2_name = "vx250_robot2"
    robot_model = "vx250"

    # Launch the robots
    process1 = launch_robot(robot1_name, robot_model)
    process2 = launch_robot(robot2_name, robot_model)

    # Check if both robots launched successfully
    if process1 and process2:
        print("Both robots launched successfully!")

        # Open a separate terminal window for stopping commands
        threading.Thread(target=command_listener, args=(process1, process2), daemon=True).start()

        # Wait for both processes to complete
        try:
            process1.wait()
            process2.wait()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Shutting down robots...")
            if process1.poll() is None:
                process1.terminate()
            if process2.poll() is None:
                process2.terminate()
    else:
        print("Failed to launch one or both robots.")
