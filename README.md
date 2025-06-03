The left most robot arm, relative to the camera view, should be connected first and initialized with the command under:
CONNECT LEFT MOST ROBOT, RELATIVE TO WORKSPACE, FIRST WITH USB CABLE SO ITS IN THE FIRST PORT
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=wx250s robot_name:=left_arm robot_usb_port:=/dev/ttyUSB0

Initialize second (right) robot arm:
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=wx250s robot_name:=right_arm robot_usb_port:=/dev/ttyUSB1 motor_configs:=/home/student/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_control/config/wx250s_2.yaml

You need two identical yaml files for the robot arms so services for each motor and robot arm have different namespaces.

ØK TORQUE:
ros2 service call /arm1/set_motor_registers interbotix_xs_msgs/srv/RegisterValues "{cmd_type: 'single', name: 'elbow', reg: 'Position_P_Gain', value: 1500}"

export ROS_DOMAIN_ID=99  # Choose a number between 1-101 (must be the same on all devices you want to communicate with)
