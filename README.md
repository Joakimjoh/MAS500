# Dual-Arm Robotic Textile Unfolding System

This project implements a dual-arm robotic system for automatic textile unfolding using synchronized Interbotix manipulators, Intel RealSense depth perception, YOLO-based fold detection, and a finite state machine.

---

## Features

- Dual-arm manipulation using Interbotix arms  
- Real-time depth and RGB processing from an Intel RealSense camera  
- AprilTag-based spatial calibration and grasp point estimation  
- Depth correction using machine learning regression models  
- Fold detection using a YOLO object detection model  
- State machine for controlling robotic unfolding behavior  
- Manual and automatic execution modes via keyboard controls  

---

## Folder Contents

| File / Folder                      | Description |
|-----------------------------------|-------------|
| `main.py`                         | Main entry point. Initializes camera, arms, and starts the unfolding loop |
| `camera.py`                       | Camera wrapper for Intel RealSense image and depth capture |
| `tag.py`                          | AprilTag detector, spatial pose estimation, and depth correction model interface |
| `frame.py`                        | Visual GUI rendering and interaction for debug or manual control |
| `process.py`                      | State machine controller for the dual-arm unfolding pipeline |
| `dual_arm_xs.py`                  | Custom Interbotix wrapper for synchronized dual-arm control |
| `machine_learning.ipynb`          | Jupyter notebook for training and evaluating depth correction models |
| `region.csv`                      | Sampled training data (auto-generated on first run) |
| `error_model_left.joblib`         | Left arm's trained depth correction model (auto-generated) |
| `error_model_right.joblib`        | Right arm's trained depth correction model (auto-generated) |
| `requirements.txt`                | Python dependencies |

---

## Installation

pip install -r requirements.txt

## Hardware Requirements
Two Interbotix manipulators (e.g., WX250s)

Intel RealSense camera (e.g., D435)

AprilTags printed and placed in camera view

ROS 2 installed (tested with ROS 2 Galactic)

## Running the System
Connect all hardware (Interbotix arms, RealSense camera).
The left most robot arm, relative to the camera view, should be connected first.

Initialized with the command under:
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=wx250s robot_name:=left_arm robot_usb_port:=/dev/ttyUSB0

Initialize second (right) robot arm:
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=wx250s robot_name:=right_arm robot_usb_port:=/dev/ttyUSB1 motor_configs:=/home/student/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_control/config/wx250s_2.yaml

You need two identical yaml files for the robot arms so services for each motor and robot arm have different namespaces.

Run the main script:
python3 main.py

## User Controls
In the GUI window:

Press r to toggle Auto / Manual mode

In Manual mode:

w: Proceed to next step

q: Go back to previous step

e: Manually select a pixel as a grasp point

s: Send arms to sleep position

d: Detect fold and display results

In Auto mode: system will detect and act automatically
