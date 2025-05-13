#!/usr/bin/env python3
import rclpy
from tag import Tag
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.executors import MultiThreadedExecutor
from threading import Lock, Thread
import sys
import time
import rclpy
from typing import Any, List, Tuple, Union
import interbotix_common_modules.angle_manipulation as ang
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from interbotix_xs_modules.xs_robot import mr_descriptions as mrd
from sensor_msgs.msg import JointState
import math
import sys
from threading import Lock
from typing import List
import modern_robotics as mr
import numpy as np

from interbotix_xs_msgs.msg import (
    JointGroupCommand,
    JointSingleCommand,
    JointTrajectoryCommand
)
from interbotix_xs_msgs.srv import (
    MotorGains,
    OperatingModes,
    Reboot,
    RegisterValues,
    RobotInfo,
    TorqueEnable,
)

sys.path.append('/home/student/interbotix_ws')

REV: float = 2 * np.pi

class SimpleNode(Node):
    def __init__(self, name: str, robot_name: str):
        super().__init__(name)
        self.robot_name = robot_name
        self.publisher = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info(f"Node {name} has started.")

    def timer_callback(self):
        msg = String()
        msg.data = f"Robot Name: {self.robot_name}"
        self.publisher.publish(msg)
        self.get_logger().info(f"Published message: {msg.data}")

class InterbotixRobotXSCore(Node):
    """Class that interfaces with the xs_sdk node ROS interfaces."""

    def __init__(
        self,
        robot_model: str,
        robot_name: str = None,
        topic_joint_states: str = 'joint_states',
        node_name: str = 'robot_manipulation',
        args=None
    ) -> None:
        super().__init__(node_name=node_name)

        self.robot_model = robot_model
        self.robot_name = robot_name or robot_model
        self.node_name = node_name
        self.ns = f'/{self.robot_name}' if self.robot_name else ''

        self.topic_joint_states = topic_joint_states
        self.joint_states: JointState = None
        self.js_mutex = Lock()

        self.srv_set_op_modes = self.create_client(
            OperatingModes, f'{self.ns}/set_operating_modes'
        )
        self.srv_set_pids = self.create_client(
            MotorGains, f'{self.ns}/set_motor_pid_gains'
        )
        self.srv_set_reg = self.create_client(
            RegisterValues, f'{self.ns}/set_motor_registers'
        )
        self.srv_get_reg = self.create_client(
            RegisterValues, f'{self.ns}/get_motor_registers'
        )
        self.srv_get_info = self.create_client(
            RobotInfo, f'{self.ns}/get_robot_info'
        )
        self.srv_torque = self.create_client(
            TorqueEnable, f'{self.ns}/torque_enable'
        )
        self.srv_reboot = self.create_client(
            Reboot, f'{self.ns}/reboot_motors'
        )

        if not self.srv_set_op_modes.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                f"Failed to find services under namespace '{self.robot_name}'. Is the xs_sdk running? Shutting down..."
            )
            sys.exit(1)

        self.srv_set_pids.wait_for_service()
        self.srv_set_reg.wait_for_service()
        self.srv_get_reg.wait_for_service()
        self.srv_get_info.wait_for_service()
        self.srv_torque.wait_for_service()
        self.srv_reboot.wait_for_service()

        self.pub_group = self.create_publisher(
            JointGroupCommand, f'{self.ns}/commands/joint_group', 10
        )
        self.pub_single = self.create_publisher(
            JointSingleCommand, f'{self.ns}/commands/joint_single', 10
        )
        self.pub_traj = self.create_publisher(
            JointTrajectoryCommand, f'{self.ns}/commands/joint_trajectory', 10
        )
        
        # Debug the topic name to ensure correctness
        self.get_logger().debug(
            f"Subscribing to joint states on topic '{self.ns}/{self.topic_joint_states}'"
        )
        
        self.sub_joint_states = self.create_subscription(
            JointState,
            f'{self.ns}/{self.topic_joint_states}',
            self._joint_state_cb,
            10,
        )
        self.get_logger().debug((
                'Trying to find joint states on topic '
                f"'{self.ns}/{self.topic_joint_states}_{self.robot_name}'..."
        ))
        while self.joint_states is None and rclpy.ok():
            rclpy.spin_once(self)
        self.get_logger().debug('Found joint states. Continuing...')

        self.js_index_map = dict(
            zip(self.joint_states.name, range(len(self.joint_states.name)))
        )
        self.get_logger().info(
            f"\nRobot Name: {self.robot_name}\nRobot Model: {self.robot_model}"
        )
        self.get_logger().info('Initialized InterbotixRobotXSCore!')

    def _joint_state_cb(self, msg: JointState):
        """Callback to process JointState message."""
        with self.js_mutex:
            self.joint_states = msg
            self.get_logger().debug("JointState callback triggered")

class InterbotixArmXSInterface:
    """Definition of the Interbotix Arm component."""

    def __init__(
        self,
        core: InterbotixRobotXSCore,
        robot_model: str,
        group_name: str,
        moving_time: float = 2.0,
        accel_time: float = 0.3,
    ) -> None:

        self.core = core
        self.robot_model = robot_model
        self.moving_time, self.accel_time = moving_time, accel_time
        self.group_name = group_name

        self.robot_des: mrd.ModernRoboticsDescription = getattr(mrd, self.robot_model)

        self.future_group_info = self.core.srv_get_info.call_async(
            RobotInfo.Request(cmd_type='group', name=group_name)
        )
        while rclpy.ok() and not self.future_group_info.done():
            rclpy.spin_until_future_complete(self.core, self.future_group_info)
            rclpy.spin_once(self.core)

        self.group_info: RobotInfo.Response = self.future_group_info.result()
        if self.group_info.profile_type != 'time':
            self.core.get_logger().error(
                "Please set the group's 'profile_type' to 'time'."
            )
            exit(1)
        if self.group_info.mode != 'position':
            self.core.get_logger().error(
                "Please set the group's 'operating mode' to 'position'."
            )
            exit(1)

        self.initial_guesses = [[0.0] * self.group_info.num_joints] * 3
        self.initial_guesses[1][0] = np.deg2rad(-120)
        self.initial_guesses[2][0] = np.deg2rad(120)
        self.joint_commands = []

        # update joint_commands with the present joint positions
        for name in self.group_info.joint_names:
            self.joint_commands.append(
                self.core.joint_states.position[self.core.js_index_map[name]]
            )
        # get the initial transform between the space and body frames
        self._update_Tsb()
        self.set_trajectory_time(self.moving_time, self.accel_time)

        # build the info index map between joint names and their index
        self.info_index_map = dict(
            zip(self.group_info.joint_names, range(self.group_info.num_joints))
        )

        self.core.get_logger().info(
            (
                '\n'
                f'\tArm Group Name: {self.group_name}\n'
                f'\tMoving Time: {self.moving_time:.2f} seconds\n'
                f'\tAcceleration Time: {self.accel_time:.2f} seconds\n'
                f'\tDrive Mode: Time-Based-Profile'
            )
        )
        self.core.get_logger().info('Initialized InterbotixArmXSInterface!')

    def set_trajectory_time(
        self,
        moving_time: float = None,
        accel_time: float = None
    ) -> None:
        
        self.core.get_logger().debug(
            f'Updating timing params: {moving_time=}, {accel_time=}'
        )
        if moving_time is not None and moving_time != self.moving_time:
            self.moving_time = moving_time
            future_moving_time = self.core.srv_set_reg.call_async(
                RegisterValues.Request(
                    cmd_type='group',
                    name=self.group_name,
                    reg='Profile_Velocity',
                    value=int(moving_time * 1000),
                )
            )
            self.core.executor.spin_once_until_future_complete(
                future=future_moving_time,
                timeout_sec=0.1
            )

        if accel_time is not None and accel_time != self.accel_time:
            self.accel_time = accel_time
            future_accel_time = self.core.srv_set_reg.call_async(
                RegisterValues.Request(
                    cmd_type='group',
                    name=self.group_name,
                    reg='Profile_Acceleration',
                    value=int(accel_time * 1000),
                )
            )
            self.core.executor.spin_once_until_future_complete(
                future=future_accel_time,
                timeout_sec=0.1
            )

    def go_to_sleep_pose(
        self,
        moving_time: float = None,
        accel_time: float = None,
        blocking: bool = True
    ) -> None:
        """
        Command the arm to go to its Sleep pose.

        :param moving_time: (optional) duration in seconds that the robot should move
        :param accel_time: (optional) duration in seconds that that robot should spend
            accelerating/decelerating (must be less than or equal to half the moving_time)
        :param blocking: (optional) whether the function should wait to return control to the user
            until the robot finishes moving
        """
        self.core.get_logger().debug('Going to sleep pose')
        self._publish_commands(
            positions=self.group_info.joint_sleep_positions,
            moving_time=moving_time,
            accel_time=accel_time,
            blocking=blocking
        )

    def _publish_commands(
        self,
        positions: List[float],
        moving_time: float = None,
        accel_time: float = None,
        blocking: bool = True,
    ) -> None:
        """
        Publish joint positions and block if necessary.

        :param positions: desired joint positions
        :param moving_time: (optional) duration in seconds that the robot should move
        :param accel_time: (optional) duration in seconds that that robot should spend
            accelerating/decelerating (must be less than or equal to half the moving_time)
        :param blocking: (optional) whether the function should wait to return control to the user
            until the robot finishes moving
        """
        self.core.get_logger().debug(f'Publishing {positions=}')
        self.set_trajectory_time(moving_time, accel_time)
        self.joint_commands = list(positions)
        joint_commands = JointGroupCommand(
            name=self.group_name, cmd=self.joint_commands
        )
        self.core.pub_group.publish(joint_commands)
        if blocking:
            time.sleep(
                self.moving_time
            )  # TODO: once released, use rclpy.clock().sleep_for()
        self._update_Tsb()

    def _update_Tsb(self) -> None:
        """Update transform between the space and body frame from the current joint commands."""
        self.core.get_logger().debug('Updating T_sb')
        self.T_sb = mr.FKinSpace(
            self.robot_des.M, self.robot_des.Slist, self.joint_commands
        )

    def go_to_home_pose(
        self,
        moving_time: float = None,
        accel_time: float = None,
        blocking: bool = True
    ) -> None:

        self.core.get_logger().debug('Going to home pose')
        self._publish_commands(
            positions=[0] * self.group_info.num_joints,
            moving_time=moving_time,
            accel_time=accel_time,
            blocking=blocking
        )

    def set_ee_pose_matrix(
        self,
        T_sd: np.ndarray,
        custom_guess: List[float] = None,
        execute: bool = True,
        moving_time: float = None,
        accel_time: float = None,
        blocking: bool = True,
    ) -> Tuple[Union[np.ndarray, Any, List[float]], bool]:
        
        self.core.get_logger().debug(f'Setting ee_pose to matrix=\n{T_sd}')
        if custom_guess is None:
            initial_guesses = self.initial_guesses
        else:
            initial_guesses = [custom_guess]

        for guess in initial_guesses:
            theta_list, success = mr.IKinSpace(
                Slist=self.robot_des.Slist,
                M=self.robot_des.M,
                T=T_sd,
                thetalist0=guess,
                eomg=0.001,
                ev=0.001,
            )
            solution_found = True

            # Check to make sure a solution was found and that no joint limits were violated
            if success:
                theta_list = self._wrap_theta_list(theta_list)
                solution_found = self._check_joint_limits(theta_list)
            else:
                solution_found = False

            if solution_found:
                if execute:
                    self._publish_commands(
                        theta_list, moving_time, accel_time, blocking
                    )
                    self.T_sb = T_sd
                return theta_list, True

        self.core.get_logger().warn('No valid pose could be found. Will not execute')
        return theta_list, False
    
    def _wrap_theta_list(self, theta_list: List[np.ndarray]) -> List[np.ndarray]:
        theta_list = (theta_list + np.pi) % REV - np.pi
        for x in range(len(theta_list)):
            if round(theta_list[x], 3) < round(self.group_info.joint_lower_limits[x], 3):
                theta_list[x] += REV
            elif round(theta_list[x], 3) > round(self.group_info.joint_upper_limits[x], 3):
                theta_list[x] -= REV
        return theta_list
    
    def _check_joint_limits(self, positions: List[float]) -> bool:
        self.core.get_logger().debug(f'Checking joint limits for {positions=}')
        theta_list = [int(elem * 1000) / 1000.0 for elem in positions]
        speed_list = [
            abs(goal - current) / float(self.moving_time)
            for goal, current in zip(theta_list, self.joint_commands)
        ]
        # check position and velocity limits
        for x in range(self.group_info.num_joints):
            if not (
                self.group_info.joint_lower_limits[x]
                <= theta_list[x]
                <= self.group_info.joint_upper_limits[x]
            ):
                return False
            if speed_list[x] > self.group_info.joint_velocity_limits[x]:
                return False
        return True
    
    def set_single_joint_position(
        self,
        joint_name: str,
        position: float,
        moving_time: float = None,
        accel_time: float = None,
        blocking: bool = True,
    ) -> bool:
        """
        Command a single joint to a desired position.

        :param joint_name: name of the joint to control
        :param position: desired position [rad]
        :param moving_time: (optional) duration in seconds that the robot should move
        :param accel_time: (optional) duration in seconds that that robot should spend
            accelerating/decelerating (must be less than or equal to half the moving_time)
        :param blocking: (optional) whether the function should wait to return control to the user
              until the robot finishes moving
        :return: `True` if single joint was set; `False` otherwise
        :details: Note that if a moving_time or accel_time is specified, the changes affect ALL the
            arm joints, not just the specified one
        """
        self.core.get_logger().debug(
            f'Setting joint {joint_name} to position={position}'
        )
        if not self._check_single_joint_limit(joint_name, position):
            return False
        self.set_trajectory_time(moving_time, accel_time)
        self.joint_commands[self.core.js_index_map[joint_name]] = position
        single_command = JointSingleCommand(name=joint_name, cmd=position)
        self.core.pub_single.publish(single_command)
        if blocking:
            time.sleep(self.moving_time)
        self._update_Tsb()
        return True
    
    def _check_single_joint_limit(self, joint_name: str, position: float) -> bool:
        """
        Ensure a desired position for a given joint is within its limits.

        :param joint_name: desired joint name
        :param position: desired joint position [rad]
        :return: `True` if within limits; `False` otherwise
        """
        self.core.get_logger().debug(
            f'Checking joint {joint_name} limits for {position=}'
        )
        theta = int(position * 1000) / 1000.0
        speed = abs(
            theta - self.joint_commands[self.info_index_map[joint_name]]
        ) / float(self.moving_time)
        ll = self.group_info.joint_lower_limits[self.info_index_map[joint_name]]
        ul = self.group_info.joint_upper_limits[self.info_index_map[joint_name]]
        vl = self.group_info.joint_velocity_limits[self.info_index_map[joint_name]]
        if not (ll <= theta <= ul):
            return False
        if speed > vl:
            return False
        return True

    def set_ee_pose_components(
        self,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        roll: float = 0,
        pitch: float = 0,
        yaw: float = None,
        custom_guess: List[float] = None,
        execute: bool = True,
        moving_time: float = None,
        accel_time: float = None,
        blocking: bool = True,
    ) -> Tuple[Union[np.ndarray, Any, List[float]], bool]:

        if self.group_info.num_joints < 6 or (
            self.group_info.num_joints >= 6 and yaw is None
        ):
            yaw = math.atan2(y, x)
        self.core.get_logger().debug(
            (
                f'Setting ee_pose components=\n'
                f'\t{x=}\n'
                f'\t{y=}\n'
                f'\t{z=}\n'
                f'\t{roll=}\n'
                f'\t{pitch=}\n'
                f'\t{yaw=}'
            )
        )
        T_sd = np.identity(4)
        T_sd[:3, :3] = ang.euler_angles_to_rotation_matrix([roll, pitch, yaw])
        T_sd[:3, 3] = [x, y, z]
        return self.set_ee_pose_matrix(
            T_sd, custom_guess, execute, moving_time, accel_time, blocking
        )
    
class InterbotixGripperXS:
    """Standalone Module to control an Interbotix Gripper using PWM or Current control."""

    def __init__(
        self,
        robot_model: str,
        gripper_name: str,
        robot_name: str = None,
        gripper_pressure: float = 0.5,
        gripper_pressure_lower_limit: int = 150,
        gripper_pressure_upper_limit: int = 350,
        topic_joint_states: str = 'joint_states',
        logging_level: LoggingSeverity = LoggingSeverity.INFO,
        node_name: str = 'robot_manipulation',
        start_on_init: bool = True,
    ) -> None:
        """
        Construct the Standalone Interbotix Gripper Module.

        :param robot_model: Interbotix Arm model (ex. 'wx200' or 'vx300s')
        :param gripper_name: name of the gripper joint as defined in the 'motor_config' yaml file;
            typically, this is 'gripper'
        :param robot_name: (optional) defaults to value given to 'robot_model'; this can be
            customized to best suit the user's needs
        :param gripper_pressure: (optional) fraction from 0 - 1 where '0' means the gripper
            operates at 'gripper_pressure_lower_limit' and '1' means the gripper operates at
            'gripper_pressure_upper_limit'
        :param gripper_pressure_lower_limit: (optional) lowest 'effort' that should be applied to
            the gripper if gripper_pressure is set to 0; it should be high enough to open/close the
            gripper (~150 PWM or ~400 mA current)
        :param gripper_pressure_upper_limit: (optional) largest 'effort' that should be applied to
            the gripper if gripper_pressure is set to 1; it should be low enough that the motor
            doesn't 'overload' when gripping an object for a few seconds (~350 PWM or ~900 mA)
        :param topic_joint_states: (optional) the specifc JointState topic output by the xs_sdk
            node
        :param logging_level: (optional) rclpy logging severity level. Can be DEBUG, INFO, WARN,
            ERROR, or FATAL. defaults to INFO
        :param node_name: (optional) name to give to the core started by this class, defaults to
            'robot_manipulation'
        :param start_on_init: (optional) set to `True` to start running the spin thread after the
            object is built; set to `False` if intending to sub-class this. If set to `False`,
            either call the `start()` method later on, or add the core to an executor in another
            thread.
        :details: note that this module doesn't really have any use case except in controlling just
            the gripper joint on an Interbotix Arm.
        """
        self.core = InterbotixRobotXSCore(
            robot_model,
            robot_name,
            topic_joint_states=topic_joint_states,
            logging_level=logging_level,
            node_name=node_name,
        )
        self.gripper = InterbotixGripperXSInterface(
            self.core,
            gripper_name,
            gripper_pressure,
            gripper_pressure_lower_limit,
            gripper_pressure_upper_limit,
        )

        if start_on_init:
            self.start()

    def start(self) -> None:
        """Start a background thread that builds and spins an executor."""
        self._execution_thread = Thread(target=self.run)
        self._execution_thread.start()

    def run(self) -> None:
        """Thread target."""
        self.ex = MultiThreadedExecutor()
        self.ex.add_node(self.core)
        self.ex.spin()

    def shutdown(self) -> None:
        """Destroy the node and shut down all threads and processes."""
        self.core.destroy_node()
        rclpy.shutdown()
        self._execution_thread.join()
        time.sleep(0.5)


class InterbotixGripperXSInterface:
    def __init__(
        self,
        core: InterbotixRobotXSCore,
        gripper_name: str,
        gripper_pressure: float = 0.5,
        gripper_pressure_lower_limit: int = 150,
        gripper_pressure_upper_limit: int = 350,
    ) -> None:
        
        self.core = core
        self.gripper_name = gripper_name
        self.gripper_pressure = gripper_pressure
        self.future_gripper_info = self.core.srv_get_info.call_async(
            RobotInfo.Request(cmd_type='single', name='gripper')
        )
        self.gripper_moving: bool = False
        self.gripper_command = JointSingleCommand(name='gripper')
        self.gripper_pressure_lower_limit = gripper_pressure_lower_limit
        self.gripper_pressure_upper_limit = gripper_pressure_upper_limit

        # value = lower + pressure * range
        self.gripper_value = gripper_pressure_lower_limit + (
            gripper_pressure
            * (gripper_pressure_upper_limit - gripper_pressure_lower_limit)
        )

        self.tmr_gripper_state = self.core.create_timer(
            timer_period_sec=0.02, callback=self.gripper_state
        )

        while rclpy.ok() and not self.future_gripper_info.done():
            rclpy.spin_until_future_complete(self.core, self.future_gripper_info)
            rclpy.spin_once(self.core)

        self.gripper_info: RobotInfo.Response = self.future_gripper_info.result()
        self.left_finger_index = self.core.js_index_map[self.gripper_info.joint_names[0]]
        self.left_finger_lower_limit = self.gripper_info.joint_lower_limits[0]
        self.left_finger_upper_limit = self.gripper_info.joint_upper_limits[0]

        if self.gripper_info.mode not in ('current', 'pwm'):
            self.core.get_logger().err(
                "Please set the gripper's 'operating mode' to 'pwm' or 'current'."
            )
            sys.exit(1)

        time.sleep(0.5)
        self.core.get_logger().info(
            (
                '\n'
                f'\tGripper Name: {self.gripper_name}\n'
                f'\tGripper Pressure: {self.gripper_pressure*100}%'
            )
        )
        self.core.get_logger().info('Initialized InterbotixGripperXSInterface!')

    def gripper_state(self) -> None:
        """Stop the gripper moving past its limits when in PWM mode using a ROS Timer Callback."""
        if self.gripper_moving:
            # update gripper position
            with self.core.js_mutex:
                gripper_pos = self.core.joint_states.position[self.left_finger_index]
            # stop the gripper if it has reached the lower or upper limit

    def gripper_controller(self, effort: float, delay: float) -> None:
        """
        Publish effort commands to the gripper (when in 'pwm' or 'current' mode).

        :param effort: effort command to send to the gripper motor
        :param delay: number of seconds to wait before returning control to the user
        """
        self.gripper_command.cmd = effort
        # update gripper position
        with self.core.js_mutex:
            gripper_pos = self.core.joint_states.position[self.left_finger_index]
        # check if the gripper is within its limits
        self.core.pub_single.publish(self.gripper_command)
        self.gripper_moving = True
        time.sleep(delay)

    def set_pressure(self, pressure: float) -> None:
        """
        Set the amount of pressure that the gripper should use when grasping an object.

        :param pressure: a scaling factor from 0 to 1 where the pressure increases as
            the factor increases
        """
        self.gripper_value = self.gripper_pressure_lower_limit + pressure * (
            self.gripper_pressure_upper_limit - self.gripper_pressure_lower_limit
        )

    def release(self, delay: float = 1.0) -> None:
        """
        Open the gripper (when in 'pwm' control mode).

        :param delay: (optional) number of seconds to delay before returning control to the user
        """
        self.gripper_controller(self.gripper_value, delay)

    def grasp(self, delay: float = 1.0) -> None:
        """
        Close the gripper (when in 'pwm' control mode).

        :param delay: (optional) number of seconds to delay before returning control to the user
        """
        self.gripper_controller(-self.gripper_value, delay)

class InterbotixManipulatorXS:
    """Standalone Module to control an Interbotix Arm and Gripper."""

    def __init__(
        self,
        robot_model: str,
        group_name: str = 'arm',
        gripper_name: str = 'gripper',
        robot_name: str = None,
        tag: Tag = None,
        moving_time: float = 2.0,
        accel_time: float = 0.3,
        gripper_pressure: float = 0.5,
        gripper_pressure_lower_limit: int = 150,
        gripper_pressure_upper_limit: int = 350,
        topic_joint_states: str = 'joint_states',
        logging_level: int = 20,  # INFO level logging
        node_name: str = 'robot_manipulation',
        start_on_init: bool = True,
        args=None,
    ) -> None:
        
        self.tag = tag
        
        # Initialize core robot interface
        self.core = InterbotixRobotXSCore(
            robot_model=robot_model,
            robot_name=robot_name,
            topic_joint_states=topic_joint_states,
            node_name=node_name,
            args=args
        )
        self.arm = InterbotixArmXSInterface(
            core=self.core,
            robot_model=robot_model,
            group_name=group_name,
            moving_time=moving_time,
            accel_time=accel_time,
        )
        if gripper_name is not None:
            self.gripper = InterbotixGripperXSInterface(
                core=self.core,
                gripper_name=gripper_name,
                gripper_pressure=gripper_pressure,
                gripper_pressure_lower_limit=gripper_pressure_lower_limit,
                gripper_pressure_upper_limit=gripper_pressure_upper_limit,
            )
