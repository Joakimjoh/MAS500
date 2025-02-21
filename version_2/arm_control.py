import camera_detection
import threading

def move_arm(bot, x, y, z, pitch=1):
    bot.arm.set_ee_pose_components(x, y, z, pitch=pitch)

def pick_up_object(bot, barrier, x, y, z, pitch=1):
    bot.gripper.release()

    bot.arm.set_ee_pose_components(x, y, z + 0.1, pitch=pitch)

    barrier.wait()

    bot.arm.set_ee_pose_components(x, y, z + 0.05, pitch=pitch)
    
    bot.gripper.grasp(0.1)

    bot.arm.set_ee_pose_components(x, y, z, pitch=pitch)

    barrier.wait()

    if y > 0:
        bot.arm.set_ee_pose_components(x, 0.25, 0.25, pitch=pitch)
    elif y < 0:
        bot.arm.set_ee_pose_components(x, -0.25, 0.25, pitch=pitch)

def lay_flat_object(bot, x, y, pitch=1):
    bot.arm.set_ee_pose_components(x, 0, 0.1, pitch=pitch)

    if y > 0:
        bot.arm.set_ee_pose_components(x, 0.25, 0.1, pitch=pitch)
    else:
        bot.arm.set_ee_pose_components(x, -0.25, 0.1, pitch=pitch)

    bot.gripper.release()
    bot.arm.go_to_sleep_pose()

def stretch(bot, barrier, x, z=0.25, stretch_rate=0.005, pitch=1):
    if bot.core.robot_name == 'arm1':
        y = -0.25
    else:
        y = 0.25
        
    while not camera_detection.is_straight:
        x += stretch_rate
        bot.arm.set_ee_pose_components(x, y, z, pitch=pitch)
        barrier.wait()

def step1(bot, x, y, z):
        barrier = threading.Barrier(2)

        pick_up_object(bot, barrier, x, y, z)

        stretch(bot, barrier, x)

        lay_flat_object(bot, x, y)