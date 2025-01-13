import globals
from cam_dec import check_line_straightness, right_left_corners
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
        if check_line_straightness():
            print("Line Straight")
            break
        else:
            print("Line not straight") 
            # Move robots back a distance
    
    # Lay item flat
