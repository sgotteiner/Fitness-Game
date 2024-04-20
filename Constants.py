from enum import Enum

POSES_OUTPUT_DIR = 'poses_pictures\\'

class POSE_NAMES(str, Enum):
    LEFT_HAND_PUNCHES = 'left hand punches',
    RIGHT_HAND_PUNCHES = 'right hand punches',
    SQUATS = 'squats'
    NONE = 'none'