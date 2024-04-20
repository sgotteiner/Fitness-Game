from Constants import POSE_NAMES
from ImageDetector import capture_images, save_pose_images
from PoseModelHandler import process_saved_images


if __name__ == '__main__':
    # save_pose_images(POSE_NAMES.NONE)
    # body_landmarks = capture_images()
    process_saved_images(POSE_NAMES.NONE, False)
