import csv
import os
import cv2
import numpy as np
from Constants import POSES_OUTPUT_DIR
from ImageDetector import process_image, draw_desired_landmarks


def process_saved_images(pose_name, is_test):
    if not is_test:
        for file in os.listdir(POSES_OUTPUT_DIR + pose_name):
            image = cv2.imread(os.path.join(POSES_OUTPUT_DIR + pose_name, file))
            image.flags.writeable = True
            desired_landmarks = process_image(image)
            if not desired_landmarks:
                continue

            write_landmarks_in_csv(pose_name, desired_landmarks)
    else:
        test_file_name = 'frame_0555.jpg'
        image = cv2.imread(os.path.join(POSES_OUTPUT_DIR + pose_name, test_file_name))
        image.flags.writeable = True
        desired_landmarks = process_image(image)
        if not desired_landmarks:
            return
        draw_desired_landmarks(image, desired_landmarks)
        cv2.waitKey(5000)


POSE_CSV = 'pose.csv'

def create_initial_csv():
    formatted_landmarks = ['class']
    NUMBER_OF_DESIRED_LANDMARKS = 13  # 33 pose - 10 face - 6 hand fingers - 4 leg fingers
    for val in range(1, NUMBER_OF_DESIRED_LANDMARKS + 1):
        formatted_landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

    with open(POSE_CSV, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(formatted_landmarks)


def write_landmarks_in_csv(pose_name, landmarks):
    if os.path.getsize(POSE_CSV) == 0:
        create_initial_csv()

    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())
    pose_row.insert(0, pose_name)
    with open(POSE_CSV, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(pose_row)