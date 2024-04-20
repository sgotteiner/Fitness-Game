import cv2  # Import opencv
import os
import time
import mediapipe as mp  # Import mediapipe

from Constants import POSES_OUTPUT_DIR

mp_holistic = mp.solutions.holistic  # Mediapipe Solutions


def process_image(image):
    """
    :param image: cvw image from video capture or from reading a file
    :return: mediapipe pose landmarks without the face and fingers
    """

    excluded_ranges = [(1, 11), (17, 23), (29, 33)]  # face, hand fingers, toe and leg fingers
    desired_landmarks = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # required for mediapipe model
    image.flags.writeable = False  # slight performance improvement

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Get Mediapipe pose detections
        results = holistic.process(image)
        if not results.pose_landmarks:
            return None

        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if not any(start <= idx < end for start, end in excluded_ranges):
                desired_landmarks.append(landmark)

    return desired_landmarks

def draw_desired_landmarks(image, desired_landmarks):
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw Pose Detections. The reason I don't use mediapipe draw_landmarks function is because I don't want all the landmarks but only my desired landmarks
    for landmark in desired_landmarks:
        cv2.circle(image, (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])),
                   radius=5, color=(255, 0, 0), thickness=-1)
    cv2.imshow('Raw Webcam Feed', image)

def capture_images():
    """
    This function is for me to see mediapipe holistic model
    """

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        desired_landmarks = process_image(image)
        if not desired_landmarks:
            continue
        draw_desired_landmarks(image, desired_landmarks)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def save_pose_images(pose_name):
    """
    Capture TARGET_FPS images per second to create a dataset for this pose.
    To create the squat pose dataset for example I run this function and do squats. Later I delete the non-squat images.
    :param pose_name: The images are saved in OUTPUT_DIR\pose_name directory
    """

    TARGET_FPS = 10  # control the number of frames
    cap = cv2.VideoCapture(0)
    frame_count = 0
    start_time = time.time()  # Track processing time

    while True:
        ret, frame = cap.read()

        # Break the loop if there are no more frames to read or 'q' is pressed
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed_time = time.time() - start_time
        if elapsed_time < 1 / TARGET_FPS:
            continue  # Skip frame if not enough time has passed

        # Save the frame with a descriptive filename (e.g., frame_0000.jpg)
        filename = os.path.join(POSES_OUTPUT_DIR + pose_name, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        frame_count += 1

        cv2.imshow('Camera', frame)

    cap.release()
    cv2.destroyAllWindows()