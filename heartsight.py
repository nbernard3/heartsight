
from contextlib import contextmanager
from datetime import datetime
import numpy as np
import cv2
import dlib
import time


def monitor_heart_rate():

    with open_webcam() as webcam:
        face_detector = dlib.get_frontal_face_detector()
        landmarks_predictor = dlib.shape_predictor(
            "models/face/shape_predictor_5_face_landmarks.dat")
        face_size = 160

        exit_requested = False
        while not exit_requested:
            rgb_frame = capture_frame(webcam)
            gray_frame = rgb_to_gray(rgb_frame)
            detected_faces = face_detector(gray_frame, 1)

            if detected_faces:
                face = detected_faces[0]
                aligned_face = align_face(
                    face, gray_frame, rgb_frame, landmarks_predictor, face_size)

                rgb_frame = draw_rectangle_around_face(rgb_frame, face)
                rgb_frame = overlay_aligned_face(rgb_frame, aligned_face)

            refresh_display(rgb_frame)

            exit_requested = detect_q_key_pressed()


def create_face_detector():

    frontal_face_detector = dlib.get_frontal_face_detector()

    def face_detector(rgb_frame):
        gray_frame = rgb_to_gray(rgb_frame)
        return frontal_face_detector(gray_frame, 1)

    return face_detector


@contextmanager
def open_webcam():
    try:
        webcam = cv2.VideoCapture(0)
        yield webcam
    finally:
        webcam.release()
        cv2.destroyAllWindows()


def capture_frame(capture):
    _, frame = capture.read()
    return frame


def rgb_to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def align_face(face, gray_frame, rgb_frame, landmarks_predictor, face_size):
    landmarks = landmarks_predictor(gray_frame, face)
    return dlib.get_face_chip(rgb_frame, landmarks, size=face_size)


def draw_rectangle_around_face(rgb_frame, face):
    x, y, w, h = dlib_rectangle_to_xywh(face)
    cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return rgb_frame


def dlib_rectangle_to_xywh(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def overlay_aligned_face(rgb_frame, aligned_face):
    offset = 10
    shape = aligned_face.shape
    rgb_frame[offset:offset+shape[0],
              offset:offset+shape[1]] = aligned_face
    return rgb_frame


def refresh_display(rgb_frame):
    cv2.imshow('frame', rgb_frame)


def detect_q_key_pressed():
    return cv2.waitKey(1) & 0xFF == ord('q')


def timestamp(name):
    return "%s_%s" \
        % (datetime.now().strftime('%Y-%m-%dT%H-%M-%S'), name)


def record_sample(facebox_width=128):
    face_frames_buffer = []
    face_detector = create_face_detector()

    with open_webcam() as webcam:

        exit_requested = False
        previous_face_rectangle = None
        recording_start_time = None

        while not exit_requested:

            rgb_frame = capture_frame(webcam)
            detected_faces = face_detector(rgb_frame)
            face_rectangle = detected_faces[0] if detected_faces else previous_face_rectangle

            if face_rectangle:
                x, y, w, h = dlib_rectangle_to_xywh(face_rectangle)
                face_frame = cv2.resize(
                    rgb_frame[y:y+h, x:x+w], (facebox_width, facebox_width))
                refresh_display(face_frame)
                face_frames_buffer.append(face_frame)
                previous_face_rectangle = face_rectangle

                if not recording_start_time:
                    recording_start_time = time.time()

            exit_requested = detect_q_key_pressed()

        recording_end_time = time.time()

    return {
        'frames': np.stack(face_frames_buffer),
        'fps':  np.float(len(face_frames_buffer))/(recording_end_time - recording_start_time)
    }


if __name__ == '__main__':
    monitor_heart_rate()
