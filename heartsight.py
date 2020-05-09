from contextlib import contextmanager
from datetime import datetime
import numpy as np
import cv2
import dlib


def record_sample():

    with open_video_resource() as webcam:

        frames_buffer = []
        exit_requested = False
        recording_start_time = datetime.now()

        while not exit_requested:

            rgb_frame = capture_frame(webcam)
            frames_buffer.append(rgb_frame)
            refresh_display(rgb_frame)
            exit_requested = detect_q_key_pressed()

        recording_end_time = datetime.now()
        elapsed_time = (recording_end_time -
                        recording_start_time).total_seconds()

    return {
        'start_time': recording_start_time,
        'frames': np.stack(frames_buffer),
        'fps':  np.float(len(frames_buffer))/elapsed_time
    }


@contextmanager
def open_video_resource(source=0):

    try:
        resource = cv2.VideoCapture(source)
        yield resource
    finally:
        resource.release()
        cv2.destroyAllWindows()


def capture_frame(source):
    _, frame = source.read()
    return frame


def detect_q_key_pressed():
    return cv2.waitKey(1) & 0xFF == ord('q')


def read_video_file(filepath, max_frames_nb=10000):

    with open_video_resource(source=filepath) as video:
        frames_buffer = []
        fps = video.get(cv2.CAP_PROP_FPS)
        frames_nb = 0

        while video.isOpened() and frames_nb < max_frames_nb:
            rgb_frame = capture_frame(video)
            frames_buffer.append(rgb_frame)
            frames_nb += 1

    return {
        'frames': np.stack(frames_buffer),
        'fps':  fps
    }


def extract_face_frames(frames, facebox_width=128):

    face_frames_buffer = []
    face_detector = create_face_detector()
    previous_face_rectangle = None

    for frame in frames:
        detected_faces = face_detector(frame)
        face_rectangle = detected_faces[0] if detected_faces else previous_face_rectangle

        if face_rectangle:
            face_frame = dlib.resize_image(
                dlib.sub_image(frame, face_rectangle), facebox_width, facebox_width)
            face_frames_buffer.append(face_frame)
            previous_face_rectangle = face_rectangle

    face_frames = np.stack(face_frames_buffer) if face_frames_buffer else None
    return face_frames


def create_face_detector():

    frontal_face_detector = dlib.get_frontal_face_detector()

    def face_detector(rgb_frame):
        gray_frame = rgb_to_gray(rgb_frame)
        return frontal_face_detector(gray_frame, 1)

    return face_detector


def rgb_to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def monitor_heart_rate():

    with open_video_resource() as webcam:
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


def timestamp(name):
    return "%s_%s" \
        % (datetime.now().strftime('%Y-%m-%dT%H-%M-%S'), name)


if __name__ == '__main__':
    pass
