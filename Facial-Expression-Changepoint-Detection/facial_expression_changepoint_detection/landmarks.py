from pathlib import Path
from typing import Optional

import cv2 as cv
import mediapipe as mp
import numpy as np

from .video_utils import get_frames

_MP_FACE_LANDMARKER_OPTIONS = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(
            Path(__file__).parent / "pretrained_models" / "face_landmarker.task"
        )
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0,
    min_face_presence_confidence=0,
    min_tracking_confidence=0,
)

_68_INDICES = [
    {46, 53, 52, 65, 55},  # right eyebrow
    {285, 295, 282, 283, 276},  # left eyebrow
    {33, 160, 158, 144, 153, 133},  # right eye
    {362, 385, 387, 380, 373, 263},  # left eye
    {6, 197, 195, 5},  # nose bridge
    {98, 97, 2, 326, 327},  # nose bottom
    {61, 40, 37, 0, 267, 270, 91, 84, 17, 314, 321, 291},  # outer lips
    {78, 81, 13, 311, 178, 14, 402, 308},  # inner lips
    {
        127,
        234,
        93,
        132,
        58,
        172,
        150,
        176,
        152,
        400,
        379,
        397,
        288,
        361,
        323,
        454,
        356,
    },  # jawline
]

_DEFAULT_INDICES = set().union(*_68_INDICES)


def preprocess_for_mediapipe(
    frame: np.ndarray, timestamp: float
) -> tuple[mp.Image, int]:
    """
    Converts the frame and timestamp to the format Mediapipe expects.
    The frame is conveted to RGB (opencv reads frames in BGR), and an mp.Image object is created from it.
    The timestamp is rounded to an integer.

    Returns:
        The frame and timestamp in a format ready to be consumed by a Mediapipe landmarker
    """

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return mp_img, round(timestamp)


def new_landmarker() -> mp.tasks.vision.FaceLandmarker:
    """
    Convenience function that returns a new instance of the Mediapipe face landmarker
    """

    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
        _MP_FACE_LANDMARKER_OPTIONS
    )
    return landmarker


class LandmarksSignalExtractor:
    def __init__(self, indices: Optional[set[int]] = None):
        """
        Parameters:
            indices:
                The set of indices describing which of the Mediapipe face landmarker results
                will be part of the return value of the get_facial_landmarks method
        """

        self.landmarks_indices = indices if indices else _DEFAULT_INDICES

    def get_facial_landmarks(
        self,
        frame: np.ndarray,
        timestamp: float,
        face_landmarker: mp.tasks.vision.FaceLandmarker,
    ) -> np.ndarray:
        """
        Returns:
            The facial landmarks (x and y coordinates), flattened to a 1D array

        Each call with the same face landmarker must be passed a larger timestamp than the previous
        """

        mp_img, ts = preprocess_for_mediapipe(frame, timestamp)

        # call mediapipe model to get landmark coordinates
        mp_results = face_landmarker.detect_for_video(mp_img, ts)

        facial_landmarks = mp_results.face_landmarks[0]
        facial_landmarks = np.array(
            [
                [landmark_container.x, landmark_container.y]
                for i, landmark_container in enumerate(facial_landmarks)
                if i in self.landmarks_indices
            ]
        )
        return facial_landmarks.ravel()

    def extract_signal(self, vid_path: Path) -> np.ndarray:
        """
        Returns:
            An n x m array, where n is the number of frames in the video and n is the number of landmark points
        """

        with new_landmarker() as face_landmarker:
            signal = np.vstack(
                [
                    self.get_facial_landmarks(
                        frame=frame,
                        timestamp=timestamp,
                        face_landmarker=face_landmarker,
                    )
                    for frame, timestamp in get_frames(vid_path=vid_path)
                ]
            )
        return signal
