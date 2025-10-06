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
        127, 234, 93, 132, 58, 172, 150, 176, 152,
        400, 379, 397, 288, 361, 323, 454, 356
    },  # jawline
]

_DEFAULT_INDICES = set().union(*_68_INDICES)


def preprocess_for_mediapipe(
    frame: np.ndarray, timestamp: float
) -> tuple[mp.Image, int]:
    """Convert BGR frame to an mp.Image (RGB) and round timestamp to int ms."""
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return mp_img, round(timestamp)


def new_landmarker() -> mp.tasks.vision.FaceLandmarker:
    return mp.tasks.vision.FaceLandmarker.create_from_options(_MP_FACE_LANDMARKER_OPTIONS)


class LandmarksSignalExtractor:
    """
    Returns per-frame face landmarks as **pixel coordinates** (x_px, y_px) for
    the selected Mediapipe indices, flattened into a 1D vector.
    """
    def __init__(self, indices: Optional[set[int]] = None):
        self.landmarks_indices = indices if indices else _DEFAULT_INDICES
        self._vec_len = 2 * len(self.landmarks_indices)  # x,y per landmark

    def get_facial_landmarks(
        self,
        frame: np.ndarray,
        timestamp: float,
        face_landmarker: mp.tasks.vision.FaceLandmarker,
    ) -> np.ndarray:
        """
        Convert Mediapipe's normalized coords to **pixels** using the current
        frame width/height. If no face is detected, return NaNs (later
        forward-filled).
        """
        h, w = frame.shape[:2]
        mp_img, ts = preprocess_for_mediapipe(frame, timestamp)
        mp_results = face_landmarker.detect_for_video(mp_img, ts)

        if not mp_results.face_landmarks:
            return np.full((self._vec_len,), np.nan, dtype=float)

        facial_landmarks = mp_results.face_landmarks[0]
        rows = []
        for i, lm in enumerate(facial_landmarks):
            if i in self.landmarks_indices:
                # convert normalized [0..1] to **pixels**
                rows.append([lm.x * w, lm.y * h])
        if not rows:
            return np.full((self._vec_len,), np.nan, dtype=float)
        arr = np.array(rows, dtype=float)
        return arr.ravel()

    def extract_signal(self, vid_path: Path) -> np.ndarray:
        rows = []
        with new_landmarker() as face_landmarker:
            for frame, timestamp in get_frames(vid_path=vid_path):
                rows.append(
                    self.get_facial_landmarks(
                        frame=frame,
                        timestamp=timestamp,
                        face_landmarker=face_landmarker,
                    )
                )
        sig = np.vstack(rows)  # (T, 2K) pixel coords

        # Forward-fill NaNs per column, fallback zeros if column is entirely NaN
        if np.isnan(sig).any():
            for j in range(sig.shape[1]):
                col = sig[:, j]
                valid = np.where(~np.isnan(col))[0]
                if valid.size == 0:
                    sig[:, j] = 0.0
                else:
                    last = col[valid[0]]
                    for i in range(sig.shape[0]):
                        if np.isnan(col[i]):
                            col[i] = last
                        else:
                            last = col[i]
                    sig[:, j] = col
        return sig
