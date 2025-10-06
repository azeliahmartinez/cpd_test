from pathlib import Path
from typing import Optional

import cv2 as cv
import mediapipe as mp
import numpy as np

from .video_utils import get_frames

# Compact subset of body joints (Pose model has 33):
# nose, L/R shoulder, L/R elbow, L/R wrist, L/R hip
_POSE_LM_DEFAULT = [0, 11, 12, 13, 14, 15, 16, 23, 24]

_POSE_OPTS = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(
            Path(__file__).parent / "pretrained_models" / "pose_landmarker.task"
        )
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.2,
    min_pose_presence_confidence=0.2,
    min_tracking_confidence=0.2,
)

def _mp_img(frame: np.ndarray) -> mp.Image:
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

class PoseSignalExtractor:
    """
    Returns per-frame pose landmarks as **pixel coordinates** (x_px, y_px) for
    selected indices, flattened. No normalization; ML can learn distances.
    """
    def __init__(self, indices: Optional[list[int]] = None):
        self.indices = indices if indices is not None else _POSE_LM_DEFAULT
        self._vec_len = 2 * len(self.indices)

    def extract_signal(self, vid_path: Path) -> np.ndarray:
        rows = []
        with mp.tasks.vision.PoseLandmarker.create_from_options(_POSE_OPTS) as pose:
            for frame, ts_ms in get_frames(vid_path):
                h, w = frame.shape[:2]
                res = pose.detect_for_video(_mp_img(frame), round(ts_ms))
                if not getattr(res, "pose_landmarks", None):
                    rows.append(np.full((self._vec_len,), np.nan, dtype=float))
                    continue

                lm = res.pose_landmarks[0]  # 33 landmarks
                # Convert normalized coords to **pixels**
                xy = np.array([[p.x * w, p.y * h] for p in lm], dtype=float)  # (33,2)
                sel = xy[self.indices]                                         # (M,2)
                rows.append(sel.ravel())                                       # (2M,)

        sig = np.vstack(rows)  # (T, 2M)

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
