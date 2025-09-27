# facial_expression_changepoint_detection/hand_extractor.py
from pathlib import Path
import mediapipe as mp
import numpy as np

from .video_utils import get_frames
from .landmarks import preprocess_for_mediapipe

_HAND_LANDMARK_COUNT = 21  # per hand

_HAND_OPTS = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(
            Path(__file__).parent / "pretrained_models" / "hand_landmarker.task"
        )
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    min_tracking_confidence=0.2,
)

class HandSignalExtractor:
    """
    Per frame vector layout:
      [ LeftHand(21*(x,y)) , RightHand(21*(x,y)) ]  -> 84 floats
    If one hand is missing in a frame, that half is NaN (later forward-filled to last seen).
    """
    def __init__(self):
        self._vec_len = 2 * _HAND_LANDMARK_COUNT * 2  # 2 hands * 21 pts * (x,y)

    def _row_from_results(self, res, w: int, h: int) -> np.ndarray:
        left  = np.full((_HAND_LANDMARK_COUNT, 2), np.nan, dtype=float)
        right = np.full((_HAND_LANDMARK_COUNT, 2), np.nan, dtype=float)

        lms        = getattr(res, "hand_landmarks", None) or []
        handedness = getattr(res, "handedness", None)

        def top_label(hcls):
            if not hcls: return "Unknown"
            best = max(hcls, key=lambda c: getattr(c, "score", 0.0))
            return getattr(best, "category_name", "Unknown")

        for i, lm21 in enumerate(lms):
            label = "Unknown"
            if handedness and i < len(handedness):
                label = top_label(handedness[i])
            xy = np.array([[p.x * w, p.y * h] for p in lm21], dtype=float)
            if str(label).lower().startswith("left"):
                left = xy
            elif str(label).lower().startswith("right"):
                right = xy
            else:
                if np.isnan(left).all():
                    left = xy
                else:
                    right = xy

        row = np.concatenate([left.ravel(), right.ravel()], axis=0)
        if row.size != self._vec_len:
            out = np.full((self._vec_len,), np.nan, dtype=float)
            out[:min(out.size, row.size)] = row[:min(out.size, row.size)]
            return out
        return row

    def extract_signal(self, vid_path: Path) -> np.ndarray:
        rows = []
        with mp.tasks.vision.HandLandmarker.create_from_options(_HAND_OPTS) as model:
            for frame, ts_ms in get_frames(vid_path):
                h, w = frame.shape[:2]
                mp_img, ts = preprocess_for_mediapipe(frame, ts_ms)
                res = model.detect_for_video(mp_img, ts)
                rows.append(self._row_from_results(res, w, h))

        sig = np.vstack(rows).astype(float)

        # forward-fill NaNs column-wise (same policy as face/pose)
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
