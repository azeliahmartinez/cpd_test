# facial_expression_changepoint_detection/raw_export.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
from collections.abc import Iterable
import csv

import cv2 as cv
import numpy as np
import mediapipe as mp

from .landmarks import new_landmarker, preprocess_for_mediapipe, _68_INDICES
from .pose_extractor import _POSE_OPTS

# BlazePose indices we care about
# Left arm: shoulder(11), elbow(13), wrist(15)
LEFT_ARM = [11, 13, 15]
# Right arm: shoulder(12), elbow(14), wrist(16)
RIGHT_ARM = [12, 14, 16]
# Torso: left_shoulder(11), right_shoulder(12), left_hip(23), right_hip(24)
TORSO = [11, 12, 23, 24]

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _write_csv(path: Path, header: List[str], rows: List[List[float | int | str]]) -> None:
    _ensure_dir(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def _sorted_flat_indices(maybe_nested) -> list[int]:
    """Flatten ints from nested structures (dicts, sets, lists, tuples, numpy arrays), then sort."""
    acc: set[int] = set()

    def _add(obj):
        if isinstance(obj, int):
            acc.add(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _add(v)
        elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            for x in obj:
                _add(x)
        else:
            raise TypeError(f"_68_INDICES contains unsupported element type: {type(obj)}")

    _add(maybe_nested)
    return sorted(acc)

def export_raw_landmarks_from_frames(
    video_path: Path,
    frames_bgr: List[np.ndarray],
    frame_indices: List[int],
    out_root: Path,
    n_frames_label: int,
) -> Dict[str, Path]:
    """
    Writes two CSVs (face-68 & pose limited to LeftArm/RightArm/Torso) with raw pixel coordinates.
    Returns dict {"face": <path>, "pose": <path>} for written files (if any).
    """
    dst_dir = out_root / "raw_landmarks" / f"{n_frames_label}_frames" / video_path.stem
    dst_dir.mkdir(parents=True, exist_ok=True)

    # strictly increasing timestamps for Tasks API
    ts_ms = [int(i * 33) for i in range(len(frames_bgr))]

    # Prepare accumulators
    face_header: Optional[List[str]] = None
    face_rows: List[List[float | int | str]] = []

    pose_header: Optional[List[str]] = None
    pose_rows: List[List[float | int | str]] = []

    with new_landmarker() as face_lm, mp.tasks.vision.PoseLandmarker.create_from_options(_POSE_OPTS) as pose_lm:
        for k, (frame, ts) in enumerate(zip(frames_bgr, ts_ms)):
            h, w = frame.shape[:2]

            # ---- Face: 68 points in pixels ----
            mp_img, ts_face = preprocess_for_mediapipe(frame, ts)
            face_res = face_lm.detect_for_video(mp_img, ts_face)
            if getattr(face_res, "face_landmarks", None):
                lm = face_res.face_landmarks[0]  # first face
                all_xy = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
                # slice to your 68 subset
                flat68 = _sorted_flat_indices(_68_INDICES)
                idx = np.array(flat68, dtype=np.int32)
                xy = all_xy[idx]


                if face_header is None:
                    face_header = (
                        ["video", "frame_index"] +
                        [f"f{j}_x" for j in range(68)] +
                        [f"f{j}_y" for j in range(68)]
                    )
                face_rows.append([video_path.name, frame_indices[k]] +
                                 xy[:, 0].astype(float).tolist() +
                                 xy[:, 1].astype(float).tolist())

            # ---- Pose: only LeftArm, RightArm, Torso (x,y,visibility) ----
            pose_res = pose_lm.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv.cvtColor(frame, cv.COLOR_BGR2RGB)),
                ts
            )
            if getattr(pose_res, "pose_landmarks", None):
                lm = pose_res.pose_landmarks[0]  # first person
                # Convert to pixel arrays
                px = np.array([p.x * w for p in lm], dtype=np.float32)
                py = np.array([p.y * h for p in lm], dtype=np.float32)
                pv = np.array([p.visibility for p in lm], dtype=np.float32)

                # Extract groups in fixed order
                def pick(arr, indices): return arr[indices].tolist()

                la_x, la_y, la_v = pick(px, LEFT_ARM), pick(py, LEFT_ARM), pick(pv, LEFT_ARM)
                ra_x, ra_y, ra_v = pick(px, RIGHT_ARM), pick(py, RIGHT_ARM), pick(pv, RIGHT_ARM)
                t_x,  t_y,  t_v  = pick(px, TORSO),    pick(py, TORSO),    pick(pv, TORSO)

                if pose_header is None:
                    pose_header = ["video", "frame_index"] \
                        + [f"la{j}_x" for j in range(len(LEFT_ARM))] \
                        + [f"ra{j}_x" for j in range(len(RIGHT_ARM))] \
                        + [f"t{j}_x"  for j in range(len(TORSO))] \
                        + [f"la{j}_y" for j in range(len(LEFT_ARM))] \
                        + [f"ra{j}_y" for j in range(len(RIGHT_ARM))] \
                        + [f"t{j}_y"  for j in range(len(TORSO))] \
                        + [f"la{j}_vis" for j in range(len(LEFT_ARM))] \
                        + [f"ra{j}_vis" for j in range(len(RIGHT_ARM))] \
                        + [f"t{j}_vis"  for j in range(len(TORSO))]

                pose_rows.append(
                    [video_path.name, frame_indices[k]]
                    + la_x + ra_x + t_x
                    + la_y + ra_y + t_y
                    + la_v + ra_v + t_v
                )

    written: Dict[str, Path] = {}

    if face_rows and face_header:
        face_csv = dst_dir / f"{video_path.stem}_face68_{n_frames_label}f.csv"
        _write_csv(face_csv, face_header, face_rows)
        written["face"] = face_csv

    if pose_rows and pose_header:
        pose_csv = dst_dir / f"{video_path.stem}_upper_body_{n_frames_label}f.csv"
        _write_csv(pose_csv, pose_header, pose_rows)
        written["pose"] = pose_csv

    return written
