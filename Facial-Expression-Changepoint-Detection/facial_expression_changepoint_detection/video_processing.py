import csv
from pathlib import Path
from typing import Callable, Optional, Protocol, Dict, List, Sequence
from .raw_export import export_raw_landmarks_from_frames


import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.signal import savgol_filter

from .landmarks import LandmarksSignalExtractor, _68_INDICES
from .pose_extractor import PoseSignalExtractor
from .video_utils import get_frames_at_indices, save_frames

# Face region labels in the same order as _68_INDICES
_FACE_REGION_NAMES = [
    "right_eyebrow", "left_eyebrow",
    "right_eye", "left_eye",
    "nose_bridge", "nose_bottom",
    "outer_lips", "inner_lips",
    "jawline",
]

# Pose region labels (based on Pose keypoints: shoulders=11/12, elbows=13/14, wrists=15/16, hips=23/24)
_POSE_REGION_NAMES = ["left_arm", "right_arm", "torso"]


class SignalExtractor(Protocol):
    def extract_signal(self, vid_path: Path) -> np.ndarray: ...


class VideoProcessor:
    def __init__(
        self,
        vid_path: Path,
        signal_extractor: Optional[SignalExtractor] = None,   # kept for backward-compat
        noise_filterer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        changepoint_detector: Optional[rpt.BottomUp] = None,
        signal_extractors: Optional[Sequence[SignalExtractor]] = None,  # preferred
    ) -> None:
        self.vid_path = vid_path

        if signal_extractors is not None:
            self.signal_extractors = list(signal_extractors)
        else:
            # Default: face + pose
            if signal_extractor is not None:
                self.signal_extractors = [signal_extractor]
            else:
                self.signal_extractors = [LandmarksSignalExtractor(), PoseSignalExtractor()]

        def default_filter(signal: np.ndarray):
            """
            Savitzky–Golay filter. Window must be odd and <= T.
            If T is too small, fall back to identity (no smoothing).
            """
            sig = signal
            T = sig.shape[0]
            if T < 3:
                return sig
            # choose odd window <= T, at least 3
            win = min(17, T if T % 2 == 1 else T - 1)
            if win < 3:  # still too small
                return sig
            poly = min(13, win - 1)
            return savgol_filter(sig, window_length=win, polyorder=poly, axis=0, mode="nearest")

        self.noise_filterer = default_filter if noise_filterer is None else noise_filterer
        self.changepoint_detector = (
            rpt.BottomUp(model="rbf", jump=1) if changepoint_detector is None else changepoint_detector
        )

        self.filtered_signal: Optional[np.ndarray] = None

        # bookkeeping for column maps
        self._face_offset: int = 0
        self._pose_offset: int = 0
        self._face_indices_sorted: Optional[List[int]] = None
        self._pose_indices_sorted: Optional[List[int]] = None
        self._region_col_indices: Optional[Dict[str, List[int]]] = None  # face+pose regions

    # ---------- internal helpers ----------

    def _ensure_filtered_signal(self) -> None:
        if self.filtered_signal is not None:
            return

        # Extract per-frame vectors for each extractor
        parts = [se.extract_signal(self.vid_path) for se in self.signal_extractors]
        # Align by shortest T
        T = min(p.shape[0] for p in parts)
        parts = [p[:T] for p in parts]

        # Record offsets and index orders for region mapping
        # Assume at most one LandmarksSignalExtractor and one PoseSignalExtractor present
        face_dim = 0
        pose_dim = 0

        for se, p in zip(self.signal_extractors, parts):
            if isinstance(se, LandmarksSignalExtractor):
                self._face_offset = 0  # face first if present
                face_dim = p.shape[1]
                self._face_indices_sorted = sorted(se.landmarks_indices)
            elif isinstance(se, PoseSignalExtractor):
                pose_dim = p.shape[1]

        # If both present and face is first in list, pose starts after face
        # If only pose present, offset is 0
        if face_dim > 0 and pose_dim > 0:
            # Ensure order in self.signal_extractors determines concat order
            if isinstance(self.signal_extractors[0], LandmarksSignalExtractor):
                self._face_offset = 0
                self._pose_offset = face_dim
            else:
                self._pose_offset = 0
                self._face_offset = pose_dim

        # Concatenate
        signal = np.concatenate(parts, axis=1)  # (T, d_face + d_pose) or (T, d_single)

        # Filter (denoise)
        self.filtered_signal = self.noise_filterer(signal)
        if self.filtered_signal.ndim == 1:
            self.filtered_signal = self.filtered_signal[:, None]

    def _build_region_column_map(self) -> None:
        if self._region_col_indices is not None:
            return

        region_map: Dict[str, List[int]] = {}

        # Face regions (if face present)
        if self._face_indices_sorted is not None:
            idx_to_pos = {lm_idx: pos for pos, lm_idx in enumerate(self._face_indices_sorted)}
            for region_name, region_indices in zip(_FACE_REGION_NAMES, _68_INDICES):
                cols: List[int] = []
                for lm_idx in region_indices:
                    if lm_idx in idx_to_pos:
                        p = idx_to_pos[lm_idx]
                        cols.extend([self._face_offset + 2 * p, self._face_offset + 2 * p + 1])
                if cols:
                    region_map[region_name] = cols

        # Pose regions (if pose present)
        # We infer which pose indices were used by inspecting the PoseSignalExtractor
        pose_extractors = [se for se in self.signal_extractors if isinstance(se, PoseSignalExtractor)]
        if pose_extractors:
            pose_se = pose_extractors[0]
            self._pose_indices_sorted = list(pose_se.indices)  # as provided
            ppos = {lm_idx: pos for pos, lm_idx in enumerate(self._pose_indices_sorted)}

            def pose_cols(for_ids: List[int]) -> List[int]:
                cols: List[int] = []
                for lm_idx in for_ids:
                    if lm_idx in ppos:
                        j = ppos[lm_idx]
                        cols.extend([self._pose_offset + 2 * j, self._pose_offset + 2 * j + 1])
                return cols

            # Define regions
            left_arm = pose_cols([11, 13, 15])   # L shoulder, elbow, wrist
            right_arm = pose_cols([12, 14, 16])  # R shoulder, elbow, wrist
            torso = pose_cols([11, 12, 23, 24])  # shoulders + hips

            if left_arm:  region_map["left_arm"] = left_arm
            if right_arm: region_map["right_arm"] = right_arm
            if torso:     region_map["torso"] = torso

        self._region_col_indices = region_map

    # ---------- change-score computations ----------

    def compute_change_scores(self) -> np.ndarray:
        """
        Global change score per frame:
        scores[t] = || filtered[t] - filtered[t-1] ||_2 ; scores[0]=0
        """
        self._ensure_filtered_signal()
        diffs = np.diff(self.filtered_signal, axis=0)
        step_l2 = np.linalg.norm(diffs, axis=1)
        return np.concatenate(([0.0], step_l2)).astype(float)

    def compute_change_scores_by_region(self) -> Dict[str, np.ndarray]:
        """
        Per-region change score per frame (same L2 step, restricted to region columns).
        """
        self._ensure_filtered_signal()
        self._build_region_column_map()
        assert self._region_col_indices is not None

        diffs = np.diff(self.filtered_signal, axis=0)

        region_scores: Dict[str, np.ndarray] = {}
        for region_name, cols in self._region_col_indices.items():
            if not cols:
                region_scores[region_name] = np.zeros(self.filtered_signal.shape[0], dtype=float)
            else:
                region_diffs = diffs[:, cols]
                step_l2 = np.linalg.norm(region_diffs, axis=1)
                region_scores[region_name] = np.concatenate(([0.0], step_l2)).astype(float)
        return region_scores

    # ---------- main API ----------

    def get_changepoints(self, num_changepoints: int) -> list[int]:
        self._ensure_filtered_signal()
        changepoints = self.changepoint_detector.fit_predict(
            signal=self.filtered_signal, n_bkps=num_changepoints
        )[:-1]
        return changepoints

    def select_frames(self, frame_count: int) -> tuple[list[np.ndarray], list[int]]:
        changepoints = self.get_changepoints(num_changepoints=max(0, frame_count - 1))
        indices = [0] + (changepoints if frame_count > 1 else [])
        frames = get_frames_at_indices(vid_path=self.vid_path, indices=indices)
        return frames, changepoints

    def save_frames_to_directory(
        self, output_dir: Path, frames: list[np.ndarray], frame_count: int
    ) -> None:
        subdir = output_dir / f"{frame_count}_frames" / self.vid_path.stem
        if not subdir.exists():
            Path.mkdir(subdir, parents=True)
        filenames = [f"{i}.png" for i in range(frame_count)]
        save_frames(output_dir=subdir, frames=frames, filenames=filenames)

    def _fmt_scores(self, values: list[float], decimals: int = 4) -> str:
        return "[" + "|".join(f"{v:.{decimals}f}" for v in values) + "]"

    def save_changepoints_to_csv(
        self,
        changepoints: list[int],
        csv_path: Path,
        frame_count: int,
        change_scores_global: Optional[List[float]] = None,
        change_scores_regions: Optional[Dict[str, List[float]]] = None,
        decimals: int = 4,
    ) -> None:
        """
        One row per (video, frame_count) with fixed columns:
        video, frame_count, frame_indices, change_scores_global,
        <face regions...>, <pose regions...>
        """
        idx_list = [0] + changepoints
        idx_str = "[" + "|".join(str(cp) for cp in idx_list) + "]"

        row = {
            "video": self.vid_path.name,
            "frame_count": frame_count,
            "frame_indices": idx_str,
            "change_scores_global": self._fmt_scores(change_scores_global or [], decimals)
                                       if change_scores_global is not None else "",
        }

        # Fill face region columns
        for rn in _FACE_REGION_NAMES:
            key = rn
            if change_scores_regions and rn in change_scores_regions:
                row[key] = self._fmt_scores(change_scores_regions[rn], decimals)
            else:
                row[key] = ""

        # Fill pose region columns
        for rn in _POSE_REGION_NAMES:
            key = rn
            if change_scores_regions and rn in change_scores_regions:
                row[key] = self._fmt_scores(change_scores_regions[rn], decimals)
            else:
                row[key] = ""

        # Write with stable column order
        fieldnames = [
            "video", "frame_count", "frame_indices", "change_scores_global",
            *_FACE_REGION_NAMES, *_POSE_REGION_NAMES
        ]
        row_df = pd.DataFrame([row]).reindex(columns=fieldnames)

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.exists():
            prev_cols = list(pd.read_csv(csv_path, nrows=0).columns)
            if prev_cols != fieldnames:
                # Reset file if schema changed
                row_df.to_csv(csv_path, index=False)
            else:
                row_df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            row_df.to_csv(csv_path, index=False)

    def select_frames_and_save_data(
        self, frame_count: int, output_dir: Path, csv_path: Path
    ) -> None:
        frames, changepoints = self.select_frames(frame_count)
        self.save_frames_to_directory(output_dir, frames, frame_count)

        # Global & regional scores
        global_scores_all = self.compute_change_scores()
        indices = [0] + changepoints

        # write raw pixel landmarks for these selected frames
        export_raw_landmarks_from_frames(
            video_path=self.vid_path,
            frames_bgr=frames,
            frame_indices=indices,        # same order as frames
            out_root=output_dir,
            n_frames_label=frame_count,
        )

        selected_global = [float(global_scores_all[i]) for i in indices]

        region_scores_all = self.compute_change_scores_by_region()
        selected_region_scores: Dict[str, List[float]] = {
            region: [float(scores[i]) for i in indices]
            for region, scores in region_scores_all.items()
        }

        self.save_changepoints_to_csv(
            csv_path=csv_path,
            changepoints=changepoints,
            frame_count=frame_count,
            change_scores_global=selected_global,
            change_scores_regions=selected_region_scores,
        )

    def process(self, frame_counts: list[int], output_dir: Path) -> None:
        for frame_count in frame_counts:
            self.select_frames_and_save_data(
                frame_count=frame_count,
                output_dir=output_dir,
                csv_path=output_dir / "changepoints.csv",
            )