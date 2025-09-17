import csv
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Optional, Protocol, Dict, List

import numpy as np
import ruptures as rpt
from scipy.signal import savgol_filter

from .landmarks import LandmarksSignalExtractor, _68_INDICES  # uses your existing groups
from .video_utils import get_frames_at_indices, save_frames


# Region labels corresponding to _68_INDICES order in landmarks.py
_REGION_NAMES = [
    "right_eyebrow",
    "left_eyebrow",
    "right_eye",
    "left_eye",
    "nose_bridge",
    "nose_bottom",
    "outer_lips",
    "inner_lips",
    "jawline",
]


class SignalExtractor(Protocol):
    """A protocol for classes that can extract signals from videos"""

    @abstractmethod
    def extract_signal(self, vid_path: Path) -> np.ndarray: ...


class VideoProcessor:
    def __init__(
        self,
        vid_path: Path,
        signal_extractor: Optional[SignalExtractor] = None,
        noise_filterer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        changepoint_detector: Optional[rpt.BottomUp] = None,
    ) -> None:
        self.vid_path = vid_path
        self.signal_extractor = (
            LandmarksSignalExtractor() if signal_extractor is None else signal_extractor
        )

        def default_filter(signal: np.ndarray):
            """Savitzky–Golay filter with window length 17 and polynomial order 13"""
            return savgol_filter(
                signal, window_length=17, polyorder=13, axis=-1, mode="nearest"
            )

        self.noise_filterer = default_filter if noise_filterer is None else noise_filterer
        self.changepoint_detector = (
            rpt.BottomUp(model="rbf", jump=1)
            if changepoint_detector is None
            else changepoint_detector
        )
        self.filtered_signal: Optional[np.ndarray] = None

        # Will be constructed lazily on first compute
        self._region_col_indices: Optional[Dict[str, List[int]]] = None

    # ---------- internal helpers ----------

    def _ensure_filtered_signal(self) -> None:
        """Compute and cache self.filtered_signal once."""
        if self.filtered_signal is None:
            signal = self.signal_extractor.extract_signal(self.vid_path)
            self.filtered_signal = self.noise_filterer(signal)
            # Ensure 2D (n_frames, n_features)
            if self.filtered_signal.ndim == 1:
                self.filtered_signal = self.filtered_signal[:, None]

    def _build_region_column_map(self) -> None:
        """
        Builds a map from region name to column indices in the filtered signal matrix.

        The filtered signal is (n_frames, n_features) where n_features = 2 * (#landmarks used),
        because each landmark contributes (x, y). Landmarks are ordered by ascending landmark index.
        """
        if self._region_col_indices is not None:
            return

        # Landmarks used by the extractor, in ascending Mediapipe index order
        lm_indices_sorted = sorted(self.signal_extractor.landmarks_indices)
        index_to_position = {lm_idx: pos for pos, lm_idx in enumerate(lm_indices_sorted)}

        region_map: Dict[str, List[int]] = {}
        for region_name, region_indices in zip(_REGION_NAMES, _68_INDICES):
            cols: List[int] = []
            for lm_idx in region_indices:
                if lm_idx in index_to_position:
                    p = index_to_position[lm_idx]
                    cols.extend([2 * p, 2 * p + 1])  # x and y columns
            # Only add regions that have at least one landmark present
            if cols:
                region_map[region_name] = cols

        self._region_col_indices = region_map

    # ---------- change-score computations ----------

    def compute_change_scores(self) -> np.ndarray:
        """
        Global change score per frame:
            scores[t] = L2-norm( filtered_signal[t] - filtered_signal[t-1] ), scores[0] = 0.0
        Returns 1D array of length n_frames.
        """
        self._ensure_filtered_signal()
        diffs = np.diff(self.filtered_signal, axis=0)           # (n_frames-1, n_features)
        step_l2 = np.linalg.norm(diffs, axis=1)                 # (n_frames-1,)
        scores = np.concatenate(([0.0], step_l2)).astype(float) # (n_frames,)
        return scores

    def compute_change_scores_by_region(self) -> Dict[str, np.ndarray]:
        """
        Per-region change score per frame (same definition as global, but restricted to region columns).
        Returns:
            dict mapping region_name -> 1D array (length n_frames)
        """
        self._ensure_filtered_signal()
        self._build_region_column_map()
        assert self._region_col_indices is not None

        diffs = np.diff(self.filtered_signal, axis=0)  # (n_frames-1, n_features)

        region_scores: Dict[str, np.ndarray] = {}
        for region_name, cols in self._region_col_indices.items():
            if not cols:
                # If a region has no columns (shouldn't happen with default indices), fill zeros
                scores = np.zeros(self.filtered_signal.shape[0], dtype=float)
            else:
                region_diffs = diffs[:, cols]                   # (n_frames-1, 2*k_region)
                step_l2 = np.linalg.norm(region_diffs, axis=1) # (n_frames-1,)
                scores = np.concatenate(([0.0], step_l2)).astype(float)
            region_scores[region_name] = scores

        return region_scores

    # ---------- main API ----------

    def get_changepoints(self, num_changepoints: int) -> list[int]:
        """
        Computes changepoints from the filtered signal.
        Return values are indices of frames that directly follow changepoints.
        """
        self._ensure_filtered_signal()
        changepoints = self.changepoint_detector.fit_predict(
            signal=self.filtered_signal, n_bkps=num_changepoints
        )[:-1]
        return changepoints

    def select_frames(self, frame_count: int) -> tuple[list[np.ndarray], list[int]]:
        """
        Returns a list of frames and the list of detected changepoints.
        Selected indices = [0] + changepoints (length == frame_count).
        """
        changepoints = (
            self.get_changepoints(num_changepoints=frame_count - 1)
            if frame_count > 1
            else []
        )
        indices = [0] + changepoints
        frames = get_frames_at_indices(vid_path=self.vid_path, indices=indices)
        return frames, changepoints

    def save_frames_to_directory(
        self, output_dir: Path, frames: list[np.ndarray], frame_count: int
    ) -> None:
        """
        Saves frames in:
        <output_dir>/<frame_count>_frames/<video_name>/{0.png,1.png,...}
        """
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
        Appends a row:
            video_file_name, frame_count, [0|i1|i2], global_scores, <per-region columns...>
        where scores are per-frame L2 changes at those indices.
        """
        idx_list = [0] + changepoints
        idx_str = "[" + "|".join(str(cp) for cp in idx_list) + "]"

        row = [self.vid_path.name, frame_count, idx_str]

        # global scores column
        if change_scores_global is not None:
            row.append(self._fmt_scores(change_scores_global, decimals))
        else:
            row.append("")

        # per-region columns in fixed order; omit regions that had no landmarks
        if change_scores_regions is not None:
            for region_name in _REGION_NAMES:
                if change_scores_regions.get(region_name) is not None:
                    row.append(self._fmt_scores(change_scores_regions[region_name], decimals))
                else:
                    row.append("")  # in case region absent
        # write
        with csv_path.open(mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(tuple(row))

    def select_frames_and_save_data(
        self, frame_count: int, output_dir: Path, csv_path: Path
    ) -> None:
        frames, changepoints = self.select_frames(frame_count)
        self.save_frames_to_directory(output_dir, frames, frame_count)

        # Global scores: compute once
        global_scores = self.compute_change_scores()
        indices = [0] + changepoints
        selected_global = [float(global_scores[i]) for i in indices]

        # Per-region scores
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
        """
        Saves selected frames for each requested count and appends rows to output/changepoints.csv.
        """
        for frame_count in frame_counts:
            self.select_frames_and_save_data(
                frame_count=frame_count,
                output_dir=output_dir,
                csv_path=output_dir / "changepoints.csv",
            )
