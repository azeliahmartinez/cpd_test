import csv
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np
import ruptures as rpt
from scipy.signal import savgol_filter

from .landmarks import LandmarksSignalExtractor
from .video_utils import get_frames_at_indices, save_frames


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
            """Savitsky-Golay filter with window length of 17 and polynomial order of 13"""

            return savgol_filter(
                signal, window_length=17, polyorder=13, axis=-1, mode="nearest"
            )

        self.noise_filterer = (
            default_filter if noise_filterer is None else noise_filterer
        )
        self.changepoint_detector = (
            rpt.BottomUp(model="rbf", jump=1)  # bottom-up change detection algorithm
            if changepoint_detector is None
            else changepoint_detector
        )
        self.filtered_signal = None

    def get_changepoints(self, num_changepoints: int) -> list[int]:
        """
        Computes the changepoints of the video by extracting a signal from it using the instance's SignalExtractor,
        filters it using the instance's noise filtering function, then computes the changepoints from that filtered signal.
        Stores the filtered signal the first time it is called (regardless of what value is passed as num_changepoints)
        to avoid having to recompute it.

        The return values may be interpreted as the indices of the frames that directly follow the changepoints.

        Returns:
            The computed changepoints
        """

        if self.filtered_signal is None:
            signal = self.signal_extractor.extract_signal(self.vid_path)
            self.filtered_signal = self.noise_filterer(signal)

        changepoints = self.changepoint_detector.fit_predict(
            signal=self.filtered_signal, n_bkps=num_changepoints
        )[:-1]
        return changepoints

    def select_frames(self, frame_count: int) -> tuple[list[np.ndarray], list[int]]:
        """
        Returns:
            A list of frame_count frames containing the first frame and the frames at frame_count-1 detected changepoints
            and the list of detected changepoints
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
        Saves frames in the following format:

        <output_dir>/
            <frame_count>_frames/
                <video_name>/
                    0.png
                    1.png
                    .
                    .
                    .
        """
        # create subdirectory if it does not yet exist
        subdir = output_dir / f"{frame_count}_frames" / self.vid_path.stem
        if not subdir.exists():
            Path.mkdir(subdir, parents=True)

        # save frames
        filenames = [f"{i}.png" for i in range(frame_count)]
        save_frames(output_dir=subdir, frames=frames, filenames=filenames)

    def save_changepoints_to_csv(
        self, changepoints: list[int], csv_path: Path, frame_count: int
    ) -> None:
        """
        Appends a row to the csv in the form:

            video_file_name,frame_count,[0|changepoints]

        where \"changepoints\" are the changepoint indices separated by |
        """

        changepoints_str = "[" + "|".join(str(cp) for cp in [0] + changepoints) + "]"

        # save changepoint data to csv
        with csv_path.open(mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(
                (
                    self.vid_path.name,
                    frame_count,
                    changepoints_str,
                )
            )

    def select_frames_and_save_data(
        self, frame_count: int, output_dir: Path, csv_path: Path
    ) -> None:
        frames, changepoints = self.select_frames(frame_count)
        self.save_frames_to_directory(output_dir, frames, frame_count)
        self.save_changepoints_to_csv(
            csv_path=csv_path,
            changepoints=changepoints,
            frame_count=frame_count,
        )

    def process(self, frame_counts: list[int], output_dir: Path) -> None:
        """
        Saves the selected frames for all counts in frame_count. Also stores the changepoints in a csv file.
        Files are stored in the following directory structure:

        <output_dir>/
            changepoints.csv
            <frame_counts[0]>_frames/
                <video_name>/
                    0.png
                    1.png
                    .
                    .
                    .
            <frame_counts[1]>_frames/
                <video_name>/
                    0.png
                    1.png
                    .
                    .
                    .
            .
            .
            .
        """

        for frame_count in frame_counts:
            self.select_frames_and_save_data(
                frame_count=frame_count,
                output_dir=output_dir,
                csv_path=output_dir / "changepoints.csv",
            )
