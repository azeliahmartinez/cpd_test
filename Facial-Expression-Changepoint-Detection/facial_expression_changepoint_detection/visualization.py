from pathlib import Path
from typing import Optional

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .video_processing import VideoProcessor
from .video_utils import get_frames


def compute_vertical_bounds(
    signal: np.ndarray, margin_size: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    minimum_values, maxixum_values = signal.min(axis=0), signal.max(axis=0)
    margins = (maxixum_values - minimum_values) * margin_size
    lower_bounds, upper_bounds = minimum_values - margins, maxixum_values + margins
    return lower_bounds, upper_bounds


class Animation:
    def __init__(
        self,
        vid_path: Path,
        video_processor: VideoProcessor,
        frame_count: int,
        title: str = "",
        ylabels: Optional[list[str]] = None,
    ):
        self.vid_path = vid_path

        # extract signal and CPD
        signal = None
        video_processor._ensure_filtered_signal()
        self.filtered_signal = video_processor.filtered_signal
        changepoints = video_processor.get_changepoints(num_changepoints=frame_count)

        if self.filtered_signal.ndim == 1:
            self.filtered_signal = self.filtered_signal[:, None]

        # init figs
        self.fig = plt.figure(figsize=(12, 6))
        self.fig.suptitle(title)
        signals_fig, img_fig = self.fig.subfigures(
            1, 2, width_ratios=[6.5, 3.5], wspace=0, hspace=0
        )

        self._init_signal_plots(signals_fig, changepoints=changepoints, ylabels=ylabels)
        self._init_img_plot(img_fig)
        self.plots = self.signal_plots + [self.img_plot]

    def _init_signal_plots(
        self, subfig, changepoints: list[int], ylabels: Optional[list[str]]
    ) -> None:
        num_signals = self.filtered_signal.shape[1]
        if ylabels is None:
            ylabels = [""] * num_signals

        signals_axs = subfig.subplots(num_signals, 1, sharex=True)
        if num_signals == 1:
            signals_axs = [signals_axs]

        lower_bounds, upper_bounds = compute_vertical_bounds(self.filtered_signal)
        for i in range(len(signals_axs)):
            signals_axs[i].set_ylabel(ylabels[i])
            signals_axs[i].set_xlim((0, self.filtered_signal.shape[0]))
            signals_axs[i].set_ylim(lower_bounds[i], upper_bounds[i])
            signals_axs[i].vlines(
                changepoints, ymin=lower_bounds[i], ymax=upper_bounds[i],
                color="red", linestyles="dotted",
            )
        self.signal_plots = [signal_ax.plot([], [])[0] for signal_ax in signals_axs]

    def _init_img_plot(self, subfig) -> None:
        img_ax = subfig.subplots(1, 1)
        img_ax.set_axis_off()
        cap = cv.VideoCapture(str(self.vid_path))
        _, first_frame = cap.read()
        first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2RGB)
        cap.release()
        self.img_plot = img_ax.imshow(first_frame)

    def run(self) -> None:
        def gen_func():
            for i, (frame, _) in enumerate(get_frames(self.vid_path)):
                yield i, cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        def animate(input):
            i, frame = input
            for idx, line_plot in enumerate(self.signal_plots):
                line_plot.set_data(range(i + 1), self.filtered_signal[: i + 1, idx])
            self.img_plot.set_data(frame)
            return self.plots

        _ = FuncAnimation(self.fig, animate, frames=gen_func, interval=1, repeat=False, blit=True)
        plt.show()