import csv
import random
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import os

from facial_expression_changepoint_detection.landmarks import LandmarksSignalExtractor
from facial_expression_changepoint_detection.video_processing import VideoProcessor
from facial_expression_changepoint_detection.visualization import Animation

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def get_all_videos() -> list[Path]:
    # dataset is a SIBLING of the project folder:
    # <parent>/
    #   dataset/
    #   Facial-Expression-Changepoint-Detection/  (this project)
    dataset_path = (Path(__file__).parent.parent / "dataset").resolve()
    vids: list[Path] = []
    for dirpath, _, filenames in os.walk(dataset_path):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in VIDEO_EXTS:
                vids.append(p)
    return sorted(vids)


def visualize(vid_paths: list[Path], frame_count: int) -> None:
    for vid_path in vid_paths:
        video_processor = VideoProcessor(
            vid_path=vid_path,
            signal_extractor=LandmarksSignalExtractor(indices={20}),
        )
        animation = Animation(
            vid_path,
            video_processor=video_processor,
            frame_count=frame_count,
            title=f"Video ID: {vid_path.stem}",
        )
        animation.run()


def process_video(vid_path: Path, frame_counts: list[int], output_dir: Path) -> str:
    """
    Processes a single video. Declared globally so it can be passed to a multiprocessing pool.

    Returns the filename of the processed video.
    """
    vp = VideoProcessor(vid_path=vid_path)
    vp.process(frame_counts, output_dir)
    return vid_path.name


def run(
    vid_paths: list[Path],
    frame_counts: list[int],
    output_dir_name: str = "output",
    use_multiprocessing: bool = True,
    chunksize: int = 8,
) -> float:
    """
    Processes the given videos, optionally using multiprocessing.

    Returns the elapsed time (seconds).
    """
    t0 = time.perf_counter()

    # Prepare output directories
    output_dir = Path(__file__).parent.parent / output_dir_name
    frame_count_subdirs = [output_dir / f"{i}_frames" for i in frame_counts]
    for subdir in frame_count_subdirs:
        if not subdir.exists():
            Path.mkdir(subdir, parents=True)

    # Prepare CSV header (adds per-region columns)
    csv_path = output_dir / "changepoints.csv"
    with csv_path.open(mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow((
            "video",
            "frame_count",
            "frame_indices",
            "change_scores_global",
            "right_eyebrow",
            "left_eyebrow",
            "right_eye",
            "left_eye",
            "nose_bridge",
            "nose_bottom",
            "outer_lips",
            "inner_lips",
            "jawline",
        ))

    # Process
    if use_multiprocessing:
        with Pool() as pool:
            filenames = pool.imap_unordered(
                partial(process_video, frame_counts=frame_counts, output_dir=output_dir),
                vid_paths,
                chunksize=chunksize,
            )
            for filename in filenames:
                print(f"Finished processing of video {filename}\n")
    else:
        for vid in vid_paths:
            processed = process_video(vid, frame_counts, output_dir)
            print(f"Finished processing of video {processed}\n")

    return time.perf_counter() - t0


def benchmark(
    frame_counts: list[int],
    sample_size: int = 30,
    chunksizes: list[int] | None = None,
) -> None:
    """
    Compares performance on a random sample, with and without multiprocessing.
    """
    if chunksizes is None:
        chunksizes = [1, 4, 8, 16]

    random.seed(0)
    vid_paths = random.sample(get_all_videos(), k=sample_size)

    times = [
        run(
            vid_paths=vid_paths,
            frame_counts=frame_counts,
            chunksize=cs,
            use_multiprocessing=use_mp,
            output_dir_name="benchmark_output",
        )
        for use_mp, cs in zip([False] + [True] * len(chunksizes), [0] + chunksizes)
    ]

    print("Time Elapsed (in seconds):")
    print(f"\tNo multiprocessing: {times[0]}")
    for chunksize, elapsed_time in zip(chunksizes, times[1:]):
        print(f"\tMultiprocessing chunksize={chunksize}: {elapsed_time}")


def configure_settings() -> tuple[list[int], str]:
    """
    Takes user input to set the frame counts and the output directory name.
    """
    default_frame_counts = [1, 2, 3]
    default_output_dir_name = "output"

    print("Configure Settings: (default)")

    # frame counts
    while True:
        print("Enter frame counts as space-separated positive integers: (1 2 3) ", end="")
        try:
            raw = input().strip()
            input_frame_counts = [int(s) for s in raw.split()] if raw else default_frame_counts
            if any(n <= 0 for n in input_frame_counts):
                raise ValueError
        except Exception:
            print("Invalid input.")
        else:
            frame_counts = input_frame_counts
            break

    # output directory name
    while True:
        print("Enter name of output directory: (output) ", end="")
        input_dir_name = input()
        if not input_dir_name:
            output_dir_name = default_output_dir_name
            break
        elif all(s.isalnum() or s == "_" for s in input_dir_name):
            output_dir_name = input_dir_name
            break
        print("Name should only consist of alphanumeric characters.")
    print("\n")
    return frame_counts, output_dir_name


def main() -> None:
    all_vids = get_all_videos()
    frame_counts, output_dir_name = configure_settings()
    time_took = run(
        vid_paths=all_vids,
        frame_counts=frame_counts,
        chunksize=8,
        output_dir_name=output_dir_name,
    )
    print(f"Finished processing all videos in {time_took/(60*60)} hours.")


if __name__ == "__main__":
    main()
