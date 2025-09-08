import argparse
import csv
from pathlib import Path
import cv2

def load_cp_indices(csv_path: Path, video_name: str, frame_count: int):
    """Return a set of frame indices for the given video and frame_count from changepoints.csv."""
    indices = None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["video"] == video_name and int(row["frame_count"]) == frame_count:
                # frame_indices looks like: [0|56|198]
                raw = row["frame_indices"].strip()
                raw = raw.strip("[]")
                if raw:
                    indices = {int(x) for x in raw.split("|")}
                else:
                    indices = set()
                break
    if indices is None:
        raise ValueError(f"No row found in {csv_path} for video={video_name!r} and frame_count={frame_count}")
    return indices

def main():
    ap = argparse.ArgumentParser(description="Overlay CPD markers on video playback.")
    ap.add_argument("--video", required=True, help="Path to the video file (e.g., dataset/Test/5000441001.avi)")
    ap.add_argument("--csv", default="output/changepoints.csv", help="Path to changepoints.csv (default: output/changepoints.csv)")
    ap.add_argument("--frames", type=int, default=3, help="Which frame_count to visualize (must exist in CSV). Default: 3")
    ap.add_argument("--fps", type=float, default=0, help="Override display FPS (0 = use source FPS)")
    args = ap.parse_args()

    video_path = Path(args.video)
    csv_path = Path(args.csv)
    frame_count = args.frames

    # Load CP indices
    cp_indices = load_cp_indices(csv_path, video_path.name, frame_count)
    print(f"Loaded CP indices for {video_path.name} (frame_count={frame_count}): {sorted(cp_indices)}")

    cap = cv2.VideoCapture(str(video_path), cv2.CAP_AVFOUNDATION)  # AVFoundation is best on macOS; falls back automatically if not supported
    if not cap.isOpened():
        # Try default backend
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    wait_ms = int(1000.0 / (args.fps if args.fps > 0 else src_fps))

    frame_idx = 0
    window = f"CPD: {video_path.name} (q to quit)"
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Overlay frame index
        cv2.putText(frame, f"frame {frame_idx}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # Highlight CP frames
        if frame_idx in cp_indices:
            # red banner + text “CP”
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 255), thickness=-1)
            cv2.putText(frame, "CHANGE-POINT (CP)", (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(window, frame)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
