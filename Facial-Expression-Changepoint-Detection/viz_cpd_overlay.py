import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np



# ----------------------------- CPD CSV LOADING -----------------------------

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


# ----------------------------- LANDMARK SETS -------------------------------
# Face subset: same “68-style” regions you use in your pipeline.
_FACE_GROUPS = [
    {46, 53, 52, 65, 55},                    # right eyebrow
    {285, 295, 282, 283, 276},               # left eyebrow
    {33, 160, 158, 144, 153, 133},           # right eye
    {362, 385, 387, 380, 373, 263},          # left eye
    {6, 197, 195, 5},                        # nose bridge
    {98, 97, 2, 326, 327},                   # nose bottom
    {61, 40, 37, 0, 267, 270, 91, 84, 17, 314, 321, 291},  # outer lips
    {78, 81, 13, 311, 178, 14, 402, 308},    # inner lips
    {127, 234, 93, 132, 58, 172, 150, 176, 152, 400, 379, 397, 288, 361, 323, 454, 356},  # jawline
]
_FACE_INDICES = sorted(set().union(*_FACE_GROUPS))

# Pose subset: nose + shoulders/elbows/wrists + hips
_POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]

# Simple pose connections to visualize limbs/torso
_POSE_EDGES = [
    (11, 13), (13, 15),     # left shoulder-elbow-wrist
    (12, 14), (14, 16),     # right shoulder-elbow-wrist
    (11, 12),               # shoulders
    (23, 24),               # hips
    (11, 23), (12, 24)      # torso diagonals
]


# ----------------------------- MODEL HELPERS -------------------------------

def build_models(models_root: Path):
    """Create MediaPipe Face & Pose landmarkers from .task models."""
    face_task = models_root / "face_landmarker.task"
    pose_task = models_root / "pose_landmarker.task"
    hand_task = models_root / "hand_landmarker.task"
    if not face_task.exists():
        raise FileNotFoundError(f"Missing model file: {face_task}")
    if not pose_task.exists():
        raise FileNotFoundError(f"Missing model file: {pose_task}")
    if not hand_task.exists():
        raise FileNotFoundError(f"Missing model file: {hand_task}")

    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Face
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    face_opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(face_task)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.0,
        min_face_presence_confidence=0.0,
        min_tracking_confidence=0.0,
    )

    # Pose
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    pose_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_task)),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.2,
        min_pose_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )

    # Hands
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    hand_opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_task)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )

    face_model = FaceLandmarker.create_from_options(face_opts)
    pose_model = PoseLandmarker.create_from_options(pose_opts)
    hand_model = HandLandmarker.create_from_options(hand_opts)
    return face_model, pose_model, hand_model


def mp_image_from_bgr(frame: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


# ----------------------------- DRAW HELPERS --------------------------------

def draw_cpd_banner(frame: np.ndarray):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 255), thickness=-1)
    cv2.putText(frame, "CHANGE-POINT (CP)", (12, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def draw_face_points(frame: np.ndarray, face_landmarks, color=(255, 200, 0)):
    """Draw selected face landmarks as small circles."""
    if not face_landmarks:
        return
    lms = face_landmarks[0]  # first face
    h, w = frame.shape[:2]
    for i in _FACE_INDICES:
        if i < len(lms):
            lm = lms[i]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 2, color, -1)


def draw_pose_points_and_edges(frame: np.ndarray, pose_landmarks, point_color=(0, 255, 0), edge_color=(0, 180, 0)):
    """Draw selected pose landmarks and simple limb/torso edges."""
    if not pose_landmarks:
        return
    lms = pose_landmarks[0]
    h, w = frame.shape[:2]

    # points
    for i in _POSE_INDICES:
        if i < len(lms):
            lm = lms[i]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, point_color, -1)

    # edges
    def valid(p):
        return (p < len(lms) and 0.0 <= lms[p].x <= 1.0 and 0.0 <= lms[p].y <= 1.0)

    for a, b in _POSE_EDGES:
        if valid(a) and valid(b):
            pa, pb = lms[a], lms[b]
            ax, ay = int(pa.x * w), int(pa.y * h)
            bx, by = int(pb.x * w), int(pb.y * h)
            cv2.line(frame, (ax, ay), (bx, by), edge_color, 2)

def draw_hand_points(frame: np.ndarray, hand_res, left_color=(0, 255, 0), right_color=(0, 160, 255)):
    """
    Draw 21 landmarks per hand as small circles (left=green, right=orange).
    """
    if not hand_res or not getattr(hand_res, "hand_landmarks", None):
        return

    h, w = frame.shape[:2]

    def _top_label(hcls):
        if not hcls:
            return "Unknown"
        best = max(hcls, key=lambda c: getattr(c, "score", 0.0))
        return getattr(best, "category_name", "Unknown")

    left_xy = None
    right_xy = None

    for i, lm in enumerate(hand_res.hand_landmarks):
        label = "Unknown"
        if getattr(hand_res, "handedness", None) and i < len(hand_res.handedness):
            label = _top_label(hand_res.handedness[i])

        xy = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        if str(label).lower().startswith("left"):
            left_xy = xy
        elif str(label).lower().startswith("right"):
            right_xy = xy
        else:
            if left_xy is None:
                left_xy = xy
            else:
                right_xy = xy

    if left_xy is not None:
        for (x, y) in left_xy:
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(frame, (int(x), int(y)), 3, left_color, -1)

    if right_xy is not None:
        for (x, y) in right_xy:
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(frame, (int(x), int(y)), 3, right_color, -1)



# --------------------------------- MAIN ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Overlay CPD markers + face/pose landmarks on video playback.")
    ap.add_argument("--video", required=True, help="Path to the video file (e.g., dataset/5000441001.avi)")
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

    # Where the .task models live relative to this script
    models_root = Path(__file__).parent / "facial_expression_changepoint_detection" / "pretrained_models"

    # Build models
    face_model, pose_model, hand_model = build_models(models_root)


    # Open video (try AVFoundation on macOS, default otherwise)
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
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
        cv2.putText(frame, f"frame {frame_idx}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Timestamp in ms for MediaPipe (use actual stream position)
        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Face landmarks (selected subset drawn)
        face_res = face_model.detect_for_video(mp_img, ts_ms)
        if face_res and face_res.face_landmarks:
            draw_face_points(frame, face_res.face_landmarks, color=(255, 200, 0))  # yellow-ish

        # Pose landmarks (upper-body)
        pose_res = pose_model.detect_for_video(mp_img, ts_ms)
        if pose_res and pose_res.pose_landmarks:
            draw_pose_points_and_edges(frame, pose_res.pose_landmarks,
                                       point_color=(0, 255, 0), edge_color=(0, 180, 0))
            
        # Hand landmarks
        hand_res = hand_model.detect_for_video(mp_img, ts_ms)
        draw_hand_points(frame, hand_res)


        # Highlight CP frames
        if frame_idx in cp_indices:
            draw_cpd_banner(frame)

        # Show
        cv2.imshow(window, frame)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Close models
    face_model.close()
    pose_model.close()
    hand_model.close()


if __name__ == "__main__":
    main()
