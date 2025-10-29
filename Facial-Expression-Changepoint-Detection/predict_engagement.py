# predict_engagement.py
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

# Fix Unicode encoding for Windows
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ---------- Your pipeline modules ----------
from facial_expression_changepoint_detection.video_processing import VideoProcessor
from facial_expression_changepoint_detection.landmarks import LandmarksSignalExtractor
from facial_expression_changepoint_detection.pose_extractor import PoseSignalExtractor
from facial_expression_changepoint_detection.hand_extractor import HandSignalExtractor
from facial_expression_changepoint_detection.raw_export import export_raw_landmarks_from_frames
from facial_expression_changepoint_detection.video_utils import save_frames

# Expected per-frame dimensions
EXPECTED_FACE_COLS  = 136
EXPECTED_POSE_COLS  = 30
EXPECTED_HANDS_COLS = 84

# Calculate per_frame_dim constant
per_frame_dim = EXPECTED_FACE_COLS + EXPECTED_POSE_COLS + EXPECTED_HANDS_COLS

# Label names for engagement levels
LABEL_NAMES = {
    0: "Disengaged",
    1: "Low",
    2: "Engaged",
    3: "Highly Engaged",
}

def _read_first_csv(directory: Path, pattern: str) -> pd.DataFrame | None:
    """Helper to read first matching CSV file in directory"""
    files = list(directory.glob(pattern))
    if not files:
        return None
    return pd.read_csv(files[0])

def get_video_path(video_name: str) -> Path:
    """Get absolute path to video (legacy helper; unused by API wrapper)"""
    base_dir = Path("D:/Desktop/cpd/PARTICIPANTS_CLIPS")
    video_path = base_dir / video_name

    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        raise FileNotFoundError(f"Video not found: {video_name}")

    print(f"[SUCCESS] Video found: {video_path}")
    return video_path

def ensure_raw_landmarks_and_frames(vid_path: Path, n_frames: int, out_root: Path) -> Dict[str, Any]:
    """
    Extract frames, generate raw landmarks CSVs, and return all file paths
    """
    print(f"[PROCESS] Generating outputs for: {vid_path.name}")

    # Create output directories
    raw_dir = out_root / "raw_landmarks" / f"{n_frames}_frames" / vid_path.stem
    frames_dir = out_root / "extracted_frames" / f"{n_frames}_frames" / vid_path.stem
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output directories created:")
    print(f"   - Raw landmarks: {raw_dir}")
    print(f"   - Extracted frames: {frames_dir}")

    # Initialize VideoProcessor
    vp = VideoProcessor(
        vid_path=vid_path,
        signal_extractors=[LandmarksSignalExtractor(), PoseSignalExtractor(), HandSignalExtractor()],
    )

    try:
        # Extract frames using changepoint detection
        print("[PROCESS] Extracting key frames...")
        frames, changepoints = vp.select_frames(frame_count=n_frames)
        print(f"[SUCCESS] Extracted {len(frames)} frames at indices: {changepoints}")

        # Save frames as images
        print("[PROCESS] Saving frame images...")
        frame_filenames = [f"frame_{idx:04d}.png" for idx in changepoints]
        save_frames(output_dir=frames_dir, frames=frames, filenames=frame_filenames)
        print(f"[SUCCESS] Saved {len(frames)} frame images")

        # Generate raw landmarks CSV files
        print("[PROCESS] Generating landmark CSVs...")
        written_files = export_raw_landmarks_from_frames(
            video_path=vid_path,
            frames_bgr=frames,
            frame_indices=changepoints,
            out_root=out_root,
            n_frames_label=n_frames,
        )

        print("[SUCCESS] Generated CSV files:")
        for file_type, file_path in written_files.items():
            print(f"   - {file_type}: {file_path.name}")

        # Build features from the generated CSVs (for consistency with training)
        features = build_features_from_csvs(raw_dir, n_frames)

        return {
            "features": features,
            "raw_dir": raw_dir,
            "frames_dir": frames_dir,
            "csv_files": written_files,
            "changepoints": changepoints,
            "frames": frames,
        }

    except Exception as e:
        print(f"[ERROR] Error generating outputs: {e}")
        raise

def build_features_from_csvs(raw_dir: Path, n_frames: int) -> np.ndarray:
    """
    Build a single fixed-length feature vector:
      concat over time of [face(136) + pose(30) + hands(84)].
    Robust to: missing CSVs, different row counts, extra/missing columns.
    """
    df_face = _read_first_csv(raw_dir, "*_face68_*f.csv")
    df_pose = _read_first_csv(raw_dir, "*_upper_body_*f.csv")
    df_hands = _read_first_csv(raw_dir, "*_hands_*f.csv")

    n_face = len(df_face) if df_face is not None else 0
    n_pose = len(df_pose) if df_pose is not None else 0
    n_hands = len(df_hands) if df_hands is not None else 0
    num_rows = max(n_face, n_pose, n_hands)

    print(
        f"[INFO] CSVs found - Face: {'Yes' if df_face is not None else 'No'} ({n_face} rows), "
        f"Pose: {'Yes' if df_pose is not None else 'No'} ({n_pose} rows), "
        f"Hands: {'Yes' if df_hands is not None else 'No'} ({n_hands} rows)"
    )

    def safe_row(df, i, expected_len):
        """Return numeric np.array of length expected_len; zeros if df missing/short."""
        if df is None or i >= len(df):
            return np.zeros(expected_len, dtype=float)
        # Drop non-feature cols if present; ignore if not
        arr = df.iloc[i].drop(labels=["video", "frame_index"], errors="ignore").to_numpy()
        # Coerce to float, replace non-numeric with 0
        arr = pd.to_numeric(pd.Series(arr), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        # Trim/pad to expected length
        if arr.size > expected_len:
            arr = arr[:expected_len]
        elif arr.size < expected_len:
            arr = np.pad(arr, (0, expected_len - arr.size), constant_values=0.0)
        return arr

    rows: List[np.ndarray] = []
    for i in range(num_rows):
        fvals = safe_row(df_face,  i, EXPECTED_FACE_COLS)
        pvals = safe_row(df_pose,  i, EXPECTED_POSE_COLS)
        hvals = safe_row(df_hands, i, EXPECTED_HANDS_COLS)
        rows.append(np.concatenate([fvals, pvals, hvals], dtype=float))

    # If no rows at all (all CSVs missing), keep zeros
    if not rows:
        rows = [np.zeros(per_frame_dim, dtype=float)]

    # Pad/trim time dimension to exactly n_frames
    if len(rows) < n_frames:
        pad_vec = np.zeros(per_frame_dim, dtype=float)
        rows.extend([pad_vec.copy() for _ in range(n_frames - len(rows))])
    elif len(rows) > n_frames:
        rows = rows[:n_frames]

    feat = np.concatenate(rows, axis=0).astype(float)
    print(f"[SUCCESS] Features built: shape=({feat.size},) (per-frame={per_frame_dim}, frames={n_frames})")
    return feat

# ---------------------------------------------------------------------
# Video -> frames -> CSVs -> features (same as your existing pipeline)
# ---------------------------------------------------------------------
def generate_outputs_from_video(vid_path: Path, out_root: Path, n_frames: int):
    """
    Returns dict with features, and paths to written CSVs/frames for inspection.
    """
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    stem = vid_path.stem
    raw_dir = out_root / "raw_landmarks" / f"{n_frames}_frames" / stem
    frames_dir = out_root / "extracted_frames" / f"{n_frames}_frames" / stem
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Output directories:")
    print("   - Raw landmarks:", raw_dir)
    print("   - Extracted frames:", frames_dir)

    # 1) Select frames using your VideoProcessor
    print("[PROCESS] Extracting key frames...")
    vp = VideoProcessor(
        vid_path=vid_path,
        signal_extractors=[LandmarksSignalExtractor(), PoseSignalExtractor(), HandSignalExtractor()],
    )
    frames_bgr, changepoints = vp.select_frames(frame_count=n_frames)
    print(f"[SUCCESS] Extracted {len(frames_bgr)} frames at indices: {changepoints}")

    # 2) Save frames
    print("[PROCESS] Saving frame images...")
    frame_files = [f"frame_{idx:04d}.png" for idx in changepoints]
    save_frames(output_dir=frames_dir, frames=frames_bgr, filenames=frame_files)
    print(f"[SUCCESS] Saved {len(frame_files)} frame images")

    # 3) Export raw landmark CSVs from those frames
    print("[PROCESS] Generating landmark CSVs...")
    written = export_raw_landmarks_from_frames(
        video_path=vid_path,
        frames_bgr=frames_bgr,
        frame_indices=changepoints,
        out_root=out_root,
        n_frames_label=n_frames,
    )
    for k, p in written.items():
        print(f"   - {k}: {Path(p).name}")

    # 4) Build features (same shape/ordering used by training)
    features = build_features_from_csvs(raw_dir, n_frames)

    return {
        "features": features,
        "raw_dir": raw_dir,
        "frames_dir": frames_dir,
        "csv_files": written,
        "changepoints": changepoints,
        "frames": frames_bgr,
    }

# ---------------------------------------------------------------------
# Prediction utils (ROBUST: use clf.classes_ for labels/probabilities)
# ---------------------------------------------------------------------
def load_model_from_args(model_path: str | None, model_dir: str | None) -> tuple:
    if model_path:
        mp = Path(model_path)
    else:
        if not model_dir:
            raise ValueError("Provide --model or --model-dir")
        # default filename used by training script
        mp = Path(model_dir) / "engagement_rf.joblib"
    if not mp.exists():
        raise FileNotFoundError(f"Model not found: {mp}")
    clf = joblib.load(mp)
    meta_path = mp.with_name("engagement_rf_meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    return clf, meta, mp

def save_summary(out_dir: Path, video_name: str, pred_label: int,
                 probs: np.ndarray, classes: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    label_name = LABEL_NAMES.get(int(pred_label), str(pred_label))
    lines = []
    lines.append("PREDICTION RESULTS")
    lines.append("=" * 40)
    lines.append(f"Video: {video_name}")
    lines.append(f"Predicted engagement: {pred_label} - {label_name}")
    lines.append("")
    lines.append("Class probabilities:")
    for c, p in zip(classes, probs):
        lines.append(f"  {int(c)} ({LABEL_NAMES.get(int(c), str(c))}): {p:.3f}")
    lines.append("")
    summary_path = out_dir / f"prediction_summary_{Path(video_name).stem}.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Summary saved -> {summary_path}")

# ---------------------------------------------------------------------
# **NEW**: Programmatic API used by the Node wrapper
# ---------------------------------------------------------------------
def predict_from_video(video_path: str, model_path: Optional[str] = None,
                       n_frames: int = 5, out_dir: str = "pred_output") -> Dict[str, Any]:
    """
    Entry-point used by api_predict.py (Node -> Python).
    Returns a dict compatible with the UI's RESULT_SCHEMA.
    """
    vid_path = Path(video_path).resolve()
    if not vid_path.exists():
        raise FileNotFoundError(f"Video not found: {vid_path}")

    # Try provided path, then common fallbacks
    clf = None
    used_model = None
    if model_path and Path(model_path).exists():
        clf = joblib.load(model_path)
        used_model = str(Path(model_path).resolve())
    else:
        repo_root = Path(__file__).resolve().parent  # this file's folder (repo root)
        # Try output_ml/model/{random_forest|engagement_rf}.joblib
        for fname in ("random_forest.joblib", "engagement_rf.joblib"):
            m = repo_root / "output_ml" / "model" / fname
            if m.exists():
                clf = joblib.load(m)
                used_model = str(m.resolve())
                break
    if clf is None:
        raise FileNotFoundError("No model file found. Provide --model or place a model in output_ml/model/")

    # Generate features via your pipeline
    gen = generate_outputs_from_video(vid_path=vid_path, out_root=Path(out_dir), n_frames=n_frames)
    features = gen["features"].reshape(1, -1)

    # Predict probabilities (robust to classifier type)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(features)[0]  # shape (num_classes,)
        classes = getattr(clf, "classes_", np.arange(len(probs), dtype=int))
    else:
        pred = clf.predict(features)[0]
        classes = getattr(clf, "classes_", np.array([pred]))
        probs = np.zeros((len(classes),), dtype=float)
        probs[np.where(classes == pred)[0][0]] = 1.0

    # Convert to 0-100 score (weighted by label intensity)
    weights = {0: 25, 1: 50, 2: 75, 3: 100}
    score = 0.0
    for c, p in zip(classes, probs):
        score += weights.get(int(c), 50) * float(p)
    score = int(np.clip(score, 0, 100))

    # Trend: constant for now (one-vector prediction); later you can compute per-segment
    trend = [score] * 20

    # Map class probs to UI buckets
    ui_states = {"Focused": 0, "Engaged": 0, "Neutral": 0, "Distracted": 0}
    for c, p in zip(classes, probs):
        label = LABEL_NAMES.get(int(c), "Unknown")
        if label == "Highly Engaged":
            ui_states["Focused"] += float(p) * 100
        elif label == "Engaged":
            ui_states["Engaged"] += float(p) * 100
        elif label == "Low":
            ui_states["Neutral"] += float(p) * 100
        elif label == "Disengaged":
            ui_states["Distracted"] += float(p) * 100
    ui_states = {k: int(round(v)) for k, v in ui_states.items()}

    key_moments = [
        {"t": "00:00", "note": "Analysis start"},
        {"t": "05:00", "note": "Representative segment"},
        {"t": "10:00", "note": "Representative segment"},
        {"t": "15:00", "note": "Representative segment"},
    ]

    result = {
        "title": vid_path.name,
        "duration": "20:00",
        "date": datetime.now().strftime("%b %d, %Y"),
        "score": score,
        "trend": trend,
        "states": ui_states,
        "keyMoments": key_moments,
        # internal plumbing for api_predict to surface in meta.used_model
        "_used_model": used_model,
    }
    return result

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Predict engagement for a video using a trained model.")
    ap.add_argument("--video", required=True, help="Path to input video file")
    ap.add_argument("--model", default=None, help="Path to .joblib model file")
    ap.add_argument("--model-dir", default=None, help="Directory containing model file")
    ap.add_argument("--n-frames", type=int, default=5, help="Number of frames to use")
    ap.add_argument("--out-dir", default="pred_output", help="Output directory root")
    args = ap.parse_args()

    vid_path = Path(args.video).resolve()
    if not vid_path.exists():
        print(f"[ERROR] Video not found: {vid_path}")
        sys.exit(1)

    # Load model (and optional meta) for CLI flow
    clf, meta, model_path = load_model_from_args(args.model, args.model_dir)
    print(f"[MODEL] Loaded model: {model_path}")
    if meta:
        print(f"[INFO] Meta: {json.dumps(meta, indent=2)}")

    # Generate features (and write CSVs/frames so you can inspect)
    out_root = Path(args.out_dir)
    gen = generate_outputs_from_video(vid_path=vid_path, out_root=out_root, n_frames=args.n_frames)

    # Predict (ROBUST to subset of classes)
    features = gen["features"].reshape(1, -1)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(features)[0]  # shape = (len(classes),)
    else:
        pred = clf.predict(features)[0]
        classes = getattr(clf, "classes_", np.array([pred]))
        probs = np.zeros((len(classes),), dtype=float)
        probs[np.where(classes == pred)[0][0]] = 1.0

    classes = getattr(clf, "classes_", np.arange(len(probs), dtype=int))
    pred_label = int(classes[np.argmax(probs)])

    # Print nicely
    print("\nPREDICTION RESULTS")
    print("=" * 40)
    print(f"Video: {vid_path.name}")
    print(f"Predicted engagement: {pred_label} - {LABEL_NAMES.get(pred_label, 'Unknown')}")
    print("Class probabilities:")
    for c, p in zip(classes, probs):
        print(f"  {int(c)} ({LABEL_NAMES.get(int(c), str(c))}): {p:.3f}")

    # Save a summary alongside artifacts
    save_summary(out_root, vid_path.name, pred_label, probs, classes)

    print("\n[SUCCESS] PREDICTION COMPLETE")
    print("[INFO] Summary report generated")
    print("[INFO] Landmark CSVs generated")
    print("[INFO] Frame images extracted")

if __name__ == "__main__":
    main()