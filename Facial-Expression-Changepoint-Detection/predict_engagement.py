# predict_engagement.py
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

# ---------- Your pipeline modules ----------
from facial_expression_changepoint_detection.video_processing import VideoProcessor
from facial_expression_changepoint_detection.landmarks import (
    LandmarksSignalExtractor,
)
from facial_expression_changepoint_detection.pose_extractor import (
    PoseSignalExtractor,
)
from facial_expression_changepoint_detection.hand_extractor import (
    HandSignalExtractor,
)
from facial_expression_changepoint_detection.video_utils import save_frames
from facial_expression_changepoint_detection.raw_export import (
    export_raw_landmarks_from_frames,
)

# Expected per-frame dimensions (kept in sync with training script)
EXPECTED_FACE_COLS = 136  # 68 x + 68 y
EXPECTED_POSE_COLS = 30   # 10*(x,y,vis)
EXPECTED_HANDS_COLS = 84  # 2 hands * 21 pts * (x,y)

# Pretty names for labels (we'll only print the ones that actually exist in clf.classes_)
LABEL_NAMES = {
    0: "Disengaged",
    1: "Low",
    2: "Engaged",
    3: "Highly Engaged",
}


# ---------------------------------------------------------------------
# Feature building (same logic as training; handles missing modalities)
# ---------------------------------------------------------------------
def _read_first_csv(raw_dir: Path, pattern: str) -> pd.DataFrame | None:
    files = list(raw_dir.glob(pattern))
    return pd.read_csv(files[0]) if files else None


def _row_vals(df: pd.DataFrame | None, i: int, expected_len: int) -> np.ndarray:
    """Return a 1D array of length expected_len. If df missing/short, pad with zeros."""
    if df is None or i >= len(df):
        return np.zeros((expected_len,), dtype=float)
    vals = df.iloc[i].drop(labels=["video", "frame_index"]).to_numpy(dtype=float)
    if vals.size < expected_len:
        out = np.zeros((expected_len,), dtype=float)
        out[: vals.size] = vals
        return out
    elif vals.size > expected_len:
        return vals[:expected_len]
    return vals


def build_features_from_csvs(raw_dir: Path, n_frames: int) -> np.ndarray:
    """
    Build a single fixed-length feature vector:
      concat over time (n_frames) of [face(136) + pose(30) + hands(84)]
    Pads with zeros or trims to exactly n_frames rows.
    """
    df_face = _read_first_csv(raw_dir, "*_face68_*f.csv")
    df_pose = _read_first_csv(raw_dir, "*_upper_body_*f.csv")
    df_hands = _read_first_csv(raw_dir, "*_hands_*f.csv")

    print(f"📊 CSVs found – Face: {'Yes' if df_face is not None else 'No'}, "
          f"Pose: {'Yes' if df_pose is not None else 'No'}, "
          f"Hands: {'Yes' if df_hands is not None else 'No'}")

    if df_face is None:
        raise RuntimeError("Face CSV missing; cannot build features.")

    # Sort to align by time
    key = "frame_index"
    df_face = df_face.sort_values(key)
    if df_pose is not None:
        df_pose = df_pose.sort_values(key)
    if df_hands is not None:
        df_hands = df_hands.sort_values(key)

    num_rows = len(df_face)
    rows: list[np.ndarray] = []
    for i in range(num_rows):
        fvals = _row_vals(df_face, i, EXPECTED_FACE_COLS)
        pvals = _row_vals(df_pose, i, EXPECTED_POSE_COLS)
        hvals = _row_vals(df_hands, i, EXPECTED_HANDS_COLS)
        rows.append(np.concatenate([fvals, pvals, hvals], axis=0))

    if not rows:
        raise RuntimeError("No rows assembled from CSVs.")

    # pad/trim to exactly n_frames
    per_frame_dim = rows[0].size
    if len(rows) < n_frames:
        pad_vec = np.zeros((per_frame_dim,), dtype=float)
        while len(rows) < n_frames:
            rows.append(pad_vec.copy())
    elif len(rows) > n_frames:
        rows = rows[:n_frames]

    feat = np.concatenate(rows, axis=0)
    print(f"✅ Features built: shape=({feat.size},) "
          f"(per-frame={per_frame_dim}, frames={n_frames})")
    return feat


# ---------------------------------------------------------------------
# Video → frames → CSVs → features (same as your existing pipeline)
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

    print("📁 Output directories:")
    print("   - Raw landmarks:", raw_dir)
    print("   - Extracted frames:", frames_dir)

    # 1) Select frames using your VideoProcessor
    print("🎬 Extracting key frames…")
    vp = VideoProcessor(
        vid_path=vid_path,
        signal_extractors=[LandmarksSignalExtractor(),
                           PoseSignalExtractor(),
                           HandSignalExtractor()],
    )
    frames_bgr, changepoints = vp.select_frames(frame_count=n_frames)
    print(f"✅ Extracted {len(frames_bgr)} frames at indices: {changepoints}")

    # 2) Save frames
    print("💾 Saving frame images…")
    frame_files = [f"frame_{idx:04d}.png" for idx in changepoints]
    save_frames(output_dir=frames_dir, frames=frames_bgr, filenames=frame_files)
    print(f"✅ Saved {len(frame_files)} frame images")

    # 3) Export raw landmark CSVs from those frames
    print("📊 Generating landmark CSVs…")
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
    lines.append("🎯 PREDICTION RESULTS")
    lines.append("====================================")
    lines.append(f"Video: {video_name}")
    lines.append(f"Predicted engagement: {pred_label} – {label_name}")
    lines.append("")
    lines.append("📊 Class probabilities:")
    for c, p in zip(classes, probs):
        lines.append(f"  {int(c)} ({LABEL_NAMES.get(int(c), str(c))}): {p:.3f}")
    lines.append("")

    summary_path = out_dir / f"prediction_summary_{Path(video_name).stem}.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"📝 Summary saved → {summary_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Predict engagement for a video using a trained model."
    )
    ap.add_argument("--video", required=True, help="Path to input video file")
    ap.add_argument("--model", default=None, help="Path to .joblib model file")
    ap.add_argument("--model-dir", default=None, help="Directory containing model file")
    ap.add_argument("--n-frames", type=int, default=5, help="Number of frames to use")
    ap.add_argument("--out-dir", default="pred_output", help="Output directory root")
    args = ap.parse_args()

    vid_path = Path(args.video).resolve()
    if not vid_path.exists():
        print(f"❌ Video not found: {vid_path}")
        sys.exit(1)

    # Load model (and optional meta)
    clf, meta, model_path = load_model_from_args(args.model, args.model_dir)
    print(f"📦 Loaded model: {model_path}")
    if meta:
        print(f"ℹ️  Meta: {json.dumps(meta, indent=2)}")

    # Generate features (and write CSVs/frames so you can inspect)
    out_root = Path(args.out_dir)
    gen = generate_outputs_from_video(vid_path=vid_path, out_root=out_root,
                                      n_frames=args.n_frames)

    # Predict (ROBUST to subset of classes)
    features = gen["features"].reshape(1, -1)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(features)[0]  # shape = (len(classes),)
    else:
        # Fallback if model has no predict_proba (e.g., SVM without prob=True)
        # Use a pseudo-probability from decision_function
        pred = clf.predict(features)[0]
        classes = getattr(clf, "classes_", np.array([pred]))
        probs = np.zeros((len(classes),), dtype=float)
        probs[np.where(classes == pred)[0][0]] = 1.0

    classes = getattr(clf, "classes_", np.arange(len(probs), dtype=int))
    pred_label = int(classes[np.argmax(probs)])

    # Print nicely
    print("\n🎯 PREDICTION RESULTS")
    print("====================================")
    print(f"Video: {vid_path.name}")
    print(f"Predicted engagement: {pred_label} – {LABEL_NAMES.get(pred_label, 'Unknown')}")
    print("📊 Class probabilities:")
    for c, p in zip(classes, probs):
        print(f"  {int(c)} ({LABEL_NAMES.get(int(c), str(c))}): {p:.3f}")

    # Save a summary alongside artifacts
    save_summary(out_root, vid_path.name, pred_label, probs, classes)

    print("\n✅ PREDICTION COMPLETE")
    print("Summary report generated")
    print("Landmark CSVs generated")
    print("Frame images extracted")


if __name__ == "__main__":
    main()
