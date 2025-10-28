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
from facial_expression_changepoint_detection.landmarks import LandmarksSignalExtractor
from facial_expression_changepoint_detection.pose_extractor import PoseSignalExtractor
from facial_expression_changepoint_detection.hand_extractor import HandSignalExtractor
from facial_expression_changepoint_detection.raw_export import export_raw_landmarks_from_frames

# Expected per-frame dimensions
EXPECTED_FACE_COLS  = 136
EXPECTED_POSE_COLS  = 30
EXPECTED_HANDS_COLS = 84

def get_video_path(video_name: str) -> Path:
    """Get absolute path to video"""
    base_dir = Path("D:/Desktop/cpd/PARTICIPANTS_CLIPS")
    video_path = base_dir / video_name
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        raise FileNotFoundError(f"Video not found: {video_name}")
    
    print(f"✅ Video found: {video_path}")
    return video_path

def ensure_raw_landmarks_and_frames(vid_path: Path, n_frames: int, out_root: Path) -> Dict[str, Path]:
    """
    Enhanced version: Extract frames, generate raw landmarks CSVs, and return all file paths
    """
    print(f"🔄 Generating outputs for: {vid_path.name}")
    
    # Create output directories
    raw_dir = out_root / "raw_landmarks" / f"{n_frames}_frames" / vid_path.stem
    frames_dir = out_root / "extracted_frames" / f"{n_frames}_frames" / vid_path.stem
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directories created:")
    print(f"   - Raw landmarks: {raw_dir}")
    print(f"   - Extracted frames: {frames_dir}")
    
    # Initialize VideoProcessor
    vp = VideoProcessor(
        vid_path=vid_path,
        signal_extractors=[
            LandmarksSignalExtractor(),
            PoseSignalExtractor(),
            HandSignalExtractor()
        ],
    )
    
    try:
        # Extract frames using changepoint detection
        print("🎬 Extracting key frames...")
        frames, changepoints = vp.select_frames(frame_count=n_frames)
        print(f"✅ Extracted {len(frames)} frames at indices: {changepoints}")
        
        # Save frames as images
        print("💾 Saving frame images...")
        from facial_expression_changepoint_detection.video_utils import save_frames
        frame_filenames = [f"frame_{idx:04d}.png" for idx in changepoints]
        save_frames(output_dir=frames_dir, frames=frames, filenames=frame_filenames)
        print(f"✅ Saved {len(frames)} frame images")
        
        # Generate raw landmarks CSV files
        print("📊 Generating landmark CSVs...")
        written_files = export_raw_landmarks_from_frames(
            video_path=vid_path,
            frames_bgr=frames,
            frame_indices=changepoints,
            out_root=out_root,
            n_frames_label=n_frames
        )
        
        print("✅ Generated CSV files:")
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
            "frames": frames
        }
        
    except Exception as e:
        print(f"❌ Error generating outputs: {e}")
        raise

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
    
    # Use training-style feature building
    rows = []
    num_rows = len(df_face) if df_face is not None else 0
    
    for i in range(num_rows):
        # Face features
        fvals = df_face.iloc[i].drop(["video", "frame_index"]).to_numpy() if df_face is not None else np.zeros(EXPECTED_FACE_COLS)
        if len(fvals) > EXPECTED_FACE_COLS:
            fvals = fvals[:EXPECTED_FACE_COLS]
        elif len(fvals) < EXPECTED_FACE_COLS:
            fvals = np.pad(fvals, (0, EXPECTED_FACE_COLS - len(fvals)))
        
        # Pose features  
        pvals = df_pose.iloc[i].drop(["video", "frame_index"]).to_numpy() if df_pose is not None else np.zeros(EXPECTED_POSE_COLS)
        if len(pvals) > EXPECTED_POSE_COLS:
            pvals = pvals[:EXPECTED_POSE_COLS]
        elif len(pvals) < EXPECTED_POSE_COLS:
            pvals = np.pad(pvals, (0, EXPECTED_POSE_COLS - len(pvals)))
        
        # Hand features
        hvals = df_hands.iloc[i].drop(["video", "frame_index"]).to_numpy() if df_hands is not None else np.zeros(EXPECTED_HANDS_COLS)
        if len(hvals) > EXPECTED_HANDS_COLS:
            hvals = hvals[:EXPECTED_HANDS_COLS]
        elif len(hvals) < EXPECTED_HANDS_COLS:
            hvals = np.pad(hvals, (0, EXPECTED_HANDS_COLS - len(hvals)))
        
        rows.append(np.concatenate([fvals, pvals, hvals]))
    
    # Pad/trim to exactly n_frames
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
