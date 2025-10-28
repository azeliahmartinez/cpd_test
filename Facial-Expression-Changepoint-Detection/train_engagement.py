"""
Train a RandomForest on DAiSEE Engagement (0-3) using facial landmarks
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import your modules
from facial_expression_changepoint_detection.video_processing import VideoProcessor
from facial_expression_changepoint_detection.landmarks import LandmarksSignalExtractor
from facial_expression_changepoint_detection.pose_extractor import PoseSignalExtractor
from facial_expression_changepoint_detection.hand_extractor import HandSignalExtractor
from facial_expression_changepoint_detection.raw_export import export_raw_landmarks_from_frames

# Expected dimensions
EXPECTED_FACE_COLS = 136
EXPECTED_POSE_COLS = 30
EXPECTED_HANDS_COLS = 84

def find_videos(root: Path) -> Dict[str, Path]:
    """Find all DAiSEE video files"""
    vidmap = {}
    data_root = root / "DataSet"
    for dirpath, _, files in os.walk(data_root):
        for fn in files:
            if fn.lower().endswith(".avi"):
                vidmap[fn] = Path(dirpath) / fn
    return vidmap

def load_split_labels(root: Path) -> Dict[str, pd.DataFrame]:
    """Load train and validation labels"""
    lbl_root = root / "Labels"
    splits = {
        "Train": lbl_root / "TrainLabels.csv",
        "Validation": lbl_root / "ValidationLabels.csv",
    }
    out = {}
    for k, p in splits.items():
        df = pd.read_csv(p)
        out[k] = df[["ClipID", "Engagement"]].copy()
    return out

def ensure_raw_landmarks(vid_path: Path, n_frames: int, out_root: Path) -> Optional[Path]:
    """Extract landmarks for a video"""
    clip_id = vid_path.stem
    raw_dir = out_root / "raw_landmarks" / f"{n_frames}_frames" / clip_id
    frames_dir = out_root / "extracted_frames" / f"{n_frames}_frames" / clip_id
    
    # Skip if already processed
    if raw_dir.exists():
        face_files = list(raw_dir.glob(f"*_face68_*f.csv"))
        if face_files:
            print(f"  Using cached: {clip_id}")
            return raw_dir

    print(f"  Processing: {clip_id}")
    
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize VideoProcessor
        vp = VideoProcessor(
            vid_path=vid_path,
            signal_extractors=[
                LandmarksSignalExtractor(),
                PoseSignalExtractor(),
                HandSignalExtractor()
            ],
        )
        
        # Extract frames
        frames, changepoints = vp.select_frames(frame_count=n_frames)
        
        # Save frames as images
        from facial_expression_changepoint_detection.video_utils import save_frames
        frame_filenames = [f"frame_{idx:04d}.png" for idx in changepoints]
        save_frames(output_dir=frames_dir, frames=frames, filenames=frame_filenames)
        
        # Generate landmarks
        export_raw_landmarks_from_frames(
            video_path=vid_path,
            frames_bgr=frames,
            frame_indices=changepoints,
            out_root=out_root,
            n_frames_label=n_frames
        )
        
        return raw_dir
        
    except Exception as e:
        print(f"  Error: {e}")
        return None

def build_features_from_csvs(raw_dir: Path, n_frames: int) -> np.ndarray:
    """Build features from the generated CSV files (OLD WORKING VERSION)"""
    print("  Building features from CSVs...")
    
    def read_csv(p: Path, pattern: str) -> Optional[pd.DataFrame]:
        files = list(p.glob(pattern))
        return pd.read_csv(files[0]) if files else None
    
    # Read generated CSV files
    df_face = read_csv(raw_dir, f"*_face68_*f.csv")
    df_pose = read_csv(raw_dir, f"*_upper_body_*f.csv") 
    df_hands = read_csv(raw_dir, f"*_hands_*f.csv")
    
    print(f"  CSVs found - Face: {'Yes' if df_face is not None else 'No'}, "
          f"Pose: {'Yes' if df_pose is not None else 'No'}, "
          f"Hands: {'Yes' if df_hands is not None else 'No'}")
    
    # Get the actual number of rows from each dataframe
    num_face_rows = len(df_face) if df_face is not None else 0
    num_pose_rows = len(df_pose) if df_pose is not None else 0
    num_hands_rows = len(df_hands) if df_hands is not None else 0
    
    print(f"  Row counts - Face: {num_face_rows}, Pose: {num_pose_rows}, Hands: {num_hands_rows}")
    
    # Use the maximum number of rows available
    num_rows = max(num_face_rows, num_pose_rows, num_hands_rows)
    
    # Use training-style feature building
    rows = []
    
    for i in range(num_rows):
        # Face features
        if df_face is not None and i < num_face_rows:
            fvals = df_face.iloc[i].drop(["video", "frame_index"]).to_numpy()
        else:
            fvals = np.zeros(EXPECTED_FACE_COLS)
        
        if len(fvals) > EXPECTED_FACE_COLS:
            fvals = fvals[:EXPECTED_FACE_COLS]
        elif len(fvals) < EXPECTED_FACE_COLS:
            fvals = np.pad(fvals, (0, EXPECTED_FACE_COLS - len(fvals)))
        
        # Pose features  
        if df_pose is not None and i < num_pose_rows:
            pvals = df_pose.iloc[i].drop(["video", "frame_index"]).to_numpy()
        else:
            pvals = np.zeros(EXPECTED_POSE_COLS)
            
        if len(pvals) > EXPECTED_POSE_COLS:
            pvals = pvals[:EXPECTED_POSE_COLS]
        elif len(pvals) < EXPECTED_POSE_COLS:
            pvals = np.pad(pvals, (0, EXPECTED_POSE_COLS - len(pvals)))
        
        # Hand features
        if df_hands is not None and i < num_hands_rows:
            hvals = df_hands.iloc[i].drop(["video", "frame_index"]).to_numpy()
        else:
            hvals = np.zeros(EXPECTED_HANDS_COLS)
            
        if len(hvals) > EXPECTED_HANDS_COLS:
            hvals = hvals[:EXPECTED_HANDS_COLS]
        elif len(hvals) < EXPECTED_HANDS_COLS:
            hvals = np.pad(hvals, (0, EXPECTED_HANDS_COLS - len(hvals)))
        
        rows.append(np.concatenate([fvals, pvals, hvals]))
    
    # Pad/trim to exactly n_frames
    if len(rows) < n_frames:
        pad_vec = np.zeros(EXPECTED_FACE_COLS + EXPECTED_POSE_COLS + EXPECTED_HANDS_COLS)
        while len(rows) < n_frames:
            rows.append(pad_vec.copy())
    elif len(rows) > n_frames:
        rows = rows[:n_frames]
    
    features = np.concatenate(rows)
    print(f"  Features built: {features.shape}")
    return features

def make_split_features(split_df: pd.DataFrame, vidmap: Dict[str, Path], n_frames: int, out_root: Path, tag: str, max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build features for a split"""
    X_list, y_list = [], []
    kept = 0
    
    # Shuffle and take samples
    split_df = split_df.sample(frac=1, random_state=42).head(max_samples)

    for _, row in split_df.iterrows():
        clipid = str(row["ClipID"]).strip()
        label = int(row["Engagement"])
        
        clipid_ext = clipid if clipid.lower().endswith(".avi") else f"{clipid}.avi"
        vid_path = vidmap.get(clipid_ext)
        if vid_path is None:
            print(f"  [{tag}] Missing video: {clipid}")
            continue

        raw_folder = ensure_raw_landmarks(vid_path, n_frames, out_root)
        if raw_folder is None:
            print(f"  [{tag}] No raw folder: {vid_path.name}")
            continue

        feat = build_features_from_csvs(raw_folder, n_frames)
        if feat is not None:
            X_list.append(feat)
            y_list.append(label)
            kept += 1
            print(f"  [{tag}] {kept}/{max_samples} - {clipid}")

    print(f"  [{tag}] Completed: {kept} samples")
    
    if not X_list:
        return np.array([]), np.array([])
    
    X = np.vstack(X_list)  # This creates proper 2D array
    y = np.array(y_list)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daisee-root", required=True, help="DAiSEE dataset root")
    ap.add_argument("--n-frames", type=int, default=5, help="Frames per video")
    ap.add_argument("--out-dir", default="output_ml", help="Output directory")
    ap.add_argument("--train-samples", type=int, default=30, help="Training samples")
    ap.add_argument("--val-samples", type=int, default=10, help="Validation samples")
    args = ap.parse_args()

    # Setup
    daisee_root = Path(args.daisee_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("📂 Loading DAiSEE dataset...")
    vidmap = find_videos(daisee_root)
    splits = load_split_labels(daisee_root)

    # Build features
    print("🧩 Building training features...")
    X_train, y_train = make_split_features(splits["Train"], vidmap, args.n_frames, out_root, "TRAIN", args.train_samples)
    
    print("🧩 Building validation features...") 
    X_val, y_val = make_split_features(splits["Validation"], vidmap, args.n_frames, out_root, "VALID", args.val_samples)

    print(f"✅ Training: {X_train.shape}, Validation: {X_val.shape}")

    # Check if we have any data
    if X_train.shape[0] == 0:
        print("❌ No training samples were built!")
        return

    # Train model
    print("🚀 Training RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    print(f"📊 Train Accuracy: {train_acc:.3f}")
    
    if X_val.shape[0] > 0:
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        print(f"📊 Validation Accuracy: {val_acc:.3f}")

    # Save model
    models_dir = out_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(clf, models_dir / "engagement_rf.joblib")
    
    meta = {
        "n_frames": args.n_frames,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc) if X_val.shape[0] > 0 else 0,
        "labels": {"0": "Disengaged", "1": "Low", "2": "Engaged", "3": "Highly Engaged"}
    }
    
    with open(models_dir / "engagement_rf_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Model saved!")

if __name__ == "__main__":
    main()