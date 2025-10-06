"""
Predict engagement (0–3) for a single video using the trained RandomForest.

Usage:
  python predict_engagement.py --video "dataset/5000441001.avi" --model-dir "output_ml/models" --n-frames 5
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from facial_expression_changepoint_detection.video_processing import VideoProcessor
from facial_expression_changepoint_detection.landmarks import LandmarksSignalExtractor
from facial_expression_changepoint_detection.pose_extractor import PoseSignalExtractor
from facial_expression_changepoint_detection.hand_extractor import HandSignalExtractor

# Reuse the same feature builder from training
from train_engagement import ensure_raw_landmarks, build_features_from_raw_folder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to a single .avi/.mp4 video")
    ap.add_argument("--model-dir", default="output_ml/models", help="Folder with engagement_rf.joblib")
    ap.add_argument("--n-frames", type=int, default=5, help="How many frames to use (must match training)")
    ap.add_argument("--out-dir", default="output_ml", help="Where to cache raw_landmarks (default: output_ml)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    clf = joblib.load(model_dir / "engagement_rf.joblib")
    meta = json.loads((model_dir / "engagement_rf_meta.json").read_text())

    vid_path = Path(args.video).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Extract raw pixel landmarks for this video (cached)
    raw_folder = ensure_raw_landmarks(vid_path, n_frames=args.n_frames, out_root=out_root)

    # Build features
    feat = build_features_from_raw_folder(raw_folder, n_frames=args.n_frames)
    if feat is None:
        raise RuntimeError("Could not build features (missing raw landmark CSVs).")

    X = feat.reshape(1, -1)
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]

    labels = meta.get("labels", {"0":"0","1":"1","2":"2","3":"3"})
    print(f"\nVideo: {vid_path.name}")
    print(f"Predicted engagement: {pred} ({labels.get(str(pred), str(pred))})")
    print("Class probabilities:")
    for i, p in enumerate(proba):
        print(f"  {i}: {p:.3f}")

if __name__ == "__main__":
    main()
