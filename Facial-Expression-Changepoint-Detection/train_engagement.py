"""
Train a RandomForest on DAiSEE Engagement (0–3) using ONLY pixel coordinates
exported by your pipeline (face-68, upper-body subset, hands) from N frames.

Usage (PowerShell from repo root):
  python train_engagement.py --daisee-root "D:\\Desktop\\cpd\\DAiSEE" --n-frames 5 --out-dir "output_ml" --limit 10

It will:
  - Walk DAiSEE/DataSet/{Train,Validation,Test} to find videos
  - Use Labels/*.csv to read Engagement labels (0–3)
  - For each video, run your VideoProcessor once to create raw_landmarks CSVs
  - Build a feature vector by concatenating ALL pixel coordinates across the selected frames:
      per-frame dims (expected):
        face_68   = 68 x + 68 y              = 136
        upper_body= (LA+RA+TORSO) (x,y,vis)  = 30   (10 pts * 3 channels)
        hands     = (L 21 + R 21) * (x,y)    = 84
      total per frame = 136 + 30 + 84 = 250
      total per sample (N frames) = 250 * N
  - Train RandomForest on Train, evaluate on Validation+Test
  - Save model to <out-dir>/models/engagement_rf.joblib
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---- Import your existing modules ----
from facial_expression_changepoint_detection.video_processing import VideoProcessor
from facial_expression_changepoint_detection.landmarks import LandmarksSignalExtractor
from facial_expression_changepoint_detection.pose_extractor import PoseSignalExtractor
from facial_expression_changepoint_detection.hand_extractor import HandSignalExtractor


# Expected per-frame dimensions (see raw_export.py headers)
EXPECTED_FACE_COLS  = 136  # 68 x's + 68 y's
EXPECTED_POSE_COLS  = 30   # (LA 3 + RA 3 + TORSO 4) * (x,y,vis) = 10*3
EXPECTED_HANDS_COLS = 84   # (L21 + R21) * (x,y) = 42*2


# ---------------------------
# Helpers: DAiSEE file system
# ---------------------------
def find_videos(root: Path) -> Dict[str, Path]:
    """Return { '5000441001.avi': full_path, ... } scanning DAiSEE/DataSet/*/*/*/*.avi"""
    vidmap: Dict[str, Path] = {}
    data_root = root / "DataSet"
    for dirpath, _, files in os.walk(data_root):
        for fn in files:
            if fn.lower().endswith(".avi"):
                vidmap[fn] = Path(dirpath) / fn
    return vidmap


def load_split_labels(root: Path) -> Dict[str, pd.DataFrame]:
    lbl_root = root / "Labels"
    splits = {
        "Train": lbl_root / "TrainLabels.csv",
        "Validation": lbl_root / "ValidationLabels.csv",
        "Test": lbl_root / "TestLabels.csv",
    }
    out = {}
    for k, p in splits.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing label file: {p}")
        df = pd.read_csv(p)
        if "ClipID" not in df.columns or "Engagement" not in df.columns:
            raise RuntimeError(f"{p} missing required columns (ClipID, Engagement)")
        out[k] = df[["ClipID", "Engagement"]].copy()
    return out


# ---------------------------------------------------
# Use your pipeline to generate raw_landmarks CSVs
# ---------------------------------------------------
def ensure_raw_landmarks(vid_path: Path, n_frames: int, out_root: Path) -> Optional[Path]:
    """
    Runs your VideoProcessor ONCE to produce raw_landmarks CSVs for this video.
    Returns the directory containing the three CSVs:
      out_root/raw_landmarks/{n}_frames/<clipid>/
    """
    out_dir = out_root
    print(f"🟦 Extracting: {vid_path}  → out_root={out_root}")
    vp = VideoProcessor(
        vid_path=vid_path,
        signal_extractors=[LandmarksSignalExtractor(), PoseSignalExtractor(), HandSignalExtractor()],
    )

    try:
        vp.select_frames_and_save_data(
            frame_count=n_frames,
            output_dir=out_dir,
            csv_path=out_dir / "changepoints.csv",
        )
    except Exception as e:
        print(f"⚠️  Skipping {vid_path.name} due to extractor error: {e}")
        return None

    raw_folder = out_dir / "raw_landmarks" / f"{n_frames}_frames" / vid_path.stem
    print(f"     ↳ Expected raw folder: {raw_folder}")
    return raw_folder if raw_folder.exists() else None


# ---------------------------------------------------
# Feature builder (pixel coordinates only) — FIXED LENGTH
# ---------------------------------------------------
def _read_face_csv(p: Path) -> Optional[pd.DataFrame]:
    files = list(p.glob(f"{p.name}_face68_*f.csv"))
    return pd.read_csv(files[0]) if files else None


def _read_pose_csv(p: Path) -> Optional[pd.DataFrame]:
    files = list(p.glob(f"{p.name}_upper_body_*f.csv"))
    return pd.read_csv(files[0]) if files else None


def _read_hands_csv(p: Path) -> Optional[pd.DataFrame]:
    files = list(p.glob(f"{p.name}_hands_*f.csv"))
    return pd.read_csv(files[0]) if files else None


def _row_vals(df: Optional[pd.DataFrame], i: int, expected_len: int) -> np.ndarray:
    """Return a 1D array of length expected_len. If df missing/short, pad with zeros."""
    if df is None or i >= len(df):
        return np.zeros((expected_len,), dtype=float)
    vals = df.iloc[i].drop(labels=["video", "frame_index"]).to_numpy(dtype=float)
    if vals.size < expected_len:
        out = np.zeros((expected_len,), dtype=float)
        out[:vals.size] = vals
        return out
    elif vals.size > expected_len:
        return vals[:expected_len]
    return vals


def build_features_from_raw_folder(raw_folder: Path, n_frames: int) -> Optional[np.ndarray]:
    """Concatenate fixed-length [face(136) + pose(30) + hands(84)] for each of n_frames."""
    if not raw_folder or not raw_folder.exists():
        print("     ❌ raw_folder missing")
        return None

    df_face = _read_face_csv(raw_folder)
    df_pose = _read_pose_csv(raw_folder)
    df_hands = _read_hands_csv(raw_folder)

    # Debug: report which CSVs exist
    print(f"     CSVs: face={'OK' if df_face is not None else 'MISSING'}, "
          f"pose={'OK' if df_pose is not None else 'MISSING'}, "
          f"hands={'OK' if df_hands is not None else 'MISSING'}")

    # Face is mandatory for stable geometry (we enforce fixed dims anyway)
    if df_face is None:
        return None

    # Sort by frame_index to align time
    key = "frame_index"
    df_face = df_face.sort_values(key)
    if df_pose is not None:  df_pose  = df_pose.sort_values(key)
    if df_hands is not None: df_hands = df_hands.sort_values(key)

    # We’ll use the number of face rows as our reference for available rows
    num_rows = len(df_face)
    rows: List[np.ndarray] = []

    for i in range(num_rows):
        fvals = _row_vals(df_face,  i, EXPECTED_FACE_COLS)
        pvals = _row_vals(df_pose,  i, EXPECTED_POSE_COLS)
        hvals = _row_vals(df_hands, i, EXPECTED_HANDS_COLS)
        rows.append(np.concatenate([fvals, pvals, hvals], axis=0))

    if not rows:
        print("     ❌ no rows assembled from CSVs")
        return None

    # Pad/trim to exactly n_frames rows
    per_frame_dim = rows[0].size
    if len(rows) < n_frames:
        pad_vec = np.zeros((per_frame_dim,), dtype=float)
        while len(rows) < n_frames:
            rows.append(pad_vec.copy())
    elif len(rows) > n_frames:
        rows = rows[:n_frames]

    feat = np.concatenate(rows, axis=0)  # shape (n_frames * per_frame_dim,)
    print(f"     ✓ feature length = {feat.size} (per-frame={per_frame_dim}, frames={n_frames})")
    return feat

# Dataset assembly

def make_split_features(
    split_df: pd.DataFrame, vidmap: Dict[str, Path], n_frames: int, out_root: Path, tag: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X_list, y_list, used_clipids = [], [], []
    kept, skipped = 0, 0

    for _, row in split_df.iterrows():
        clipid = str(row["ClipID"]).strip()
        label = int(row["Engagement"]) if not pd.isna(row["Engagement"]) else None
        if label is None:
            skipped += 1
            continue

        clipid_ext = clipid if clipid.lower().endswith(".avi") else f"{clipid}.avi"
        vid_path = vidmap.get(clipid_ext)
        if vid_path is None:
            print(f"   [{tag}] ❌ Missing video for ClipID={clipid}")
            skipped += 1
            continue

        raw_folder = ensure_raw_landmarks(vid_path, n_frames=n_frames, out_root=out_root)
        if raw_folder is None:
            print(f"   [{tag}] ❌ No raw folder for {vid_path.name}")
            skipped += 1
            continue

        feat = build_features_from_raw_folder(raw_folder, n_frames=n_frames)
        if feat is None:
            print(f"   [{tag}] ❌ Failed to build features for {vid_path.name}")
            skipped += 1
            continue

        X_list.append(feat)
        y_list.append(label)
        used_clipids.append(vid_path.name)
        kept += 1

    print(f"   [{tag}] Kept {kept} videos, skipped {skipped}")
    X = np.vstack(X_list) if X_list else np.empty((0, EXPECTED_FACE_COLS + EXPECTED_POSE_COLS + EXPECTED_HANDS_COLS))
    y = np.array(y_list, dtype=int)
    return X, y, used_clipids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daisee-root", required=True, help="Folder containing DAiSEE (with DataSet/ and Labels/)")
    ap.add_argument("--n-frames", type=int, default=5, help="How many frames to use per video (default: 5)")
    ap.add_argument("--out-dir", default="output_ml", help="Where to store raw_landmarks and models")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of videos per split for testing")
    args = ap.parse_args()

    daisee_root = Path(args.daisee_root).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"📂 DAiSEE root: {daisee_root}")
    print(f"📦 Output root: {out_root}")

    print("🔎 Scanning DAiSEE videos…")
    vidmap = find_videos(daisee_root)
    print(f"   Found {len(vidmap)} videos")

    print("📄 Loading labels…")
    splits = load_split_labels(daisee_root)

    # Optional dataset size limit
    if args.limit:
        for split_name, df in splits.items():
            splits[split_name] = df.sample(frac=1, random_state=42).head(args.limit)
            print(f"🔹 Limited {split_name} to {len(splits[split_name])} rows")

    # Extract features per split
    print("🧩 Building Train features…")
    Xtr, ytr, _ = make_split_features(splits["Train"], vidmap, args.n_frames, out_root, tag="Train")
    print(f"   Train: X={Xtr.shape}, y={ytr.shape}, label counts={np.bincount(ytr, minlength=4) if ytr.size else '[]'}")

    if Xtr.shape[0] == 0:
        print("\n❌ No training samples were built. Common causes:")
        print("   • Your extractor didn’t produce raw CSVs (face_68/upper_body/hands).")
        print(f"   • Check one video’s folder under: {out_root / 'raw_landmarks' / f'{args.n_frames}_frames'}")
        print("   • Ensure MediaPipe models exist and video paths are correct.")
        print("   • Try --limit 2 first to debug, and watch the per-video logs above.")
        return

    print("🧩 Building Validation features…")
    Xva, yva, _ = make_split_features(splits["Validation"], vidmap, args.n_frames, out_root, tag="Valid")
    print(f"   Validation: X={Xva.shape}, y={yva.shape}, label counts={np.bincount(yva, minlength=4) if yva.size else '[]'}")

    print("🧩 Building Test features…")
    Xte, yte, _ = make_split_features(splits["Test"], vidmap, args.n_frames, out_root, tag="Test")
    print(f"   Test: X={Xte.shape}, y={yte.shape}, label counts={np.bincount(yte, minlength=4) if yte.size else '[]'}")

    # ---------------- Train ----------------
    print("\n🚀 Training RandomForest (pixel features only)…")
    clf = RandomForestClassifier(
        n_estimators=600,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(Xtr, ytr)

    # ---------------- Eval ----------------
    def eval_split(name, X, y):
        if X.shape[0] == 0:
            print(f"{name}: no samples")
            return
        pred = clf.predict(X)
        print(f"\n📊 {name} Accuracy: {accuracy_score(y, pred):.4f}")
        print(classification_report(y, pred, digits=4))
        print("Confusion Matrix:\n", confusion_matrix(y, pred))

    eval_split("Validation", Xva, yva)
    eval_split("Test", Xte, yte)

    # ------ Save -------
    models_dir = out_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "engagement_rf.joblib"
    joblib.dump(clf, model_path)

    meta = {
        "n_frames": args.n_frames,
        "per_frame_dim": EXPECTED_FACE_COLS + EXPECTED_POSE_COLS + EXPECTED_HANDS_COLS,
        "feature_dim": int(Xtr.shape[1]) if Xtr.shape[0] else 0,
        "labels": {"0": "Disengaged", "1": "Low", "2": "Engaged", "3": "Highly Engaged"},
    }
    with (models_dir / "engagement_rf_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Model saved to {model_path}")
    print("Done.")


if __name__ == "__main__":
    main()
