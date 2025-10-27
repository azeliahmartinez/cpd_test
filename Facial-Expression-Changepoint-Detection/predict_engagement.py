"""
ENHANCED PREDICT: Predict engagement (0–3) and generate all outputs (CSV, frames, landmarks)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import joblib
import pandas as pd
from typing import Dict, List, Optional

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
    """Build features from the generated CSV files (same as training pipeline)"""
    print("🔨 Building features from CSVs...")
    
    def read_csv(p: Path, pattern: str) -> Optional[pd.DataFrame]:
        files = list(p.glob(pattern))
        return pd.read_csv(files[0]) if files else None
    
    # Read generated CSV files
    df_face = read_csv(raw_dir, f"*_face68_*f.csv")
    df_pose = read_csv(raw_dir, f"*_upper_body_*f.csv") 
    df_hands = read_csv(raw_dir, f"*_hands_*f.csv")
    
    print(f"📊 CSVs found - Face: {'Yes' if df_face is not None else 'No'}, "
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
        pad_vec = np.zeros(EXPECTED_FACE_COLS + EXPECTED_POSE_COLS + EXPECTED_HANDS_COLS)
        while len(rows) < n_frames:
            rows.append(pad_vec.copy())
    elif len(rows) > n_frames:
        rows = rows[:n_frames]
    
    features = np.concatenate(rows)
    print(f"✅ Features built: {features.shape}")
    return features

def save_prediction_summary(vid_path: Path, prediction: int, probabilities: np.ndarray, 
                          outputs: Dict, out_dir: Path, meta: Dict):
    """Save comprehensive prediction summary"""
    summary_file = out_dir / f"prediction_summary_{vid_path.stem}.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("PREDICTION SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("VIDEO INFORMATION:\n")
        f.write(f"  Video: {vid_path.name}\n")
        f.write(f"  Path: {vid_path}\n\n")
        
        f.write("PREDICTION RESULTS:\n")
        labels = meta.get("labels", {"0": "Disengaged", "1": "Low", "2": "Engaged", "3": "Highly Engaged"})
        f.write(f"  Predicted Engagement: {prediction} - {labels.get(str(prediction), 'Unknown')}\n")
        f.write("  Class Probabilities:\n")
        for i, prob in enumerate(probabilities):
            label_name = labels.get(str(i), f"Class_{i}")
            f.write(f"    {i} ({label_name}): {prob:.4f} ({prob*100:.2f}%)\n")
        f.write("\n")
        
        f.write("GENERATED OUTPUTS:\n")
        f.write(f"  Raw Landmarks Directory: {outputs['raw_dir']}\n")
        f.write(f"  Extracted Frames Directory: {outputs['frames_dir']}\n")
        f.write("  CSV Files Generated:\n")
        for file_type, file_path in outputs.get('csv_files', {}).items():
            f.write(f"    - {file_type}: {file_path.name}\n")
        f.write(f"  Change Points (frame indices): {outputs.get('changepoints', [])}\n")
        f.write(f"  Frames Extracted: {len(outputs.get('frames', []))}\n")
        f.write(f"  Feature Vector Dimension: {outputs['features'].shape}\n")
    
    print(f"📄 Summary saved: {summary_file}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Video filename (e.g., Brodeth_Charlize_clip_01.mp4)")
    ap.add_argument("--model-dir", default="output_ml/models")
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--out-dir", default="predictions")
    ap.add_argument("--generate-outputs", action="store_true", default=True, 
                   help="Generate CSV files and extracted frames")
    args = ap.parse_args()

    # Load model
    model_dir = Path(args.model_dir)
    try:
        clf = joblib.load(model_dir / "engagement_rf.joblib")
        print(f"✅ Model loaded. Classes: {clf.classes_}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load metadata
    meta_path = model_dir / "engagement_rf_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"labels": {"0": "Disengaged", "1": "Low", "2": "Engaged", "3": "Highly Engaged"}}

    # Get video path
    try:
        vid_path = get_video_path(args.video)
    except FileNotFoundError as e:
        print(e)
        return

    # Create output directory
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n🎯 STARTING PREDICTION PIPELINE")
    print(f"📹 Video: {vid_path.name}")
    print(f"📁 Output directory: {out_root}")
    print(f"🔄 Generating outputs: {args.generate_outputs}")
    print("=" * 50)

    try:
        if args.generate_outputs:
            # Enhanced pipeline: Generate all outputs
            outputs = ensure_raw_landmarks_and_frames(vid_path, args.n_frames, out_root)
            features = outputs["features"]
        else:
            # Simple pipeline: Just extract features
            features = build_features_direct(vid_path, args.n_frames)
            outputs = {"features": features}
        
        # Ensure correct feature dimension
        expected_dim = args.n_frames * 250
        if features.shape[0] != expected_dim:
            print(f"🔧 Adjusting features to match expected dimension")
            if features.shape[0] < expected_dim:
                features = np.pad(features, (0, expected_dim - features.shape[0]))
            else:
                features = features[:expected_dim]
        
        # Make prediction
        prediction = clf.predict(features.reshape(1, -1))[0]
        probabilities = clf.predict_proba(features.reshape(1, -1))[0]
        
        # Display results
        labels = meta.get("labels", {"0": "Disengaged", "1": "Low", "2": "Engaged", "3": "Highly Engaged"})
        
        print(f"\n" + "="*50)
        print(f"🎯 PREDICTION RESULTS")
        print(f"="*50)
        print(f"📹 Video: {vid_path.name}")
        print(f"🔮 Predicted engagement: {prediction} - {labels.get(str(prediction), 'Unknown')}")
        print(f"📊 Class probabilities:")
        
        for i, prob in enumerate(probabilities):
            label_name = labels.get(str(i), f"Class_{i}")
            print(f"   {i} ({label_name}): {prob:.3f} ({prob*100:.1f}%)")
        
        # Save comprehensive summary
        outputs["features"] = features  # Store final features
        save_prediction_summary(vid_path, prediction, probabilities, outputs, out_root, meta)
        
        print(f"\n✅ PREDICTION COMPLETE")
        print(f"📄 Summary report generated")
        if args.generate_outputs:
            print(f"📊 Landmark CSVs generated")
            print(f"🖼️  Frame images extracted")
        print(f"="*50)
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

# Fallback direct feature extraction (if CSV generation fails)
def build_features_direct(vid_path: Path, n_frames: int) -> np.ndarray:
    """Fallback: Direct feature extraction without CSV generation"""
    print("🔄 Using direct feature extraction...")
    
    extractors = {
        "face": LandmarksSignalExtractor(),
        "pose": PoseSignalExtractor(),
        "hands": HandSignalExtractor()
    }
    
    signals = {}
    for name, extractor in extractors.items():
        try:
            signal = extractor.extract_signal(vid_path)
            signals[name] = signal if signal is not None else np.zeros((100, 
                EXPECTED_FACE_COLS if name == "face" else EXPECTED_POSE_COLS if name == "pose" else EXPECTED_HANDS_COLS))
        except Exception as e:
            print(f"⚠️  {name} extractor failed: {e}")
            signals[name] = np.zeros((100, 
                EXPECTED_FACE_COLS if name == "face" else EXPECTED_POSE_COLS if name == "pose" else EXPECTED_HANDS_COLS))
    
    min_frames = min(s.shape[0] for s in signals.values())
    combined = np.concatenate([s[:min_frames] for s in signals.values()], axis=1)
    
    if combined.shape[0] >= n_frames:
        indices = np.linspace(0, combined.shape[0]-1, n_frames, dtype=int)
        selected = combined[indices]
    else:
        selected = np.zeros((n_frames, combined.shape[1]))
        selected[:combined.shape[0]] = combined
    
    return selected.flatten()

if __name__ == "__main__":
    main()