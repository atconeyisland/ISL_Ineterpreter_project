import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import os
import sys
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm

MP_MODEL_PATH = "hand_landmarker.task"

def download_mp_model():
    if os.path.exists(MP_MODEL_PATH):
        return
    print("📥 Downloading MediaPipe model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MP_MODEL_PATH)
    print("✅ Downloaded!")

def extract_landmarks(image_path, detector):
    """Extract landmarks from both hands."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    results = detector.detect(mp_image)
    
    if not results.hand_landmarks or not results.handedness:
        return None, None
    
    left_coords = None
    right_coords = None
    
    for hand_lms, handedness in zip(results.hand_landmarks, results.handedness):
        hand_label = handedness[0].category_name
        
        coords = []
        for lm in hand_lms:
            coords.extend([lm.x, lm.y, lm.z])
        
        if hand_label == "Left":
            left_coords = coords
        elif hand_label == "Right":
            right_coords = coords
    
    return left_coords, right_coords

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input data folder')
    parser.add_argument('--output', default='isl_landmarks.csv', help='Output CSV')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🤟 ISL TWO-HAND LANDMARK EXTRACTION")
    print("="*60)
    
    # Setup MediaPipe
    download_mp_model()
    base_options = python.BaseOptions(model_asset_path=MP_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.3,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Scan folders
    input_path = Path(args.input)
    print(f"\n📂 Scanning: {args.input}")
    
    # Expected: data/1/, data/2/, ..., data/A/, data/B/, ...
    image_files = []
    labels = []
    
    for class_folder in sorted(input_path.iterdir()):
        if not class_folder.is_dir():
            continue
        
        label = class_folder.name
        for img_file in sorted(class_folder.glob("*.jpg")):
            image_files.append(img_file)
            labels.append(label)
    
    print(f"✅ Found {len(image_files)} images across {len(set(labels))} classes")
    print(f"   Classes: {sorted(set(labels))}")
    
    # Extract landmarks
    print(f"\n⏳ Extracting landmarks from {len(image_files)} images...")
    print(f"   (This will take ~15-30 minutes for 42,000 images)")
    
    data = []
    failed = 0
    one_hand = 0
    two_hands = 0
    
    for img_path, label in tqdm(zip(image_files, labels), total=len(image_files)):
        left_coords, right_coords = extract_landmarks(img_path, detector)
        
        # Skip if NO hands detected
        if left_coords is None and right_coords is None:
            failed += 1
            continue
        
        # Fill with zeros if one hand missing
        if left_coords is None:
            left_coords = [0.0] * 63
            one_hand += 1
        if right_coords is None:
            right_coords = [0.0] * 63
            one_hand += 1
        else:
            two_hands += 1
        
        # Combine: 63 left + 63 right + 1 label = 127 columns
        row = left_coords + right_coords + [label]
        data.append(row)
    
    # Create CSV headers
    left_headers = [f"L_lm{i}_{coord}" for i in range(21) for coord in ['x','y','z']]
    right_headers = [f"R_lm{i}_{coord}" for i in range(21) for coord in ['x','y','z']]
    columns = left_headers + right_headers + ['label']
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Save
    print(f"\n💾 Saving to: {args.output}")
    df.to_csv(args.output, index=False)
    
    if len(image_files) == 0:
        print("❌ No images found! Check your --input path.")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*60)
    print("✅ EXTRACTION COMPLETE!")
    print("="*60)
    print(f"Total images: {len(image_files)}")
    print(f"✓ Extracted: {len(data)} ({len(data)/len(image_files)*100:.1f}%)")
    print(f"  ↳ Both hands: {two_hands}")
    print(f"  ↳ One hand:   {one_hand}")
    print(f"✗ Failed:     {failed} ({failed/len(image_files)*100:.1f}%)")
    print(f"\n📁 Output: {args.output}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)} (63 left + 63 right + 1 label)")
    print(f"   Classes: {df['label'].nunique()}")
    print(f"\n📊 Class distribution (first 10):")
    for label, count in list(df['label'].value_counts().items())[:10]:
        print(f"   {label}: {count}")
    print("="*60 + "\n")
    
    detector.close()

if __name__ == "__main__":
    main()