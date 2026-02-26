import os
import sys
import argparse
import pickle
import time
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# CONFIG
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_ESTIMATORS = 100
DEFAULT_RANDOM_STATE = 42

def load_data(csv_path):
    """Load two-hand landmark data (126 features + 1 label)."""
    print(f"📂 Loading data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: File not found: {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} samples")
    print(f"📊 Total columns: {len(df.columns)}")
    
    # Check format
    if len(df.columns) != 127:
        print(f"⚠️  WARNING: Expected 127 columns (126 features + 1 label), got {len(df.columns)}")
        print(f"   Continuing anyway...")
    
    # Extract features and labels
    # Assuming label is in the LAST column
    X = df.iloc[:, :-1].values  # All columns except last (126 features)
    y = df.iloc[:, -1].values   # Last column (label)
    
    # Convert labels to string
    y = y.astype(str)
    
    # Remove NaN labels
    valid_mask = pd.notna(y) & (y != 'nan') & (y != '')
    if not valid_mask.all():
        print(f"⚠️  Removing {(~valid_mask).sum()} rows with missing labels")
        X = X[valid_mask]
        y = y[valid_mask]
    
    # Clean features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"📊 Features shape: {X.shape}")
    print(f"📊 Labels shape: {y.shape}")
    print(f"📊 Expected features: 126 (63 left + 63 right)")
    
    if X.shape[1] != 126:
        print(f"❌ ERROR: Expected 126 features, got {X.shape[1]}")
        print(f"   Make sure your CSV has:")
        print(f"   - 63 left hand features")
        print(f"   - 63 right hand features")
        print(f"   - 1 label column")
        sys.exit(1)
    
    return X, y, df


def analyze_dataset(X, y):
    """Analyze dataset."""
    print("\n" + "="*60)
    print("📊 DATASET ANALYSIS")
    print("="*60)
    
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"\n📌 Total samples: {len(y)}")
    print(f"📌 Number of classes: {len(unique_labels)}")
    
    # Sort labels
    try:
        sorted_labels = sorted(unique_labels)
    except:
        sorted_labels = list(unique_labels)
    
    print(f"📌 Classes: {sorted_labels}")
    
    print(f"\n📈 Samples per class:")
    label_counts = dict(zip(unique_labels, counts))
    
    for label in sorted_labels:
        count = label_counts[label]
        bar = "█" * (count // 100)
        print(f"   {label:>3}: {count:>5} samples {bar}")
    
    # Check balance
    min_samples = counts.min()
    max_samples = counts.max()
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    
    if imbalance_ratio > 3:
        print(f"\n⚠️  Class imbalance detected!")
        print(f"   Min: {min_samples}, Max: {max_samples}, Ratio: {imbalance_ratio:.2f}x")
    else:
        print(f"\n✅ Classes reasonably balanced (ratio: {imbalance_ratio:.2f}x)")
    
    print("="*60 + "\n")


def train_random_forest(X_train, y_train, n_estimators=100):
    """Train RandomForest on two-hand data."""
    print("="*60)
    print("🌲 TRAINING TWO-HAND RANDOM FOREST MODEL")
    print("="*60)
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Features: {X_train.shape[1]} (63 left + 63 right)")
    print(f"📊 Estimators: {n_estimators}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"\n⏳ Training model...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"✅ Training complete in {training_time:.2f} seconds")
    print("="*60 + "\n")
    
    return model, training_time


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model."""
    print("="*60)
    print("📈 MODEL EVALUATION")
    print("="*60)
    
    # Training accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\n🎯 Training Accuracy: {train_accuracy*100:.2f}%")
    
    # Testing accuracy
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"🎯 Testing Accuracy:  {test_accuracy*100:.2f}%")
    
    # Analysis
    accuracy_diff = train_accuracy - test_accuracy
    if accuracy_diff > 0.1:
        print(f"\n⚠️  WARNING: Possible overfitting!")
        print(f"   Train-Test difference: {accuracy_diff*100:.2f}%")
    elif test_accuracy > 0.90:
        print(f"\n✅ EXCELLENT! Two-hand model is highly accurate!")
    elif test_accuracy > 0.85:
        print(f"\n✅ Great performance!")
    elif test_accuracy > 0.75:
        print(f"\n✅ Good performance, acceptable for ISL")
    else:
        print(f"\n⚠️  Performance could be improved")
    
    print("\n" + "="*60)
    print("📊 DETAILED CLASSIFICATION REPORT")
    print("="*60)
    
    unique_classes = np.unique(np.concatenate([y_test, y_test_pred]))
    report = classification_report(
        y_test,
        y_test_pred,
        labels=unique_classes,
        zero_division=0
    )
    print(report)
    print("="*60 + "\n")
    
    return train_accuracy, test_accuracy


def save_model(model, model_path):
    """Save model."""
    print(f"💾 Saving model to: {model_path}")
    
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅ Model saved! File size: {file_size_mb:.2f} MB")
    
    # Test loading
    with open(model_path, 'rb') as f:
        test_load = pickle.load(f)
    print(f"✅ Model loading test: PASSED\n")


def save_training_info(output_path, model, train_accuracy, test_accuracy, 
                       n_estimators, training_time, class_names):
    """Save training info."""
    print(f"📝 Saving training info to: {output_path}")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ISL TWO-HAND MODEL TRAINING REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL CONFIGURATION\n")
        f.write("-"*60 + "\n")
        f.write(f"Model Type: RandomForestClassifier\n")
        f.write(f"Number of Estimators: {n_estimators}\n")
        f.write(f"Features: 126 (63 left hand + 63 right hand)\n")
        f.write(f"Number of Classes: {len(class_names)}\n")
        f.write(f"Classes: {', '.join(map(str, sorted(class_names)))}\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*60 + "\n")
        f.write(f"Training Accuracy: {train_accuracy*100:.2f}%\n")
        f.write(f"Testing Accuracy:  {test_accuracy*100:.2f}%\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n\n")
        
        f.write("FEATURE IMPORTANCE (Top 10)\n")
        f.write("-"*60 + "\n")
        feature_importance = model.feature_importances_
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        
        for idx in top_indices:
            # Determine if left or right hand
            if idx < 63:
                hand = "L"
                landmark_idx = idx // 3
            else:
                hand = "R"
                landmark_idx = (idx - 63) // 3
            
            coord = ['x', 'y', 'z'][idx % 3]
            f.write(f"  {hand}_lm{landmark_idx}_{coord}: {feature_importance[idx]:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"✅ Training info saved!\n")


def main():
    parser = argparse.ArgumentParser(description='Train two-hand ISL model')
    parser.add_argument('--data', type=str, required=True, help='Path to 127-column CSV')
    parser.add_argument('--model', type=str, default='isl_model.pkl', help='Output model path')
    parser.add_argument('--n-estimators', type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument('--test-size', type=float, default=DEFAULT_TEST_SIZE)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ISL MODEL TRAINING")
    print("="*60)
    print(f"Person B - Model Lead")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Load
    X, y, df = load_data(args.data)
    
    # Analyze
    analyze_dataset(X, y)
    
    # Split
    print(f"✂️  Splitting data: {int((1-args.test_size)*100)}% train, {int(args.test_size*100)}% test")
    
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    
    if min_count < 2:
        print(f"⚠️  Some classes have < 2 samples, using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=DEFAULT_RANDOM_STATE
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=DEFAULT_RANDOM_STATE,
            stratify=y
        )
    
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing:  {len(X_test)} samples\n")
    
    # Train
    model, training_time = train_random_forest(X_train, y_train, args.n_estimators)
    
    # Evaluate
    train_acc, test_acc = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Save
    save_model(model, args.model)
    
    # Save info
    info_path = args.model.replace('.pkl', '_info.txt')
    class_names = np.unique(y)
    save_training_info(info_path, model, train_acc, test_acc,
                      args.n_estimators, training_time, class_names)
    
    # Summary
    print("="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"✅ Model saved: {args.model}")
    print(f"✅ Info saved:  {info_path}")
    print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
    print(f"⏱️  Total time: {training_time:.2f} seconds")
    print(f"\n📝 Model details:")
    print(f"   - Features: 126 (63 left + 63 right hand)")
    print(f"   - Classes: {len(class_names)} (A-Z, 0-9)")
    print(f"   - Estimators: {args.n_estimators}")
    print("\n🚀 Next steps:")
    print("   1. Check accuracy in info file")
    print("   2. Use main_two_hands.py for live testing")
    print("   3. If accuracy < 85%, collect more data")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()