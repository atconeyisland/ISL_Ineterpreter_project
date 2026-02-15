#!/usr/bin/env python3
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

DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_ESTIMATORS = 100
DEFAULT_RANDOM_STATE = 42

def load_data(csv_path):
    """Load landmark data from CSV file."""
    print(f"Loading data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    print(f"Total columns: {len(df.columns)}")
    
    # Check if label column exists by name
    if 'label' in df.columns:
        print(f"Found 'label' column")
        y = df['label'].values
        X = df.drop('label', axis=1).values
    else:
        # Assume label is last column
        print(f"Assuming label is in last column")
        X = df.iloc[:, :-1].values  # All columns except last
        y = df.iloc[:, -1].values    # Last column
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Remove NaN/None from labels
    valid_mask = pd.notna(y)
    if not valid_mask.all():
        print(f"WARNING: Found {(~valid_mask).sum()} rows with missing labels, removing them")
        X = X[valid_mask]
        y = y[valid_mask]
        print(f"After removing NaN: {len(y)} samples")
    
    # Convert labels to string to avoid sorting issues
    y = y.astype(str)
    
    try:
        unique_labels = sorted(set(y))
        print(f"Unique labels ({len(unique_labels)}): {unique_labels[:20]}")
        if len(unique_labels) > 20:
            print(f"    ... and {len(unique_labels) - 20} more")
    except:
        print(f"Unique labels: {set(y)}")
    
    return X, y, df

def analyze_dataset(X, y):
    """Analyze dataset."""
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"\nTotal samples: {len(y)}")
    print(f"Number of classes: {len(unique_labels)}")
    
    # Sort labels if possible
    try:
        sorted_labels = sorted(unique_labels)
    except:
        sorted_labels = unique_labels
    
    print(f"Classes: {sorted_labels[:30]}")
    if len(sorted_labels) > 30:
        print(f"    ... and {len(sorted_labels) - 30} more")
    
    print(f"\nSamples per class (first 20):")
    label_counts = dict(zip(unique_labels, counts))
    
    # Show first 20 labels
    for i, label in enumerate(sorted_labels[:20]):
        count = label_counts[label]
        bar = "█" * (count // 100)
        print(f"   {label:>10}: {count:>5} samples {bar}")
    
    if len(unique_labels) > 20:
        print(f"   ... and {len(unique_labels) - 20} more classes")
    
    print("="*60 + "\n")

def train_random_forest(X_train, y_train, n_estimators=100):
    """Train RandomForest."""
    print("="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    print(f"Number of estimators: {n_estimators}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"\nTraining model...")
    start_time = time.time()
    
    # Clean data
    X_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    model.fit(X_clean, y_train)
    
    training_time = time.time() - start_time
    print(f"Training complete in {training_time:.2f} seconds")
    print("="*60 + "\n")
    
    return model, training_time

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model."""
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Clean data
    X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_clean = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Training accuracy
    y_train_pred = model.predict(X_train_clean)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining Accuracy: {train_accuracy*100:.2f}%")
    
    # Testing accuracy
    y_test_pred = model.predict(X_test_clean)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Testing Accuracy:  {test_accuracy*100:.2f}%")
    
    if test_accuracy > 0.85:
        print(f"\nExcellent performance! Model is ready!")
    elif test_accuracy > 0.75:
        print(f"\nGood performance!")
    else:
        print(f"\nPerformance could be improved.")
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (First 20 classes)")
    print("="*60)
    
    # Show report for subset if too many classes
    unique_test_labels = np.unique(y_test)
    if len(unique_test_labels) > 20:
        labels_to_show = unique_test_labels[:20]
    else:
        labels_to_show = unique_test_labels
    
    report = classification_report(
        y_test, 
        y_test_pred, 
        labels=labels_to_show,
        zero_division=0
    )
    print(report)
    
    if len(unique_test_labels) > 20:
        print(f"... and {len(unique_test_labels) - 20} more classes")
    
    print("="*60 + "\n")
    
    return train_accuracy, test_accuracy

def save_model(model, model_path):
    """Save model."""
    print(f"Saving model to: {model_path}")
    
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model saved! File size: {file_size_mb:.2f} MB\n")

def main():
    parser = argparse.ArgumentParser(description='Train ISL classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV')
    parser.add_argument('--model', type=str, default='isl_model.pkl', help='Output path')
    parser.add_argument('--n-estimators', type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument('--test-size', type=float, default=DEFAULT_TEST_SIZE)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ISL INTERPRETER - MODEL TRAINING")
    print("="*60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Load data
    X, y, df = load_data(args.data)
    
    # Analyze
    analyze_dataset(X, y)
    
    # Split (80-20)
    print(f"Splitting data: {int((1-args.test_size)*100)}% train, {int(args.test_size*100)}% test")
    
    # Check if we have enough samples per class for stratify
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    
    if min_count < 2:
        print(f"Some classes have < 2 samples, using random split (no stratify)")
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
    
    # Summary
    print("="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved: {args.model}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Time: {training_time:.2f} seconds")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()