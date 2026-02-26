#!/usr/bin/env python3
"""
Quick Model Accuracy Verifier
Loads your trained model and verifies accuracy on test data

Usage:
    python verify_accuracy.py --data your_landmarks.csv --model isl_model.pkl
"""

import pickle
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Same config as training
TEST_SIZE = 0.2
RANDOM_STATE = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='CSV file path')
    parser.add_argument('--model', required=True, help='Model file path')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔍 ACCURACY VERIFICATION")
    print("="*60)
    
    # Load data
    print(f"\n📂 Loading data: {args.data}")
    df = pd.read_csv(args.data)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(str)
    
    # Clean
    valid_mask = pd.notna(y) & (y != 'nan') & (y != '')
    X = X[valid_mask]
    y = y[valid_mask]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"✅ Loaded {len(y)} samples, {X.shape[1]} features")
    print(f"   Classes: {len(np.unique(y))}")
    
    # Split EXACTLY like training (same random_state!)
    print(f"\n✂️  Splitting with same random_state as training...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"   Train: {len(X_train)} samples (80%)")
    print(f"   Test:  {len(X_test)} samples (20%)")
    
    # Load model
    print(f"\n📂 Loading model: {args.model}")
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded")
    
    # Test on TRAINING data
    print(f"\n🎯 Testing on TRAINING data...")
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"   Training Accuracy: {train_acc*100:.2f}%")
    
    # Test on TEST data (the real accuracy!)
    print(f"\n🎯 Testing on TEST data (80-20 split)...")
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"   Testing Accuracy:  {test_acc*100:.2f}%")
    
    # Analysis
    print(f"\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    print(f"Training Accuracy: {train_acc*100:.2f}%")
    print(f"Testing Accuracy:  {test_acc*100:.2f}%")
    print(f"Difference:        {(train_acc - test_acc)*100:.2f}%")
    
    if test_acc >= 0.90:
        print(f"\n✅ EXCELLENT! Model is highly accurate.")
    elif test_acc >= 0.85:
        print(f"\n✅ GREAT! Model meets target accuracy.")
    elif test_acc >= 0.75:
        print(f"\n⚠️  ACCEPTABLE but could be improved.")
    else:
        print(f"\n❌ POOR! Model needs retraining.")
    
    if train_acc - test_acc > 0.15:
        print(f"⚠️  WARNING: Overfitting detected (train >> test)")
    
    # Per-class breakdown
    print(f"\n📊 Per-Class Accuracy (on test set):")
    classes = sorted(np.unique(y_test))
    for cls in classes:
        mask = y_test == cls
        if mask.sum() > 0:
            cls_acc = accuracy_score(y_test[mask], y_test_pred[mask])
            status = "✅" if cls_acc >= 0.9 else "⚠️ " if cls_acc >= 0.8 else "❌"
            print(f"   {cls:>3}: {cls_acc*100:>6.2f}% {status}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()