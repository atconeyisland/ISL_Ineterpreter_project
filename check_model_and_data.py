#!/usr/bin/env python3
"""
Model & Data Checker
Checks both your trained model accuracy AND your training data quality

Usage:
    python check_model_and_data.py --data your_landmarks.csv --model isl_model.pkl
"""

import pickle
import pandas as pd
import numpy as np
import argparse
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold

def load_model(model_path):
    """Load trained model."""
    print(f"📂 Loading model: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Classes: {len(model.classes_)}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)


def load_data(csv_path):
    """Load training data."""
    print(f"\n📂 Loading data: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Data loaded successfully")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        # Extract features and labels
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.astype(str)
        
        # Clean
        valid_mask = pd.notna(y) & (y != 'nan') & (y != '')
        X = X[valid_mask]
        y = y[valid_mask]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, y, df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)


def check_data_quality(X, y):
    """Check training data quality."""
    print("\n" + "="*70)
    print("📊 TRAINING DATA QUALITY CHECK")
    print("="*70)
    
    # Basic stats
    print(f"\n📌 Dataset Statistics:")
    print(f"   Total samples: {len(y)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📈 Class Distribution:")
    class_dist = dict(zip(unique, counts))
    
    for label in sorted(unique):
        count = class_dist[label]
        bar = "█" * (count // 50)
        print(f"   {label:>3}: {count:>5} samples {bar}")
    
    # Check for issues
    min_samples = counts.min()
    max_samples = counts.max()
    mean_samples = counts.mean()
    
    print(f"\n📊 Class Balance:")
    print(f"   Min samples: {min_samples}")
    print(f"   Max samples: {max_samples}")
    print(f"   Mean samples: {mean_samples:.1f}")
    print(f"   Imbalance ratio: {max_samples/min_samples:.2f}x")
    
    if max_samples / min_samples > 5:
        print(f"   ⚠️  SEVERE class imbalance! Some signs underrepresented.")
    elif max_samples / min_samples > 3:
        print(f"   ⚠️  Moderate class imbalance.")
    else:
        print(f"   ✅ Good class balance.")
    
    # Feature statistics
    print(f"\n📊 Feature Statistics:")
    print(f"   Mean: {X.mean():.4f}")
    print(f"   Std:  {X.std():.4f}")
    print(f"   Min:  {X.min():.4f}")
    print(f"   Max:  {X.max():.4f}")
    
    # Check for zeros (missing hands)
    zero_rows = (X == 0).all(axis=1).sum()
    print(f"\n📊 Data Completeness:")
    print(f"   Rows with all zeros: {zero_rows} ({zero_rows/len(X)*100:.1f}%)")
    
    if zero_rows > len(X) * 0.1:
        print(f"   ⚠️  More than 10% of samples are all zeros!")
        print(f"   This suggests missing hand data.")
    
    # Check left vs right hand presence (first 63 vs last 63)
    if X.shape[1] == 126:
        left_zeros = (X[:, :63] == 0).all(axis=1).sum()
        right_zeros = (X[:, 63:] == 0).all(axis=1).sum()
        both_present = len(X) - left_zeros - right_zeros
        
        print(f"\n📊 Two-Hand Analysis:")
        print(f"   Both hands present: {both_present} ({both_present/len(X)*100:.1f}%)")
        print(f"   Left hand missing:  {left_zeros} ({left_zeros/len(X)*100:.1f}%)")
        print(f"   Right hand missing: {right_zeros} ({right_zeros/len(X)*100:.1f}%)")
        
        if both_present < len(X) * 0.7:
            print(f"   ⚠️  Less than 70% of samples have both hands!")
            print(f"   This may hurt accuracy for two-hand signs.")


def check_model_on_training_data(model, X, y):
    """Check model accuracy on FULL training data."""
    print("\n" + "="*70)
    print("🎯 MODEL ACCURACY ON TRAINING DATA")
    print("="*70)
    
    print(f"\n⏳ Predicting on {len(X)} samples...")
    
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"\n✅ Training Data Accuracy: {accuracy*100:.2f}%")
    
    if accuracy == 1.0:
        print(f"   ⚠️  WARNING: 100% accuracy on training data!")
        print(f"   This is suspicious - might indicate:")
        print(f"   1. Model memorized data (overfitting)")
        print(f"   2. Data leakage (test samples in training)")
        print(f"   3. Perfect dataset (rare but possible)")
    elif accuracy > 0.95:
        print(f"   ⚠️  Very high accuracy - check for overfitting")
    elif accuracy > 0.90:
        print(f"   ✅ Excellent training accuracy")
    elif accuracy > 0.85:
        print(f"   ✅ Good training accuracy")
    else:
        print(f"   ⚠️  Low training accuracy - model struggles to learn")
    
    # Per-class accuracy
    print(f"\n📊 Per-Class Accuracy on Training Data:")
    unique_classes = sorted(np.unique(y))
    
    for cls in unique_classes:
        mask = y == cls
        if mask.sum() == 0:
            continue
        cls_acc = accuracy_score(y[mask], y_pred[mask])
        status = "✅" if cls_acc > 0.9 else "⚠️ " if cls_acc > 0.8 else "❌"
        print(f"   {cls:>3}: {cls_acc*100:>6.2f}% {status}")
    
    # Find worst performing classes
    worst_classes = []
    for cls in unique_classes:
        mask = y == cls
        if mask.sum() == 0:
            continue
        cls_acc = accuracy_score(y[mask], y_pred[mask])
        if cls_acc < 0.9:
            worst_classes.append((cls, cls_acc))
    
    if worst_classes:
        print(f"\n⚠️  Classes with <90% accuracy on training data:")
        for cls, acc in sorted(worst_classes, key=lambda x: x[1]):
            print(f"   {cls}: {acc*100:.2f}%")


def cross_validate_model(model, X, y):
    """Run cross-validation to get REAL accuracy estimate."""
    print("\n" + "="*70)
    print("🔄 CROSS-VALIDATION (Real-World Accuracy Estimate)")
    print("="*70)
    
    print(f"\n⏳ Running 5-fold cross-validation...")
    print(f"   This tests model on unseen data...\n")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        print(f"📊 Cross-Validation Results:")
        for i, score in enumerate(scores, 1):
            print(f"   Fold {i}: {score*100:.2f}%")
        
        print(f"\n📊 Summary:")
        print(f"   Mean CV Accuracy: {scores.mean()*100:.2f}%")
        print(f"   Std Deviation:    ±{scores.std()*100:.2f}%")
        print(f"   Min:  {scores.min()*100:.2f}%")
        print(f"   Max:  {scores.max()*100:.2f}%")
        
        print(f"\n🎯 VERDICT:")
        if scores.mean() > 0.95:
            print(f"   ✅ EXCELLENT! Model generalizes very well.")
        elif scores.mean() > 0.90:
            print(f"   ✅ GREAT! Model is highly accurate.")
        elif scores.mean() > 0.85:
            print(f"   ✅ GOOD! Model is production-ready.")
        elif scores.mean() > 0.75:
            print(f"   ⚠️  ACCEPTABLE but could be improved.")
        else:
            print(f"   ❌ POOR! Model needs improvement.")
            print(f"   Consider: more data, better features, or different model.")
        
        # Check variance
        if scores.std() > 0.05:
            print(f"\n⚠️  High variance between folds!")
            print(f"   Model performance is inconsistent.")
            print(f"   Possible causes:")
            print(f"   - Small dataset")
            print(f"   - Class imbalance")
            print(f"   - Overfitting to specific samples")
        
        return scores.mean()
        
    except Exception as e:
        print(f"❌ Cross-validation failed: {e}")
        return None


def confusion_analysis(model, X, y):
    """Analyze confusion matrix."""
    print("\n" + "="*70)
    print("🔀 CONFUSION MATRIX ANALYSIS")
    print("="*70)
    
    y_pred = model.predict(X)
    classes = sorted(np.unique(y))
    cm = confusion_matrix(y, y_pred, labels=classes)
    
    # Find most confused pairs
    confused_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i][j] > 0:
                confused_pairs.append((classes[i], classes[j], cm[i][j]))
    
    if not confused_pairs:
        print(f"\n✅ No confusions! Model predicts all signs correctly.")
    else:
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\n⚠️  Top 10 Most Confused Sign Pairs:")
        print(f"{'True Sign':<12} {'Predicted As':<15} {'Times':<10} {'% of True'}")
        print("-"*60)
        
        for true_sign, pred_sign, count in confused_pairs[:10]:
            true_count = (y == true_sign).sum()
            pct = count / true_count * 100
            print(f"{true_sign:<12} {pred_sign:<15} {count:<10} {pct:.1f}%")


def final_summary(cv_accuracy, train_accuracy):
    """Print final summary."""
    print("\n" + "="*70)
    print("📋 FINAL SUMMARY")
    print("="*70)
    
    print(f"\n📊 Accuracy Comparison:")
    print(f"   Training Data Accuracy:      {train_accuracy*100:.2f}%")
    if cv_accuracy:
        print(f"   Cross-Validation Accuracy:   {cv_accuracy*100:.2f}%")
        print(f"   Difference:                  {(train_accuracy - cv_accuracy)*100:.2f}%")
        
        if train_accuracy - cv_accuracy > 0.15:
            print(f"\n⚠️  OVERFITTING DETECTED!")
            print(f"   Model memorizes training data but doesn't generalize well.")
            print(f"   Real-world performance will be closer to {cv_accuracy*100:.1f}%")
        elif cv_accuracy > 0.90:
            print(f"\n✅ MODEL IS READY FOR DEPLOYMENT!")
            print(f"   Real-world accuracy expected: ~{cv_accuracy*100:.1f}%")
        elif cv_accuracy > 0.85:
            print(f"\n✅ MODEL IS GOOD ENOUGH FOR PROJECT")
            print(f"   Real-world accuracy expected: ~{cv_accuracy*100:.1f}%")
        else:
            print(f"\n⚠️  MODEL NEEDS IMPROVEMENT")
            print(f"   Current accuracy too low for reliable use")


def main():
    parser = argparse.ArgumentParser(description='Check model and data quality')
    parser.add_argument('--data', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔍 ISL MODEL & DATA QUALITY CHECKER")
    print("="*70)
    
    # Load
    model = load_model(args.model)
    X, y, df = load_data(args.data)
    
    # Check data quality
    check_data_quality(X, y)
    
    # Check model on training data
    check_model_on_training_data(model, X, y)
    train_accuracy = accuracy_score(y, model.predict(X))
    
    # Cross-validation (real accuracy)
    cv_accuracy = cross_validate_model(model, X, y)
    
    # Confusion analysis
    confusion_analysis(model, X, y)
    
    # Final summary
    final_summary(cv_accuracy, train_accuracy)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()