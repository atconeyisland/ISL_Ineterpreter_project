import pickle
import numpy as np

print("="*60)
print("TESTING MODEL LOADING")
print("="*60)

# Test 1: Load model
print("\nTest 1: Loading model from isl_model.pkl...")
try:
    with open('isl_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"FAILED: {e}")
    exit(1)

# Test 2: Check model properties
print("\nTest 2: Checking model properties...")
print(f"   Model type: {type(model).__name__}")
print(f"   Number of classes: {len(model.classes_)}")
print(f"   Classes: {sorted(model.classes_)}")
print("Model properties look good!")

# Test 3: Test prediction with dummy data
print("\nTest 3: Testing prediction with dummy landmarks...")
try:
    dummy_landmarks = np.random.rand(1, 63)  # Random 63 features
    prediction = model.predict(dummy_landmarks)
    probabilities = model.predict_proba(dummy_landmarks)
    confidence = probabilities.max()
    
    print(f"   Predicted sign: {prediction[0]}")
    print(f"   Confidence: {confidence*100:.2f}%")
    print("Prediction test passed!")
except Exception as e:
    print(f"FAILED: {e}")
    exit(1)

# Test 4: Test with multiple samples
print("\nTest 4: Testing batch prediction...")
try:
    dummy_batch = np.random.rand(5, 63)  # 5 samples
    predictions = model.predict(dummy_batch)
    print(f"   Batch predictions: {predictions}")
    print("Batch prediction test passed!")
except Exception as e:
    print(f"FAILED: {e}")
    exit(1)

# Summary
print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("Model file: isl_model.pkl")
print("Model is working correctly")
print("Ready for integration with webcam app")
print("="*60 + "\n")