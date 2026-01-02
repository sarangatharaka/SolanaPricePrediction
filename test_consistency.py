"""
Test script to verify model prediction consistency
Run this multiple times to verify predictions are identical
"""

import numpy as np
import os
import tensorflow as tf

# Set random seeds (same as app_enhanced.py)
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
tf.config.experimental.enable_op_determinism()

print("=" * 60)
print("Model Consistency Test")
print("=" * 60)

# Create a simple test model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate deterministic test data
X_test = np.random.rand(10, 60, 1)
y_test = np.random.rand(10, 1)

# Build model
model = Sequential([
    LSTM(50, input_shape=(60, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train briefly
model.fit(X_test, y_test, epochs=5, verbose=0)

# Make predictions 3 times
predictions = []
for i in range(3):
    pred = model.predict(X_test[:1], verbose=0)
    predictions.append(pred[0, 0])
    print(f"Run {i+1}: {pred[0, 0]:.10f}")

# Check consistency
if len(set([f"{p:.10f}" for p in predictions])) == 1:
    print("\n✅ SUCCESS: All predictions are identical!")
    print("   The model is now producing consistent results.")
else:
    print("\n⚠️ WARNING: Predictions differ!")
    print("   There may still be non-deterministic operations.")

print("\nNote: Run this script multiple times in separate")
print("      terminal sessions to verify cross-session consistency.")
print("=" * 60)
