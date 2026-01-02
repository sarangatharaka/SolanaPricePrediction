import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Configuration
SYMBOL = "SOL-USD"
TRAIN_SIZE = 0.8
SEQUENCE_LENGTH = 60
LSTM_UNITS = 50
EPOCHS = 50
BATCH_SIZE = 32

print("Solana Price Prediction using LSTM")
print("=" * 50)

# Step 1: Download historical data
print("\n1. Downloading SOL-USD historical data...")
data = yf.download(SYMBOL, start="2020-01-01", progress=False)
print(f"Data shape: {data.shape}")
print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")

# Use closing prices
prices = data['Close'].values.reshape(-1, 1)
print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")

# Step 2: Normalize data using MinMaxScaler
print("\n2. Preprocessing data with MinMaxScaler...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)
print(f"Scaled data shape: {scaled_prices.shape}")

# Step 3: Create sequences for LSTM
print("\n3. Creating sequences for LSTM training...")
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_prices, SEQUENCE_LENGTH)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Step 4: Split data into train and test sets
print("\n4. Splitting data into train (80%) and test (20%) sets...")
train_size = int(len(X) * TRAIN_SIZE)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Step 5: Build LSTM model
print("\n5. Building LSTM model...")
model = Sequential([
    LSTM(LSTM_UNITS, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
    LSTM(LSTM_UNITS, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
print(model.summary())

# Step 6: Train the model
print("\n6. Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

# Step 7: Make predictions
print("\n7. Making predictions...")
train_predictions = model.predict(X_train, verbose=0)
test_predictions = model.predict(X_test, verbose=0)

# Inverse transform predictions to original price scale
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train_actual = scaler.inverse_transform(y_train)
y_test_actual = scaler.inverse_transform(y_test)

# Calculate Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test_actual, test_predictions)
mae = mean_absolute_error(y_test_actual, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, test_predictions)

print(f"\nModel Performance on Test Set:")
print(f"MSE: ${mse:.4f}")
print(f"RMSE: ${rmse:.4f}")
print(f"MAE: ${mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Step 8: Predict next day's price
print("\n8. Predicting next day's price...")
last_sequence = scaled_prices[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
next_price_scaled = model.predict(last_sequence, verbose=0)
next_price = scaler.inverse_transform(next_price_scaled)[0][0]
print(f"Current price: ${prices[-1][0]:.2f}")
print(f"Predicted next day's price: ${next_price:.2f}")
price_change = ((next_price - prices[-1][0]) / prices[-1][0]) * 100
print(f"Expected change: {price_change:+.2f}%")

# Step 9: Plotting
print("\n9. Creating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Training and validation loss
axes[0, 0].plot(history.history['loss'], label='Training Loss')
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
axes[0, 0].set_title('Model Loss Over Epochs')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: Full price history
axes[0, 1].plot(prices, label='Historical Price', color='blue')
axes[0, 1].set_title('SOL-USD Historical Closing Prices')
axes[0, 1].set_xlabel('Days')
axes[0, 1].set_ylabel('Price ($)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot 3: Training predictions
train_plot_prices = np.concatenate([y_train_actual, train_predictions])
axes[1, 0].plot(y_train_actual, label='Actual', linewidth=2)
axes[1, 0].plot(train_predictions, label='Predicted', linewidth=2, alpha=0.8)
axes[1, 0].set_title('Training Set: Actual vs Predicted Prices')
axes[1, 0].set_xlabel('Sample')
axes[1, 0].set_ylabel('Price ($)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot 4: Test predictions
axes[1, 1].plot(y_test_actual, label='Actual', linewidth=2, marker='o', markersize=4)
axes[1, 1].plot(test_predictions, label='Predicted', linewidth=2, marker='s', markersize=4, alpha=0.8)
axes[1, 1].set_title('Test Set: Actual vs Predicted Prices')
axes[1, 1].set_xlabel('Sample')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('solana_price_prediction.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'solana_price_prediction.png'")
plt.show()

print("\n" + "=" * 50)
print("Prediction complete!")
