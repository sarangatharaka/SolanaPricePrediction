import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import yfinance as yf
import warnings
import datetime
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Solana Price Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .big-font {
        font-size:48px !important;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Configuration
SYMBOL = "SOL-USD"
TRAIN_SIZE = 0.8
SEQUENCE_LENGTH = 60
LSTM_UNITS = 50
EPOCHS = 50
BATCH_SIZE = 32

# Sidebar
st.sidebar.title("⚙️ Configuration")

# Date range selector
st.sidebar.subheader("📅 Select Data Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.date(2020, 1, 1),
        max_value=datetime.date.today()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.date.today(),
        max_value=datetime.date.today()
    )

# Model parameters
st.sidebar.subheader("🧠 Model Parameters")
lstm_units = st.sidebar.slider("LSTM Units", 20, 100, LSTM_UNITS, 10)
epochs = st.sidebar.slider("Training Epochs", 10, 100, EPOCHS, 10)
batch_size = st.sidebar.slider("Batch Size", 16, 64, BATCH_SIZE, 4)

# Main title
st.title("💹 Solana Price Prediction")
st.markdown("*Using LSTM Deep Learning Model*")

# Cache the data download
@st.cache_data
def download_data(symbol, start, end):
    # Add 1 day to end date to ensure we get the latest available data
    end_extended = end + datetime.timedelta(days=1)
    data = yf.download(symbol, start=start, end=end_extended, progress=False)
    return data

# Get real-time current price (not cached)
def get_realtime_price(symbol):
    """Fetch the current real-time price"""
    try:
        ticker = yf.Ticker(symbol)
        # Try to get the current price from ticker info
        info = ticker.info
        if 'currentPrice' in info and info['currentPrice']:
            return info['currentPrice']
        elif 'regularMarketPrice' in info and info['regularMarketPrice']:
            return info['regularMarketPrice']
        # Fallback: get the latest price from history
        hist = ticker.history(period='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        pass
    return None

# Cache the model training
@st.cache_resource
def train_model(prices, lstm_units_param, epochs_param, batch_size_param):
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
    
    # Create sequences
    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(scaled_prices, SEQUENCE_LENGTH)
    
    # Split data
    train_size = int(len(X) * TRAIN_SIZE)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build model
    model = Sequential([
        LSTM(lstm_units_param, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
        LSTM(lstm_units_param, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs_param,
        batch_size=batch_size_param,
        validation_split=0.1,
        verbose=0
    )
    
    return model, scaler, history, X_test, y_test

# Load data
with st.spinner("📥 Downloading data..."):
    try:
        data = download_data(SYMBOL, start_date, end_date)
        prices = data['Close'].values
        
        if len(prices) < SEQUENCE_LENGTH + 1:
            st.error(f"❌ Not enough data. Please select a larger date range (minimum {SEQUENCE_LENGTH + 1} days).")
            st.stop()
        
        st.success(f"✅ Data loaded: {len(prices)} days of SOL-USD prices")
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

# Display data statistics
col1, col2, col3, col4 = st.columns(4)
with col1:
    # Get real-time current price
    realtime_price = get_realtime_price(SYMBOL)
    if realtime_price:
        st.metric("Current Price (Live)", f"${realtime_price:.2f}")
        st.caption(f"Historical data as of: {data.index[-1].date()}")
    else:
        st.metric("Current Price", f"${float(prices[-1]):.2f}")
        st.caption(f"As of: {data.index[-1].date()}")
with col2:
    st.metric("High (Period)", f"${float(prices.max()):.2f}")
with col3:
    st.metric("Low (Period)", f"${float(prices.min()):.2f}")
with col4:
    st.metric("Data Points", len(prices))

# Display historical price chart
st.subheader("📈 Historical Price Chart")
df_display = pd.DataFrame({
    'Date': data.index,
    'Price': prices.flatten() if prices.ndim > 1 else prices
})
st.line_chart(data=df_display.set_index('Date')['Price'], width='stretch')

# Retrain button
st.sidebar.markdown("---")
col_retrain = st.sidebar.columns([1, 1])
retrain_button = col_retrain[0].button("🔄 Retrain Model", use_container_width=True)

if retrain_button:
    # Clear cache to force retraining
    st.cache_resource.clear()
    st.rerun()

# Train or load model
with st.spinner("🧠 Training model..."):
    try:
        model, scaler, history, X_test, y_test = train_model(prices, lstm_units, epochs, batch_size)
        st.success("✅ Model trained successfully!")
    except Exception as e:
        st.error(f"Error training model: {e}")
        st.stop()

# Make predictions
scaled_prices = scaler.transform(prices.reshape(-1, 1))

# Create sequences for predictions
def create_sequences_for_pred(data, sequence_length):
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
    return np.array(X)

# Split point
train_size_int = int(len(scaled_prices) * TRAIN_SIZE)

# Create prediction sequences
if train_size_int >= SEQUENCE_LENGTH:
    X_all = create_sequences_for_pred(scaled_prices, SEQUENCE_LENGTH)
    X_train_pred = X_all[:train_size_int - SEQUENCE_LENGTH]
    X_test_pred = X_all[train_size_int - SEQUENCE_LENGTH:]
    
    train_predictions = model.predict(X_train_pred, verbose=0)
    test_predictions = model.predict(X_test_pred, verbose=0)
    
    # Inverse transform
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
else:
    train_predictions = np.array([])
    test_predictions = np.array([])

# Predict next day
last_sequence = scaled_prices[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
next_price_scaled = model.predict(last_sequence, verbose=0)
next_price = scaler.inverse_transform(next_price_scaled)[0][0]

# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
if len(test_predictions) > 0 and len(y_test) > 0:
    y_test_actual = scaler.inverse_transform(y_test)
    # Ensure same length by taking minimum
    min_len = min(len(y_test_actual), len(test_predictions))
    y_test_actual_trimmed = y_test_actual[:min_len]
    test_predictions_trimmed = test_predictions[:min_len]
    
    mse = mean_squared_error(y_test_actual_trimmed, test_predictions_trimmed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual_trimmed, test_predictions_trimmed)
    r2 = r2_score(y_test_actual_trimmed, test_predictions_trimmed)
else:
    mse = rmse = mae = r2 = 0.0
    y_test_actual = np.array([])
    test_predictions_trimmed = np.array([])

# Display prediction
st.markdown("---")
st.subheader("🎯 Tomorrow's Prediction")

# Big font prediction
col_pred = st.columns([1, 1, 1])
with col_pred[1]:
    st.markdown(f'<div class="big-font">${float(next_price):.2f}</div>', unsafe_allow_html=True)

# Price change
realtime_price_for_change = get_realtime_price(SYMBOL)
current_price_for_change = realtime_price_for_change if realtime_price_for_change else float(prices[-1])
price_change = ((next_price - current_price_for_change) / current_price_for_change) * 100
col_change1, col_change2, col_change3 = st.columns(3)
with col_change1:
    st.metric("Predicted Price", f"${float(next_price):.2f}")
with col_change2:
    st.metric("Current Price", f"${float(prices[-1]):.2f}")
with col_change3:
    if price_change > 0:
        st.metric("Expected Change", f"{price_change:+.2f}%", delta=f"{price_change:.2f}%")
    else:
        st.metric("Expected Change", f"{price_change:+.2f}%", delta=f"{price_change:.2f}%")

# Model Performance
st.markdown("---")
st.subheader("📊 Model Performance")

col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
with col_metric1:
    st.metric("MSE", f"${float(mse):.4f}")
with col_metric2:
    st.metric("RMSE", f"${float(rmse):.4f}")
with col_metric3:
    st.metric("MAE", f"${float(mae):.4f}")
with col_metric4:
    st.metric("R² Score", f"{float(r2):.4f}")

# Training history chart
st.subheader("📉 Training History")
col_loss1, col_loss2 = st.columns(2)

with col_loss1:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_title('Model Loss Over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Predictions vs Actual
with col_loss2:
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(test_predictions) > 0 and len(y_test) > 0:
        # Use the trimmed versions to ensure matching lengths
        min_len = min(len(y_test_actual), len(test_predictions))
        y_test_plot = y_test_actual[:min_len]
        test_pred_plot = test_predictions[:min_len]
        
        ax.plot(y_test_plot, label='Actual', linewidth=2, marker='o', markersize=4)
        ax.plot(test_pred_plot, label='Predicted', linewidth=2, marker='s', markersize=4, alpha=0.8)
        ax.set_title('Test Set: Actual vs Predicted')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p>⚠️ <strong>Disclaimer:</strong> This is a predictive model for educational purposes only. 
    Cryptocurrency prices are highly volatile and unpredictable. 
    Do not use this for actual trading decisions.</p>
    <p>Built with Streamlit, TensorFlow/Keras, and yfinance</p>
    </div>
    """, unsafe_allow_html=True)
