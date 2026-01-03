import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import yfinance as yf
import requests
import warnings
import datetime
import os
import hashlib
import pickle
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
warnings.filterwarnings('ignore')

# Fix random seeds for reproducibility
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)
# Force deterministic operations
tf.config.experimental.enable_op_determinism()

# Theme palette
PRIMARY_BG = "#0b1220"
CARD_BG = "#111827"
SURFACE_BG = "#0f172a"
TEXT_COLOR = "#e5e7eb"
ACCENT = "#38bdf8"
GRID_COLOR = "rgba(148, 163, 184, 0.35)"

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
    html, body, [class*="css"], .stApp {
        background-color: %(PRIMARY_BG)s !important;
        color: %(TEXT_COLOR)s !important;
    }
    .big-font {
        font-size:48px !important;
        font-weight: bold;
        color: %(ACCENT)s;
        text-align: center;
    }
    .metric-card, .stMetric, .stDataFrame, .stPlotlyChart, .stAltairChart, .element-container {
        background-color: %(CARD_BG)s !important;
        color: %(TEXT_COLOR)s !important;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: %(SURFACE_BG)s;
        color: %(TEXT_COLOR)s;
        border-radius: 8px;
        border: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"]:focus {
        outline: 2px solid %(ACCENT)s;
    }
    div[data-testid="stMetricValue"] {
        color: %(TEXT_COLOR)s;
    }
    /* Buttons on light backgrounds should have dark text */
    .stButton>button {
        background: #e5e7eb;
        color: #0f172a;
        font-weight: 600;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
    }
    .stButton>button:hover {
        border-color: %(ACCENT)s;
        box-shadow: 0 0 0 1px %(ACCENT)s inset;
    }
    /* Inputs */
    .stSelectbox, .stMultiSelect, .stSlider, .stNumberInput, .stDateInput, .stTextInput {
        color: %(TEXT_COLOR)s !important;
    }
    .stSlider [role='slider'] {
        background: %(ACCENT)s;
    }
    .css-1dp5vir, .css-1vbkxwb {
        background-color: %(SURFACE_BG)s !important;
    }
    .js-plotly-plot, .plot-container, .plotly {
        background-color: %(SURFACE_BG)s !important;
    }
    </style>
    """ % {
        "PRIMARY_BG": PRIMARY_BG,
        "TEXT_COLOR": TEXT_COLOR,
        "ACCENT": ACCENT,
        "SURFACE_BG": SURFACE_BG,
        "CARD_BG": CARD_BG
    }, unsafe_allow_html=True)

# Configuration
SYMBOL = "SOL-USD"
TRAIN_SIZE = 0.8
SEQUENCE_LENGTH = 60
BINANCE_SYMBOL = "SOLUSDT"
BINANCE_INTERVAL = "4h"
DEFAULT_TP_PCT = 0.02  # 2%
DEFAULT_SL_PCT = 0.01  # 1%
MODEL_CACHE_DIR = ".model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://www.theblock.co/rss"
]

# Model architectures
def create_lstm_model(input_shape, units=50):
    """Standard LSTM model"""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def create_gru_model(input_shape, units=50):
    """GRU model"""
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(units, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def create_bidirectional_lstm_model(input_shape, units=50):
    """Bidirectional LSTM model"""
    model = Sequential([
        Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(units, return_sequences=False)),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def create_transformer_model(input_shape, units=50):
    """Simplified Transformer model"""
    from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D
    from tensorflow.keras.models import Model
    
    inputs = Input(shape=input_shape)
    
    # Multi-head attention
    attention_output = MultiHeadAttention(num_heads=4, key_dim=units//4)(inputs, inputs)
    attention_output = Dropout(0.2)(attention_output)
    attention_output = LayerNormalization()(inputs + attention_output)
    
    # Feed forward
    x = Dense(units, activation='relu')(attention_output)
    x = Dropout(0.2)(x)
    x = Dense(input_shape[-1])(x)
    x = LayerNormalization()(attention_output + x)
    
    # Global pooling and output
    x = GlobalAveragePooling1D()(x)
    x = Dense(25, activation='relu')(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

MODEL_FACTORY = {
    'LSTM': create_lstm_model,
    'GRU': create_gru_model,
    'Bidirectional LSTM': create_bidirectional_lstm_model,
    'Transformer': create_transformer_model
}


def apply_dark_theme(fig, showlegend=True):
    """Apply consistent dark theming to Plotly figures."""
    fig.update_layout(
        paper_bgcolor=PRIMARY_BG,
        plot_bgcolor=SURFACE_BG,
        font=dict(color=TEXT_COLOR, size=12),
        hovermode='x unified',
        legend=dict(bgcolor='rgba(0,0,0,0.7)', font=dict(color=TEXT_COLOR, size=11), x=0.01, y=0.99),
        margin=dict(l=60, r=30, t=60, b=50),
        showlegend=showlegend,
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, color=TEXT_COLOR),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, color=TEXT_COLOR)
    )
    return fig

# --- News & sentiment utilities ---
def simple_sentiment_tag(text: str) -> str:
    """Lightweight rule-based sentiment for headlines."""
    t = text.lower()
    positive_keywords = ["surge", "rally", "gain", "bull", "up", "growth", "record", "partnership", "upgrade", "approve"]
    negative_keywords = ["fall", "drop", "bear", "down", "hack", "exploit", "outage", "lawsuit", "ban", "crash", "decline"]
    pos_hits = sum(k in t for k in positive_keywords)
    neg_hits = sum(k in t for k in negative_keywords)
    if pos_hits > neg_hits:
        return "Positive"
    if neg_hits > pos_hits:
        return "Negative"
    return "Neutral"


def parse_rss_items(xml_bytes: bytes, source: str):
    """Parse RSS XML bytes into a list of items."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    channel = root.find('channel')
    if channel is None:
        return []
    items = []
    for item in channel.findall('item'):
        title_el = item.find('title')
        link_el = item.find('link')
        pub_el = item.find('pubDate')
        title = title_el.text.strip() if title_el is not None and title_el.text else None
        link = link_el.text.strip() if link_el is not None and link_el.text else None
        pub_date = pub_el.text.strip() if pub_el is not None and pub_el.text else None
        if not title or not link:
            continue
        try:
            published_dt = parsedate_to_datetime(pub_date) if pub_date else None
        except Exception:
            published_dt = None
        items.append({
            "title": title,
            "link": link,
            "published": published_dt,
            "source": source,
            "sentiment": simple_sentiment_tag(title)
        })
    return items


@st.cache_data(ttl=600)
def fetch_latest_news(feeds=None, per_feed_limit: int = 5, total_limit: int = 15):
    """Fetch latest headlines from RSS feeds with simple sentiment tags."""
    feeds = feeds or NEWS_FEEDS
    collected = []
    for feed_url in feeds:
        try:
            resp = requests.get(feed_url, timeout=5)
            resp.raise_for_status()
            source_name = feed_url.split('//')[-1].split('/')[0]
            items = parse_rss_items(resp.content, source_name)
            collected.extend(items[:per_feed_limit])
        except Exception:
            continue
    collected.sort(key=lambda x: x["published"] or datetime.datetime.min, reverse=True)
    return collected[:total_limit]

# Binance data utilities
@st.cache_data(ttl=300)
def fetch_binance_klines(symbol: str, interval: str = BINANCE_INTERVAL, limit: int = 300):
    """Fetch klines from Binance; ttl keeps requests light."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    klines = resp.json()
    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(klines, columns=cols)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df.set_index('close_time', inplace=True)
    df.rename(columns={'open_time': 'start_time'}, inplace=True)
    return df

def compute_atr(df: pd.DataFrame, period: int = 14):
    """Compute a lightweight ATR-like volatility measure."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def generate_trading_signal(df: pd.DataFrame, tp_pct: float = DEFAULT_TP_PCT, sl_pct: float = DEFAULT_SL_PCT):
    """Rule-based signal from recent 4h candles."""
    if len(df) < 30:
        return None
    close = df['close']
    rsi = RSIIndicator(close=close, window=14).rsi()
    ma_fast = close.rolling(9).mean()
    ma_slow = close.rolling(21).mean()
    atr = compute_atr(df, period=14)
    latest_price = float(close.iloc[-1])
    latest_rsi = float(rsi.iloc[-1])
    momentum = ma_fast.iloc[-1] - ma_slow.iloc[-1]
    trend_bias = momentum / latest_price if latest_price else 0
    volatility = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else latest_price * 0.01
    action = "Hold"
    if momentum > 0 and latest_rsi < 70:
        action = "Buy"
    elif momentum < 0 and latest_rsi > 30:
        action = "Sell"
    tp = latest_price * (1 + tp_pct) if action == "Buy" else latest_price * (1 - tp_pct)
    sl = latest_price * (1 - sl_pct) if action == "Buy" else latest_price * (1 + sl_pct)
    rr = abs((tp - latest_price) / (sl - latest_price)) if sl != latest_price else np.nan
    confidence = min(max(abs(trend_bias) / (volatility / latest_price + 1e-6), 0), 5)
    return {
        "action": action,
        "start_price": latest_price,
        "take_profit": tp,
        "stop_loss": sl,
        "liquidate_price": sl,
        "rsi": latest_rsi,
        "trend_bias": trend_bias,
        "volatility": volatility,
        "risk_reward": rr,
        "confidence": confidence,
        "timeframe": BINANCE_INTERVAL,
        "generated_at": df.index[-1]
    }

# Feature engineering functions
def add_technical_indicators(df):
    """Add technical indicators to dataframe"""
    df = df.copy()
    
    # Ensure data is pandas Series (1D) by squeezing if needed
    close_series = df['Close'].squeeze() if hasattr(df['Close'], 'squeeze') else pd.Series(df['Close'].values.flatten(), index=df.index)
    volume_series = df['Volume'].squeeze() if hasattr(df['Volume'], 'squeeze') else pd.Series(df['Volume'].values.flatten(), index=df.index)
    
    # RSI
    rsi = RSIIndicator(close=close_series, window=14)
    df['RSI'] = rsi.rsi()
    
    # MACD
    macd = MACD(close=close_series)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = BollingerBands(close=close_series)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    df['BB_Width'] = df['BB_High'] - df['BB_Low']
    
    # Volume
    df['Volume_MA'] = volume_series.rolling(window=20).mean()
    
    # Price changes
    df['Price_Change'] = close_series.pct_change()
    df['Price_MA_7'] = close_series.rolling(window=7).mean()
    df['Price_MA_30'] = close_series.rolling(window=30).mean()
    
    # Fill NaN values with forward and backward fill
    df = df.ffill().bfill()
    
    return df

# Sidebar
st.sidebar.title("⚙️Configuration")

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

# Model selection
st.sidebar.subheader("🧠 Model Selection")
selected_models = st.sidebar.multiselect(
    "Select Models to Train",
    list(MODEL_FACTORY.keys()),
    default=['LSTM', 'GRU']
)

use_ensemble = st.sidebar.checkbox("Use Ensemble (Average all models)", value=True)

# Feature engineering
st.sidebar.subheader("🔧 Feature Engineering")
use_technical_indicators = st.sidebar.checkbox("Use Technical Indicators", value=True)
use_volume = st.sidebar.checkbox("Include Volume Data", value=True)

# Forecasting
st.sidebar.subheader("📊 Forecasting Options")
forecast_days = st.sidebar.select_slider(
    "Forecast Horizon",
    options=[1, 7, 30],
    value=7
)

# Trading signals
st.sidebar.subheader("⚡ Trading Signals")
tp_pct = st.sidebar.slider("Take Profit %", 0.5, 10.0, DEFAULT_TP_PCT * 100, 0.5) / 100
sl_pct = st.sidebar.slider("Stop Loss %", 0.2, 10.0, DEFAULT_SL_PCT * 100, 0.2) / 100

# Model parameters
st.sidebar.subheader("🎛️ Model Parameters")
lstm_units = st.sidebar.slider("LSTM/GRU Units", 20, 100, 50, 10)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50, 10)
batch_size = st.sidebar.slider("Batch Size", 16, 64, 32, 4)

st.sidebar.info("💡 **Models are cached** - Same data/params = instant load. Change params to retrain.")

# Clear cache button
if st.sidebar.button("🗑️ Clear Model Cache"):
    import shutil
    if os.path.exists(MODEL_CACHE_DIR):
        shutil.rmtree(MODEL_CACHE_DIR)
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    st.cache_resource.clear()
    st.sidebar.success("Cache cleared! Refresh to retrain.")

# Main title
st.title("💹Solana Price Prediction")
st.markdown("*Advanced LSTM/GRU/Transformer Models with Technical Indicators*")

# Download data without caching (simpler, more reliable for Streamlit deployment)
def download_data(symbol, start, end):
    """Download price history with multiple fallbacks (Yahoo then Binance)."""
    min_days = SEQUENCE_LENGTH + 1
    
    def _trim_to_range(df: pd.DataFrame):
        if df.empty:
            return df
        idx = df.index.tz_localize(None) if getattr(df.index, "tz", None) else df.index
        return df.loc[(idx.date >= start) & (idx.date <= end)]

    def _pull_yf_with_retry(symbol_value, start_value=None, end_value=None, period=None, max_retries=3):
        """Pull from Yahoo Finance with retry logic."""
        for attempt in range(max_retries):
            try:
                if period:
                    df = yf.download(symbol_value, period=period, progress=False, timeout=15)
                else:
                    df = yf.download(symbol_value, start=start_value, end=end_value, progress=False, timeout=15)
                if df is not None and not df.empty:
                    return df.sort_index()
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1 + attempt)  # Progressive backoff: 1s, 2s, 3s
                continue
        return pd.DataFrame()

    def _pull_binance(symbol_value="SOLUSDT", interval="1d", limit=1200):
        """Pull from Binance API."""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": symbol_value, "interval": interval, "limit": limit}
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            klines = resp.json()
            if not klines:
                return pd.DataFrame()
            cols = [
                "open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"
            ]
            df = pd.DataFrame(klines, columns=cols)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.set_index("close_time").sort_index()
            df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }, inplace=True)
            df["Adj Close"] = df["Close"]
            return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        except Exception:
            return pd.DataFrame()

    end_extended = end + datetime.timedelta(days=1)

    # 1) Try user-selected window first
    data = _pull_yf_with_retry(symbol, start, end_extended)
    if len(data) >= min_days:
        return _trim_to_range(data)

    # 2) Widen the window automatically on Yahoo
    widened_start = max(start - datetime.timedelta(days=max(min_days * 2, 120)), datetime.date(2018, 1, 1))
    data_wide = _pull_yf_with_retry(symbol, widened_start, end_extended)
    if len(data_wide) >= min_days:
        return _trim_to_range(data_wide)

    # 3) Max history on Yahoo
    data_max = _pull_yf_with_retry(symbol, period="max")
    if len(data_max) >= min_days:
        return _trim_to_range(data_max)

    # 4) Binance daily candles (SOLUSDT)
    data_binance = _pull_binance(symbol_value="SOLUSDT", interval="1d", limit=1200)
    if len(data_binance) >= min_days:
        return _trim_to_range(data_binance)

    # 5) Binance 4h candles aggregated to daily
    data_binance_4h = _pull_binance(symbol_value="SOLUSDT", interval="4h", limit=1500)
    if not data_binance_4h.empty:
        daily = data_binance_4h.resample("1D").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Adj Close": "last",
            "Volume": "sum"
        }).dropna()
        if len(daily) >= min_days:
            return _trim_to_range(daily)

    raise ValueError(f"Insufficient data for {symbol} (0 rows across all sources). Please try again or expand your date range.")

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

# Cache feature engineering (use resource cache for this since it's less problematic)
@st.cache_resource(show_spinner=False)
def engineer_features(data, use_tech_indicators, use_vol):
    if use_tech_indicators:
        data = add_technical_indicators(data)
    
    feature_columns = ['Close']
    if use_tech_indicators:
        feature_columns.extend(['RSI', 'MACD', 'MACD_Signal', 'BB_Width', 'Price_Change', 'Price_MA_7'])
    if use_vol:
        feature_columns.append('Volume')
    
    return data, feature_columns

# Train model function
def train_single_model(model_name, X_train, y_train, input_shape, units, epochs_param, batch_size_param):
    """Train a single model with early stopping and checkpointing"""
    model_func = MODEL_FACTORY[model_name]
    model = model_func(input_shape, units)
    
    # Create callbacks for stability
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    
    checkpoint_path = os.path.join(MODEL_CACHE_DIR, f"temp_{model_name}.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs_param,
        batch_size=batch_size_param,
        validation_split=0.2,  # Increased from 0.1 to 0.2 for better stability
        verbose=0,
        callbacks=[early_stop, checkpoint]
    )
    
    # Load best weights
    if os.path.exists(checkpoint_path):
        model = load_model(checkpoint_path)
    
    return model, history

@st.cache_resource
def get_cached_model(model_name, _X_train, _y_train, input_shape, units, epochs_param, batch_size_param, data_hash):
    """Cache trained models to avoid retraining on every run"""
    model_path = os.path.join(MODEL_CACHE_DIR, f"{model_name}_{data_hash}.h5")
    history_path = os.path.join(MODEL_CACHE_DIR, f"{model_name}_{data_hash}_history.pkl")
    
    if os.path.exists(model_path) and os.path.exists(history_path):
        # Load cached model
        model = load_model(model_path)
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        return model, history
    
    # Train new model
    model, history = train_single_model(model_name, _X_train, _y_train, input_shape, units, epochs_param, batch_size_param)
    
    # Save to cache
    model.save(model_path)
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    return model, history

def compute_data_hash(X_train, y_train, model_params):
    """Generate hash for cache invalidation"""
    data_str = f"{X_train.shape}_{y_train.shape}_{model_params}"
    return hashlib.md5(data_str.encode()).hexdigest()[:8]

# Multi-day forecasting
def forecast_multiple_days(model, last_sequence, scaler, num_days, num_features):
    """Forecast multiple days ahead"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(num_days):
        # Predict next day
        next_pred = model.predict(current_sequence, verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Update sequence (slide window)
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred[0, 0]
    
    # Inverse transform - pad with zeros for other features
    predictions = np.array(predictions).reshape(-1, 1)
    # Create array with correct shape for scaler (predictions + zeros for other features)
    predictions_padded = np.concatenate([predictions, np.zeros((predictions.shape[0], num_features - 1))], axis=1)
    predictions_rescaled = scaler.inverse_transform(predictions_padded)[:, 0]
    
    return predictions_rescaled

# Load data
with st.spinner("📥 Downloading data..."):
    try:
        data = download_data(SYMBOL, start_date, end_date)
        
        if len(data) < SEQUENCE_LENGTH + 1:
            st.error(f"❌ Not enough data. Please select a larger date range (minimum {SEQUENCE_LENGTH + 1} days).")
            st.stop()
        
        st.success(f"✅ Data loaded: {len(data)} days of SOL-USD prices")
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

# Feature engineering
data_features, feature_columns = engineer_features(data, use_technical_indicators, use_volume)

# Fetch news once for all tabs
news_items = fetch_latest_news()

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🔮 Predictions", "📊 Model Comparison", "📉 Technical Analysis", "⚡ Trading Signals"])

with tab1:
    st.subheader("📊 Data Overview")
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Get real-time current price
        realtime_price = get_realtime_price(SYMBOL)
        if realtime_price:
            st.metric("Current Price (Live)", f"${realtime_price:.2f}")
            # Show data as of date below
            st.caption(f"Historical data as of: {data.index[-1].date()}")
        else:
            # Fallback to latest historical price
            st.metric("Current Price", f"${float(data['Close'].iloc[-1]):.2f}")
            st.caption(f"As of: {data.index[-1].date()}")
    with col2:
        st.metric("High (Period)", f"${float(data['Close'].max()):.2f}")
    with col3:
        st.metric("Low (Period)", f"${float(data['Close'].min()):.2f}")
    with col4:
        st.metric("Data Points", len(data))
    
    # Interactive price chart with Plotly
    st.subheader("📈 Interactive Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index.astype(str),
        y=data['Close'].values.flatten(),
        mode='lines',
        name='Close Price',
        line=dict(color='#00ff00', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.15)',
        opacity=1.0
    ))
    
    fig.update_layout(
        title='SOL-USD Historical Price',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        showlegend=True
    )
    apply_dark_theme(fig)
    
    st.plotly_chart(fig, use_container_width=True, theme=None)
    
    # Show latest data
    st.subheader("📋 Latest Data")
    st.dataframe(data.tail(10), use_container_width=True)

    st.subheader("📰 Latest Crypto Headlines (10 min cache)")
    if news_items:
        # Create columns for responsive layout
        cols_per_row = 1
        for i, item in enumerate(news_items):
            published_text = item["published"].strftime("%b %d, %H:%M") if item["published"] else "Unknown time"
            
            # Sentiment color mapping
            sentiment_color_map = {
                "Positive": "🟢",
                "Negative": "🔴",
                "Neutral": "⚪"
            }
            sentiment_icon = sentiment_color_map.get(item['sentiment'], "⚪")
            
            # Create styled container for each headline
            with st.container():
                col1, col2 = st.columns([0.08, 0.92])
                with col1:
                    st.markdown(f"<div style='font-size: 20px; text-align: center;'>{sentiment_icon}</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div style='background-color: #111827; padding: 15px; border-radius: 8px; border-left: 3px solid #38bdf8; margin-bottom: 10px;'>
                        <a href='{item['link']}' target='_blank' style='color: #38bdf8; text-decoration: none; font-weight: 600; font-size: 14px;'>
                            {item['title']}
                        </a>
                        <div style='color: #9ca3af; font-size: 12px; margin-top: 8px;'>
                            <span style='color: #60a5fa;'>{item['source']}</span> • <span style='color: #a78bfa;'>{published_text}</span> • <span style='color: #34d399; font-weight: 500;'>{item['sentiment']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Add divider between items except for the last one
            if i < len(news_items) - 1:
                st.divider()
    else:
        st.info("📡 News feeds unavailable right now. They refresh every 10 minutes when reachable.")

with tab4:
    st.subheader("📉 Technical Analysis")
    
    if use_technical_indicators:
        # RSI Chart
        fig_tech = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price with Bollinger Bands', 'RSI', 'MACD'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Bollinger Bands
        fig_tech.add_trace(go.Scatter(x=data_features.index.astype(str), y=data_features['Close'].values.flatten(), name='Close', line=dict(color='#00ff00', width=3), opacity=1.0), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=data_features.index.astype(str), y=data_features['BB_High'].values.flatten(), name='BB High', line=dict(color='#ff0000', width=1, dash='dash'), opacity=1.0), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=data_features.index.astype(str), y=data_features['BB_Low'].values.flatten(), name='BB Low', line=dict(color='#0099ff', width=1, dash='dash'), opacity=1.0), row=1, col=1)
        
        # RSI
        fig_tech.add_trace(go.Scatter(x=data_features.index.astype(str), y=data_features['RSI'].values.flatten(), name='RSI', line=dict(color='#ffff00', width=3), opacity=1.0), row=2, col=1)
        fig_tech.add_hline(y=70, line_dash="dash", line_color="#ff0000", row=2, col=1)
        fig_tech.add_hline(y=30, line_dash="dash", line_color="#0099ff", row=2, col=1)
        
        # MACD
        fig_tech.add_trace(go.Scatter(x=data_features.index.astype(str), y=data_features['MACD'].values.flatten(), name='MACD', line=dict(color='#00ff00', width=3), opacity=1.0), row=3, col=1)
        fig_tech.add_trace(go.Scatter(x=data_features.index.astype(str), y=data_features['MACD_Signal'].values.flatten(), name='Signal', line=dict(color='#ff00ff', width=2), opacity=1.0), row=3, col=1)
        
        fig_tech.update_layout(height=900, showlegend=True)
        apply_dark_theme(fig_tech)
        st.plotly_chart(fig_tech, use_container_width=True, theme=None)
        
        # Technical indicators summary
        col1, col2, col3 = st.columns(3)
        with col1:
            current_rsi = float(data_features['RSI'].iloc[-1])
            st.metric("Current RSI", f"{current_rsi:.2f}", 
                     "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral")
        with col2:
            current_macd = float(data_features['MACD'].iloc[-1])
            st.metric("MACD", f"{current_macd:.4f}")
        with col3:
            bb_position = (data_features['Close'].iloc[-1] - data_features['BB_Low'].iloc[-1]) / (data_features['BB_High'].iloc[-1] - data_features['BB_Low'].iloc[-1])
            st.metric("BB Position", f"{float(bb_position)*100:.1f}%")
    else:
        st.info("Enable 'Use Technical Indicators' in the sidebar to see technical analysis.")

with tab5:
    st.subheader("⚡ Real-Time Trading Signals (4h)")
    try:
        with st.spinner("Fetching 4h data from Binance..."):
            binance_df = fetch_binance_klines(BINANCE_SYMBOL, BINANCE_INTERVAL)
        if binance_df.empty:
            st.warning("No Binance data returned.")
        else:
            latest_row = binance_df.iloc[-1]
            latest_price = float(latest_row['close'])
            st.metric("Live Price", f"${latest_price:.2f}", help="From Binance 4h close")
            signal = generate_trading_signal(binance_df, tp_pct=tp_pct, sl_pct=sl_pct)
            if signal:
                cols = st.columns(3)
                cols[0].metric("Action", signal['action'])
                cols[1].metric("Start Price", f"${signal['start_price']:.2f}")
                cols[2].metric("Confidence", f"{signal['confidence']:.2f}")

                cols2 = st.columns(4)
                cols2[0].metric("Take Profit", f"${signal['take_profit']:.2f}")
                cols2[1].metric("Stop Loss", f"${signal['stop_loss']:.2f}")
                rr_val = signal['risk_reward']
                rr_text = f"{rr_val:.2f}" if rr_val is not None and np.isfinite(rr_val) else "-"
                cols2[2].metric("Risk/Reward", rr_text)
                cols2[3].metric("RSI", f"{signal['rsi']:.1f}")

                st.caption(f"Timeframe: {signal['timeframe']} | Generated at: {signal['generated_at']}")

                # Chart with TP/SL overlays (last 60 candles)
                window = min(60, len(binance_df))
                df_plot = binance_df.tail(window)
                fig_sig = go.Figure()
                fig_sig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'], mode='lines', name='Close'))
                fig_sig.add_hline(y=signal['take_profit'], line_dash="dash", line_color="green", annotation_text="TP")
                fig_sig.add_hline(y=signal['stop_loss'], line_dash="dash", line_color="red", annotation_text="SL")
                fig_sig.update_layout(title="4h Price with TP/SL", xaxis_title="Time", yaxis_title="Price ($)", height=400)
                apply_dark_theme(fig_sig)
                st.plotly_chart(fig_sig, use_container_width=True, theme=None)

                # Recent candles table
                display_cols = ['start_time', 'open', 'high', 'low', 'close', 'volume']
                st.dataframe(binance_df.tail(15)[display_cols].rename(columns=str.capitalize), use_container_width=True)
            else:
                st.info("Not enough data to generate a signal yet.")
    except Exception as e:
        st.error(f"Binance fetch error: {e}")

# Prepare data for training
with tab2:
    st.subheader("🔮 Model Predictions")
    
    # Prepare features
    features = data_features[feature_columns].values
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences
    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length, 0])  # Predict Close price
        return np.array(X), np.array(y)
    
    X, y = create_sequences(scaled_features, SEQUENCE_LENGTH)
    
    # Split data
    train_size = int(len(X) * TRAIN_SIZE)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    input_shape = (SEQUENCE_LENGTH, scaled_features.shape[1])
    
    # Train models
    if not selected_models:
        st.warning("⚠️ Please select at least one model from the sidebar.")
        st.stop()
    
    # Compute hash for caching
    model_params = f"{lstm_units}_{epochs}_{batch_size}_{use_technical_indicators}_{use_volume}"
    data_hash = compute_data_hash(X_train, y_train, model_params)
    
    models = {}
    histories = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, model_name in enumerate(selected_models):
        status_text.text(f"Loading/Training {model_name} model... ({idx+1}/{len(selected_models)})")
        
        with st.spinner(f"Processing {model_name}..."):
            model, history = get_cached_model(
                model_name, X_train, y_train, input_shape,
                lstm_units, epochs, batch_size, data_hash
            )
            models[model_name] = model
            histories[model_name] = history
        
        progress_bar.progress((idx + 1) / len(selected_models))
    
    status_text.text("✅ All models ready!")
    progress_bar.empty()
    
    # Make predictions for each model
    predictions_dict = {}
    multi_day_forecasts = {}
    
    for model_name, model in models.items():
        # Test predictions
        test_pred = model.predict(X_test, verbose=0)
        
        # Inverse transform (only first column for Close price)
        test_pred_rescaled = scaler.inverse_transform(
            np.concatenate([test_pred, np.zeros((test_pred.shape[0], scaled_features.shape[1]-1))], axis=1)
        )[:, 0]
        
        predictions_dict[model_name] = test_pred_rescaled
        
        # Multi-day forecast
        last_sequence = scaled_features[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, scaled_features.shape[1])
        forecast = forecast_multiple_days(model, last_sequence, scaler, forecast_days, scaled_features.shape[1])
        multi_day_forecasts[model_name] = forecast
    
    # Ensemble prediction
    if use_ensemble and len(selected_models) > 1:
        ensemble_test_pred = np.mean([predictions_dict[m] for m in selected_models], axis=0)
        predictions_dict['Ensemble'] = ensemble_test_pred
        
        ensemble_forecast = np.mean([multi_day_forecasts[m] for m in selected_models], axis=0)
        multi_day_forecasts['Ensemble'] = ensemble_forecast
    
    # Display predictions
    st.subheader(f"📅 {forecast_days}-Day Forecast")
    
    # Create forecast dates
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    
    # Display forecast table
    forecast_df = pd.DataFrame(multi_day_forecasts, index=forecast_dates)
    forecast_df.index.name = 'Date'
    
    st.dataframe(forecast_df.style.format("${:.2f}"), use_container_width=True)
    
    # Tomorrow's predictions
    st.subheader("🎯 Tomorrow's Prediction")
    cols = st.columns(len(multi_day_forecasts))
    for idx, (model_name, forecast) in enumerate(multi_day_forecasts.items()):
        with cols[idx]:
            tomorrow_price = float(forecast[0])
            # Use real-time price if available, otherwise use historical
            realtime_price = get_realtime_price(SYMBOL)
            current_price = realtime_price if realtime_price else float(data['Close'].iloc[-1])
            change = ((tomorrow_price - current_price) / current_price) * 100
            
            st.metric(
                f"{model_name}",
                f"${tomorrow_price:.2f}",
                f"{change:+.2f}%"
            )
    
    # Forecast visualization
    st.subheader("📊 Forecast Visualization")
    fig_forecast = go.Figure()
    
    # Historical prices (last 30 days)
    historical_window = min(30, len(data))
    fig_forecast.add_trace(go.Scatter(
        x=data.index[-historical_window:].astype(str),
        y=data['Close'].iloc[-historical_window:].values.flatten(),
        mode='lines',
        name='Historical',
        line=dict(color='#00ff00', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.15)',
        opacity=1.0
    ))
    
    # Forecasts
    colors = ['#ff0000', '#0099ff', '#ff00ff', '#ffff00', '#ff6600']
    for idx, (model_name, forecast) in enumerate(multi_day_forecasts.items()):
        fig_forecast.add_trace(go.Scatter(
            x=forecast_dates.astype(str),
            y=forecast,
            mode='lines+markers',
            name=f'{model_name} Forecast',
            line=dict(color=colors[idx % len(colors)], width=2, dash='dash'),
            marker=dict(size=8),
            opacity=1.0
        ))
    
    fig_forecast.update_layout(
        title=f'{forecast_days}-Day Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        showlegend=True
    )
    apply_dark_theme(fig_forecast)
    
    st.plotly_chart(fig_forecast, use_container_width=True, theme=None)

    st.subheader("🧠 Why These Predictions")
    def latest_or_none(series_name):
        return float(data_features[series_name].iloc[-1]) if series_name in data_features.columns else None

    latest_close = float(data_features['Close'].iloc[-1])
    pct_7d = float(data_features['Close'].pct_change(7).iloc[-1] * 100) if len(data_features) > 7 else None
    pct_30d = float(data_features['Close'].pct_change(30).iloc[-1] * 100) if len(data_features) > 30 else None
    rsi_val = latest_or_none('RSI')
    macd_hist = latest_or_none('MACD_Diff')
    bb_width = latest_or_none('BB_Width')
    rationale_lines = []
    if pct_7d is not None:
        direction = "up" if pct_7d > 0 else "down"
        rationale_lines.append(f"Recent 7d trend is {direction} ({pct_7d:+.2f}%), which influences short-horizon forecasts.")
    if pct_30d is not None:
        direction = "up" if pct_30d > 0 else "down"
        rationale_lines.append(f"30d drift is {direction} ({pct_30d:+.2f}%), informing medium-term bias for {forecast_days}-day outputs.")
    if rsi_val is not None:
        if rsi_val > 70:
            rationale_lines.append(f"RSI {rsi_val:.1f} signals overbought conditions, tempering upside in model predictions.")
        elif rsi_val < 30:
            rationale_lines.append(f"RSI {rsi_val:.1f} is oversold, giving room for mean-reversion in forecasts.")
        else:
            rationale_lines.append(f"RSI {rsi_val:.1f} is neutral; momentum cues are modest.")
    if macd_hist is not None:
        trend_side = "bullish" if macd_hist > 0 else "bearish"
        rationale_lines.append(f"MACD histogram is {trend_side} ({macd_hist:+.4f}), guiding the trend component the models learn.")
    if bb_width is not None:
        rationale_lines.append(f"Bollinger band width at {bb_width:.4f} reflects current volatility; wider bands allow larger forecast swings.")
    if use_volume and 'Volume' in data_features.columns:
        vol_ma = float(data_features['Volume'].rolling(20).mean().iloc[-1]) if len(data_features) >= 20 else None
        if vol_ma:
            rationale_lines.append("Volume data is included; elevated volume can reinforce breakout signals in recurrent models.")
    if use_ensemble and 'Ensemble' in multi_day_forecasts:
        rationale_lines.append("Ensemble averages smooth individual model noise, reducing variance in the displayed forecast.")
    if not rationale_lines:
        rationale_lines.append("Forecasts are driven by recent price sequences; enable technical indicators for deeper rationale.")

    for line in rationale_lines:
        st.markdown(f"- {line}")

with tab3:
    st.subheader("📊 Model Performance Comparison")
    
    # Calculate metrics for each model
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics_df = []
    y_test_actual = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_features.shape[1]-1))], axis=1)
    )[:, 0]
    
    for model_name, test_pred in predictions_dict.items():
        min_len = min(len(y_test_actual), len(test_pred))
        y_true = y_test_actual[:min_len]
        y_pred = test_pred[:min_len]
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics_df.append({
            'Model': model_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R² Score': r2
        })
    
    metrics_df = pd.DataFrame(metrics_df)
    
    # Display metrics table
    st.dataframe(
        metrics_df.style.format({
            'MSE': '${:.4f}',
            'RMSE': '${:.4f}',
            'MAE': '${:.4f}',
            'R² Score': '{:.4f}'
        }).background_gradient(subset=['R² Score'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # Visualization comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart for metrics
        fig_metrics = go.Figure()
        for metric in ['RMSE', 'MAE']:
            fig_metrics.add_trace(go.Bar(
                name=metric,
                x=metrics_df['Model'],
                y=metrics_df[metric],
                text=metrics_df[metric].round(2),
                textposition='auto'
            ))
        
        fig_metrics.update_layout(
            title='Error Metrics Comparison',
            barmode='group',
            yaxis_title='Value ($)',
            height=400
        )
        apply_dark_theme(fig_metrics)
        st.plotly_chart(fig_metrics, use_container_width=True, theme=None)
    
    with col2:
        # R² Score comparison
        fig_r2 = go.Figure(go.Bar(
            x=metrics_df['Model'],
            y=metrics_df['R² Score'],
            text=metrics_df['R² Score'].round(4),
            textposition='auto',
            marker_color='lightblue'
        ))
        
        fig_r2.update_layout(
            title='R² Score Comparison (Higher is Better)',
            yaxis_title='R² Score',
            height=400
        )
        apply_dark_theme(fig_r2)
        st.plotly_chart(fig_r2, use_container_width=True, theme=None)
    
    # Predictions comparison
    st.subheader("🔍 Predictions vs Actual")
    fig_pred = go.Figure()
    
    # Actual values
    fig_pred.add_trace(go.Scatter(
        x=list(range(len(y_test_actual))),
        y=y_test_actual,
        mode='lines',
        name='Actual',
        line=dict(color='black', width=2)
    ))
    
    # Model predictions
    for model_name, test_pred in predictions_dict.items():
        min_len = min(len(y_test_actual), len(test_pred))
        fig_pred.add_trace(go.Scatter(
            x=list(range(min_len)),
            y=test_pred[:min_len],
            mode='lines',
            name=model_name,
            line=dict(width=2),
            opacity=0.7
        ))
    
    fig_pred.update_layout(
        title='Test Set: Actual vs Predicted Prices',
        xaxis_title='Sample',
        yaxis_title='Price ($)',
        height=500
    )
    apply_dark_theme(fig_pred)
    
    st.plotly_chart(fig_pred, use_container_width=True, theme=None)
    
    # Training history
    st.subheader("📈 Training History")
    fig_history = go.Figure()
    
    for model_name, history in histories.items():
        fig_history.add_trace(go.Scatter(
            x=list(range(len(history.history['loss']))),
            y=history.history['loss'],
            mode='lines',
            name=f'{model_name} - Train Loss',
            line=dict(width=2)
        ))
        fig_history.add_trace(go.Scatter(
            x=list(range(len(history.history['val_loss']))),
            y=history.history['val_loss'],
            mode='lines',
            name=f'{model_name} - Val Loss',
            line=dict(width=2, dash='dash')
        ))
    
    fig_history.update_layout(
        title='Training Loss Comparison',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        height=500
    )
    apply_dark_theme(fig_history)
    
    st.plotly_chart(fig_history, use_container_width=True, theme=None)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p>⚠️ <strong>Disclaimer:</strong> This is an advanced predictive model for educational purposes only. 
    Cryptocurrency prices are highly volatile and unpredictable. 
    Do not use this for actual trading decisions.</p>
    <p>✨ Enhanced with Multiple Models, Technical Indicators & Advanced Forecasting</p>
    <p>Built with Streamlit, TensorFlow/Keras, TA-Lib, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)
