from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import yfinance as yf
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

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

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Get parameters
            symbol = request_data.get('symbol', 'SOL-USD')
            days = int(request_data.get('days', 365))
            forecast_days = int(request_data.get('forecast_days', 30))
            
            # Fetch data
            data = yf.download(symbol, period=f'{days}d', progress=False)
            
            if len(data) < 100:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'Insufficient data available'
                }).encode())
                return
            
            # Prepare data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data[['Close']])
            
            # Create sequences
            sequence_length = 60
            X_train = []
            y_train = []
            
            for i in range(len(scaled_data) - sequence_length):
                X_train.append(scaled_data[i:i+sequence_length])
                y_train.append(scaled_data[i+sequence_length])
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Build model
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            
            # Make predictions
            last_sequence = scaled_data[-sequence_length:]
            predictions = []
            
            for _ in range(forecast_days):
                next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
                predictions.append(next_pred[0, 0])
                last_sequence = np.append(last_sequence[1:], next_pred)
            
            # Inverse transform
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            # Get real-time current price
            realtime_price = get_realtime_price(symbol)
            current_price = realtime_price if realtime_price else float(data['Close'].iloc[-1])
            predicted_prices = predictions.flatten().tolist()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                'success': True,
                'current_price': current_price,
                'symbol': symbol,
                'predictions': predicted_prices,
                'forecast_days': forecast_days
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'error': str(e)
            }).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
