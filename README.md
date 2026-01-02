# Enhanced Solana Price Prediction

Advanced cryptocurrency price prediction using multiple deep learning models with technical analysis.

## 🚀 New Features (Phase 1 Complete)

### 1. Multiple Model Architectures

- **LSTM** - Long Short-Term Memory networks
- **GRU** - Gated Recurrent Units (faster training)
- **Bidirectional LSTM** - Process sequences in both directions
- **Transformer** - Attention-based architecture

### 2. Technical Indicators & Feature Engineering

- **RSI (Relative Strength Index)** - Momentum indicator
- **MACD (Moving Average Convergence Divergence)** - Trend indicator
- **Bollinger Bands** - Volatility indicator
- **Moving Averages** - 7-day and 30-day trends
- **Volume Analysis** - Trading volume patterns
- **Price Change Rate** - Percentage changes

### 3. Advanced Forecasting

- **Multi-day predictions** - 1, 7, or 30-day forecasts
- **Ensemble Methods** - Combines multiple models for better accuracy
- **Confidence visualization** - Compare predictions across models

### 4. Enhanced UI/UX

- **Interactive Plotly charts** - Zoom, pan, and hover for details
- **Tabbed interface** - Overview, Predictions, Comparison, Technical Analysis
- **Model comparison dashboard** - Side-by-side metrics
- **Real-time training progress** - Visual feedback during training

## 📦 Installation

```bash
# Install dependencies
pip install streamlit tensorflow pandas numpy yfinance scikit-learn matplotlib plotly ta

# Or use requirements.txt
pip install -r requirements.txt
```

## 🎯 Usage

### Run the basic version:

```bash
streamlit run app.py
```

### Run the enhanced version (Phase 1):

```bash
streamlit run app_enhanced.py
```

## 🛠️ Configuration Options

### Sidebar Controls:

- **Date Range** - Select historical data period
- **Model Selection** - Choose which models to train (LSTM, GRU, Bidirectional LSTM, Transformer)
- **Ensemble Mode** - Average predictions from all selected models
- **Technical Indicators** - Enable/disable technical analysis features
- **Volume Data** - Include trading volume in predictions
- **Forecast Horizon** - Predict 1, 7, or 30 days ahead
- **Model Parameters** - Adjust LSTM units, epochs, and batch size

## 📊 Features Breakdown

### Tab 1: Overview

- Current price and statistics
- Interactive historical price chart
- Latest data table

### Tab 2: Predictions

- Multi-day forecast table
- Tomorrow's predictions from all models
- Forecast visualization with historical context

### Tab 3: Model Comparison

- Performance metrics (MSE, RMSE, MAE, R²)
- Visual metric comparisons
- Predictions vs actual price chart
- Training loss history for all models

### Tab 4: Technical Analysis

- Price with Bollinger Bands
- RSI indicator with overbought/oversold zones
- MACD with signal line
- Current technical indicator values

## 🎓 Model Performance Metrics

- **MSE (Mean Squared Error)** - Average squared difference
- **RMSE (Root Mean Squared Error)** - Standard deviation of errors
- **MAE (Mean Absolute Error)** - Average absolute difference
- **R² Score** - Coefficient of determination (0-1, higher is better)

## 🔮 How It Works

1. **Data Collection** - Download historical SOL-USD data from Yahoo Finance
2. **Feature Engineering** - Calculate technical indicators and normalize data
3. **Sequence Creation** - Create 60-day sequences for time series prediction
4. **Model Training** - Train selected models on 80% of data
5. **Prediction** - Generate forecasts for test set and future days
6. **Ensemble** - Average predictions from multiple models
7. **Visualization** - Display results with interactive charts

## 📈 Technical Indicators Explained

### RSI (Relative Strength Index)

- Range: 0-100
- > 70: Overbought (potential sell signal)
- < 30: Oversold (potential buy signal)

### MACD

- Difference between fast and slow moving averages
- Signal line crossovers indicate trend changes
- Histogram shows momentum strength

### Bollinger Bands

- Upper/Lower bands show volatility
- Price touching bands suggests potential reversal
- Band width indicates market volatility

## 🚨 Disclaimer

**This is an educational project for learning purposes only.**

- Cryptocurrency markets are highly volatile and unpredictable
- Past performance does not guarantee future results
- Do NOT use these predictions for actual trading decisions
- Always do your own research and consult financial advisors
- Trading cryptocurrencies involves significant risk

## 📝 Files

- `solana_price_prediction.py` - Original CLI script
- `app.py` - Basic Streamlit web application
- `app_enhanced.py` - **Enhanced version with Phase 1 features**
- `README.md` - This documentation

## 🔧 Requirements

```
streamlit>=1.28.0
tensorflow>=2.15.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.32
scikit-learn>=1.3.0
matplotlib>=3.7.0
plotly>=5.17.0
ta>=0.11.0
```

## 🎯 Future Enhancements (Phase 2 & 3)

- Real-time price updates
- Sentiment analysis from social media
- More cryptocurrencies support
- Backtesting framework
- REST API endpoints
- Database integration for caching
- Advanced visualization options

## 👨‍💻 Development

To contribute or modify:

1. Clone the repository
2. Create a virtual environment
3. Install dependencies
4. Run the application
5. Make your changes
6. Test thoroughly

## 📄 License

MIT License - Feel free to use for educational purposes

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning
- Streamlit for the amazing web framework
- yfinance for cryptocurrency data
- TA-Lib for technical analysis
- Plotly for interactive visualizations
