# Solana Price Predictor - Vercel Edition

Advanced cryptocurrency price prediction using LSTM neural networks. Now deployable on Vercel! 🚀

## 📱 Access Your App

Once deployed to Vercel, your app will be accessible at:

```
https://your-project-name.vercel.app
```

## ✨ Features

- **AI-Powered Predictions**: Uses LSTM deep learning networks
- **Real-time Data**: Fetches live crypto prices from Yahoo Finance
- **Interactive Charts**: Beautiful visualization of predictions
- **Multi-Crypto Support**: Solana, Bitcoin, Ethereum
- **Customizable Parameters**: Adjust history and forecast periods
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🏗️ Project Structure

```
solana-price-predictor/
├── api/
│   └── predict.py              # Serverless API for predictions
├── public/
│   └── index.html              # Frontend interface
├── vercel.json                 # Vercel configuration
├── requirements.txt            # Python dependencies
├── package.json                # Project metadata
├── VERCEL_DEPLOYMENT_GUIDE.md  # Full deployment guide
└── [other files...]
```

## 🚀 Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the API locally
vercel dev

# 3. Visit http://localhost:3000
```

### Deploy to Vercel

#### Option 1: Using GitHub (Recommended)

1. Push your code to GitHub
2. Visit [vercel.com/new](https://vercel.com/new)
3. Import your repository
4. Click "Deploy"

#### Option 2: Using Vercel CLI

```bash
# Install Vercel CLI globally
npm install -g vercel

# Deploy from your project directory
vercel
```

## 📊 How It Works

1. **Data Collection**: Fetches historical cryptocurrency data
2. **Preprocessing**: Normalizes data and creates sequences
3. **Model Training**: Trains LSTM neural network
4. **Prediction**: Generates future price forecasts
5. **Visualization**: Displays results in interactive chart

## ⚠️ Important Notes

### For Free Vercel Deployments

- **Timeout Limit**: 10 seconds (free tier)
- **Model Training**: Takes 30-60 seconds
- **Solution**: Upgrade to Vercel Pro ($20/month) for 60-second timeout

### For Best Results

- Use 365+ days of historical data
- Keep forecast period under 90 days
- Train on Vercel Pro for faster predictions

## 📚 Documentation

See [VERCEL_DEPLOYMENT_GUIDE.md](./VERCEL_DEPLOYMENT_GUIDE.md) for:

- Detailed deployment steps
- Troubleshooting guide
- Production recommendations
- Environment configuration

## 🔧 API Reference

### POST /api/predict

Request:

```json
{
  "symbol": "SOL-USD",
  "days": 365,
  "forecast_days": 30
}
```

Response:

```json
{
  "success": true,
  "current_price": 150.25,
  "symbol": "SOL-USD",
  "predictions": [151.30, 152.45, ...],
  "forecast_days": 30
}
```

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Backend**: Python, TensorFlow/Keras
- **Data**: yfinance, pandas, NumPy
- **ML**: scikit-learn, LSTM neural networks
- **Deployment**: Vercel serverless functions

## 📈 Model Details

- **Architecture**: LSTM (Long Short-Term Memory)
- **Input Features**: Historical closing prices
- **Sequence Length**: 60 days
- **Training Epochs**: 10 (optimized for Vercel)
- **Batch Size**: 32
- **Optimizer**: Adam (learning rate: 0.001)

## 🐛 Troubleshooting

**Q: Predictions timeout?**

- A: Upgrade to Vercel Pro for 60-second timeout

**Q: Getting CORS errors?**

- A: Check API response in browser console

**Q: Chart not displaying?**

- A: Clear browser cache and refresh

## 📝 Example Predictions

```
Current SOL Price: $150.25
30-Day Forecast: $165.75
Expected Change: +10.3%
```

## 🔐 Security & Privacy

- No API keys stored in frontend code
- All data fetched from public sources
- No user data collection
- Predictions are real-time calculations

## 📞 Support

- **Vercel Docs**: https://vercel.com/docs
- **Python Functions**: https://vercel.com/docs/functions/serverless-functions/python
- **Issue Tracker**: Create a GitHub issue

## 📄 License

MIT License - Feel free to use for personal and commercial projects

## 🎉 Ready to Deploy?

1. **Prepare**: Push to GitHub
2. **Deploy**: Click "Import on Vercel"
3. **Share**: Copy your Vercel URL
4. **Monitor**: Check Vercel dashboard for analytics

---

**Your Solana Price Predictor is ready for production!** 🌟

For detailed deployment instructions, see [VERCEL_DEPLOYMENT_GUIDE.md](./VERCEL_DEPLOYMENT_GUIDE.md)
