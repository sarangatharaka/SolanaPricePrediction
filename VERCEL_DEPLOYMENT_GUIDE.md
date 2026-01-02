# Deploying to Vercel

This guide explains how to deploy the Solana Price Predictor to Vercel.

## Project Structure Changes

The project has been restructured for Vercel deployment:

```
├── api/
│   └── predict.py          # Serverless API endpoint for predictions
├── public/
│   └── index.html          # Web interface
├── vercel.json             # Vercel configuration
├── .vercelignore           # Files to ignore during deployment
├── package.json            # Project metadata
├── requirements.txt        # Python dependencies
└── [original files...]
```

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com) (free tier available)
2. **Git Repository**: Push your code to GitHub, GitLab, or Bitbucket
3. **Node.js 18+**: Installed locally (for `vercel` CLI)

## Deployment Steps

### 1. Prepare Your Repository

```bash
# Initialize git if not already done
git init
git add .
git commit -m "Prepare project for Vercel deployment"
```

### 2. Push to GitHub

```bash
# Create a new repository on GitHub
# Then push your code:
git remote add origin https://github.com/YOUR_USERNAME/solana-price-predictor.git
git branch -M main
git push -u origin main
```

### 3. Install Vercel CLI (Optional but Recommended)

```bash
npm install -g vercel
```

### 4. Deploy to Vercel

#### Option A: Using GitHub (Recommended)

1. Visit [vercel.com/new](https://vercel.com/new)
2. Click "Import Git Repository"
3. Select your GitHub repository
4. Vercel will auto-detect the configuration
5. Click "Deploy"

#### Option B: Using Vercel CLI

```bash
vercel
```

Follow the prompts to:

- Link your Vercel account
- Select your project scope
- Deploy

### 5. Configure Environment Variables (if needed)

In your Vercel Dashboard:

1. Go to your project settings
2. Navigate to "Environment Variables"
3. Add any required variables

## Deployment Considerations

### Python Dependencies

- **TensorFlow**: Heavy library (~500MB). Vercel supports it but deployment may take 3-5 minutes.
- **yfinance**: Requires internet access to fetch data (works on Vercel)
- **scikit-learn**: Fully supported

### Timeout

- Standard Vercel functions have a **10-second timeout** for free tier
- **Problem**: Training the LSTM model takes longer than 10 seconds
- **Solutions**:

  **Option 1: Use Pro Plan** ($20/month) - 60-second timeout

  **Option 2: Model Caching** (Advanced)

  - Pre-train models locally
  - Store model weights in the serverless function
  - Load pre-trained weights instead of training from scratch

  **Option 3: Background Job Service**

  - Use a service like Firebase Cloud Functions or AWS Lambda
  - Call from Vercel frontend

### File Size Limits

- Vercel deployment package: **250MB max**
- TensorFlow can be large; monitor your build output

## Testing Locally Before Deployment

### 1. Test the API Locally

```bash
# Install Vercel CLI
npm install -g vercel

# Run locally
vercel dev
```

Then visit: `http://localhost:3000`

### 2. Test Predictions

The frontend will automatically send requests to the API:

```
POST /api/predict
Content-Type: application/json

{
  "symbol": "SOL-USD",
  "days": 365,
  "forecast_days": 30
}
```

## Frontend Features

The new HTML/JavaScript frontend includes:

✅ **Clean, Modern UI**

- Responsive design (mobile-friendly)
- Real-time prediction display
- Interactive chart visualization

✅ **Multiple Cryptocurrencies**

- Solana (SOL-USD)
- Bitcoin (BTC-USD)
- Ethereum (ETH-USD)

✅ **Customizable Parameters**

- Historical data range (30-1000 days)
- Forecast period (1-90 days)

✅ **Live Results**

- Current price display
- 30-day forecast
- Percentage change indicator
- Prediction chart

## Important Notes

### CORS Configuration

The API includes CORS headers to allow requests from your frontend domain. This is already configured in `api/predict.py`.

### Model Training Time

- **Free/Hobby Plan**: May experience timeout if model training exceeds limits
- **Recommended**: Upgrade to Pro Plan for production use

### Data Availability

- The API fetches live data from Yahoo Finance (yfinance)
- Internet connection required
- Some cryptocurrencies may have limited historical data

## Troubleshooting

### Deployment Fails

- Check build logs: `vercel logs <project-url>`
- Ensure all dependencies are in `requirements.txt`
- Verify Python version compatibility

### Predictions Timeout

- Model training takes >10 seconds on free plan
- Solution: Upgrade to Vercel Pro ($20/month)
- Or implement model caching

### CORS Errors

- Check browser console for details
- Verify API endpoint is responding
- Clear browser cache and reload

## Next Steps for Production

1. **Add authentication** if needed
2. **Implement caching** to reduce API calls
3. **Add error handling** and logging
4. **Monitor performance** with Vercel Analytics
5. **Set up custom domain** in Vercel settings

## Commands Reference

```bash
# Deploy to production
vercel --prod

# View deployment logs
vercel logs

# List all deployments
vercel list

# Revert to previous deployment
vercel rollback
```

## Support & Resources

- **Vercel Docs**: https://vercel.com/docs
- **Python on Vercel**: https://vercel.com/docs/functions/serverless-functions/python
- **Vercel Community**: https://vercel.com/support

---

**Your project is now ready for Vercel deployment!** 🚀
