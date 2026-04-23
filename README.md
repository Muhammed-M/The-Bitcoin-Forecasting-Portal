# ₿ Bitcoin Price Forecasting Portal

An interactive Streamlit web application for analyzing and forecasting Bitcoin (BTC) price trends using multiple time-series models.

---

## Features

- Upload any Kaggle-style Bitcoin historical CSV (minute-level or daily)
- Choose which OHLC price column to forecast (`Open`, `High`, `Low`, `Close`)
- Three forecasting models: **Prophet**, **ARIMA**, and **Hybrid ML** (ElasticNet + XGBoost)
- Adjustable forecast horizon (7–90 days) and confidence interval (80%, 90%, 95%, 99%)
- Backtesting with MAE and RMSE metrics in USD
- Interactive Plotly chart with historical data, forecast line, and uncertainty bands
- Downloadable forecast CSV

---

## Setup

### 1. Clone or unzip the project

```bash
cd btc-forecaster
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Prophet requires `pystan`. On some systems you may need to install it separately:
> ```bash
> pip install pystan==2.19.1.1
> pip install prophet
> ```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Dataset

This app was tested with the **Bitcoin Historical Data** dataset from Kaggle:

🔗 [https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

Download `btcusd_1-min_data.csv` and upload it directly through the app's sidebar.

The app also supports daily-level CSVs such as:

🔗 [https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd](https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd)

### Expected CSV Format

Your CSV must contain:
- A date/time column named one of: `Date`, `Timestamp`, `Open time`, `time`, `datetime`
- At least one price column: `Open`, `High`, `Low`, `Close`

Example:
```
Open time,Open,High,Low,Close
2021-01-01,29000,29500,28800,29300
2021-01-02,29300,30100,29100,30000
```

---

## Project Structure

```
btc-forecaster/
├── app.py               # Main Streamlit application
├── data_preprocess.py   # Data loading, resampling, and feature engineering
├── models.py            # Prophet, ARIMA, and Hybrid ML model classes
├── style.css            # Custom UI styling
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Model Explanations

### Prophet
Facebook's Prophet is designed for time series with strong seasonal patterns and trend shifts — both very common in crypto markets. It uses a decomposable model (trend + seasonality + holidays) and handles missing data and outliers well. The `changepoint_prior_scale` is set conservatively (0.05) to avoid overfitting Bitcoin's sharp but temporary price spikes. Seasonality is set to **multiplicative** mode because Bitcoin's price swings are proportional to its level (e.g., a 10% move at $60,000 is far larger in absolute terms than at $6,000).

### ARIMA(5,1,0)
ARIMA (AutoRegressive Integrated Moving Average) is a classical statistical approach. The `d=1` (first-order differencing) makes the series stationary by modelling **returns** rather than raw prices — essential for financial data which is non-stationary. The `p=5` autoregressive terms capture short-term momentum (up to 5 days of autocorrelation in returns). ARIMA is best suited for short-horizon forecasts and provides analytically derived confidence intervals from its residual variance.

### Hybrid ML (ElasticNet + XGBoost)
A two-stage ensemble approach:
1. **ElasticNetCV** (L1 + L2 regularisation) captures the long-term linear trend using calendar and lag features. Cross-validated regularisation prevents overfitting.
2. **XGBoost** is trained on the residuals — the non-linear patterns that the linear model misses. It picks up on volatility clusters, momentum, and mean-reversion effects.

Multi-step forecasting is done **recursively**: each predicted price feeds back into the rolling feature window for the next step, mimicking how future lags would behave. Confidence intervals are estimated using the standard deviation of in-sample residuals, scaled by a z-score corresponding to the chosen confidence level.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `plotly` | Interactive charts |
| `prophet` | Prophet forecasting model |
| `statsmodels` | ARIMA model |
| `scikit-learn` | ElasticNetCV, preprocessing, metrics |
| `xgboost` | Gradient boosting residual model |
| `pandas` / `numpy` | Data manipulation |
| `pyarrow` | Arrow-based data serialisation (required by pandas/streamlit) |
