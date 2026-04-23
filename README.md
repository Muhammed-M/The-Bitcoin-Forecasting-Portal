# ₿ Bitcoin Price Forecasting Portal

> An interactive web app for analyzing and forecasting Bitcoin prices — built with Streamlit, Prophet, and XGBoost.

**🔗 Live App → [Open in Streamlit](https://the-bitcoin-forecasting-app-ypahyzgmh6vz7vkedshtjm.streamlit.app/#bitcoin-price-forecast)**

---

## What It Does

Upload a Bitcoin historical CSV, train a forecasting model, and generate an interactive price prediction chart — all in your browser with no code required.

- Supports any Kaggle-style BTC CSV (minute-level or daily)
- Two forecasting models: **Prophet** and **Hybrid ML** (ElasticNet + XGBoost)
- Adjustable forecast horizon (7–90 days) and confidence intervals (80% – 99%)
- Backtest metrics: MAE and RMSE in USD
- Downloadable forecast CSV

---

## How to Use It

The sidebar walks you through three steps:

**① Data** — Upload your CSV and select a price column (`Open`, `High`, `Low`, or `Close`)

**② Train Model** — Pick an algorithm and confidence interval, then click **⚡ Train Model**

**③ Forecast** — Set how many days ahead to predict, then click **📈 Generate Forecast**

> If you change the algorithm or confidence after training, the app will warn you to retrain before forecasting again.

> Hybrid ML (ElasticNet + XGBoost) may take a few minutes to train then you can generate the forecast 

---

## Dataset

Tested with the **Bitcoin Historical Data** dataset from Kaggle:

🔗 [kaggle.com/datasets/mczielinski/bitcoin-historical-data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

Also works with daily-level CSVs like:

🔗 [kaggle.com/datasets/prasoonkottarathil/btcinusd](https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd)

### Expected CSV Format

Your file needs at minimum:

| Column type | Accepted names |
|---|---|
| Date / Time | `Date`, `Open time`, `Timestamp`, `datetime`, `time` |
| Price | `Open`, `High`, `Low`, `Close` |

```
Open time,Open,High,Low,Close,Volume
2021-01-01,29000,29500,28800,29300,1200
2021-01-02,29300,30100,29100,30000,1350
```

Missing trading days are forward-filled automatically.

---

## Models

### Prophet
Facebook's Prophet decomposes the price series into trend, weekly seasonality, and yearly seasonality. It uses **multiplicative** seasonality mode — appropriate for Bitcoin because price swings scale with the price level (a 10% move at $100k is very different in dollar terms than at $10k). The `changepoint_prior_scale` is set conservatively at `0.05` to avoid overfitting sharp but temporary spikes.

### Hybrid ML (ElasticNet + XGBoost)
A two-stage ensemble:
1. **ElasticNetCV** captures the long-term linear trend using calendar features, lag prices, and rolling statistics. Cross-validated regularisation prevents it from memorising noise.
2. **XGBoost** is trained on the residuals — the non-linear patterns the linear model misses, including volatility clusters and momentum effects.

Forecasting is done **recursively**: each predicted price feeds back into the feature window for the next step. Confidence intervals are derived from the standard deviation of in-sample residuals.

---

## Local Setup

```bash
# 1. Clone or unzip the project

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

> **Note on Prophet:** On some systems, Prophet requires `pystan` to be installed first:
> ```bash
> pip install pystan==2.19.1.1
> pip install prophet
> ```

---

## Project Files

```
btc-forecaster/
├── app.py               # Streamlit application — UI, session state, train/predict flow
├── data_preprocess.py   # CSV loading, date parsing, resampling, feature engineering
├── models.py            # ProphetForecaster and HybridMLForecaster classes
├── style.css            # Dark terminal theme — Plus Jakarta Sans + DM Mono
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.28 | Web UI framework |
| `plotly` | ≥5.14 | Interactive charts |
| `prophet` | ≥1.1 | Prophet model |
| `scikit-learn` | ≥1.2 | ElasticNetCV, StandardScaler, metrics |
| `xgboost` | ≥1.7 | Gradient boosting residual model |
| `statsmodels` | ≥0.14 | Time series utilities |
| `pandas` | ≥1.5 | Data manipulation |
| `numpy` | ≥1.23 | Numerical computing |
| `pyarrow` | ≥12.0 | Arrow serialisation for pandas/streamlit |

---