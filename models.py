# models.py

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
import logging
from collections import deque

logging.getLogger('prophet').setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ─────────────────────────── Prophet ───────────────────────────
class ProphetForecaster:
    """
    Facebook Prophet wrapper — handles trend changepoints,
    multiplicative seasonality (weekly + yearly), and uncertainty
    intervals natively. Well-suited for crypto's volatile, seasonal patterns.
    """

    def __init__(self):
        self.model = None
        self.history_df = None
        self.forecast_df = None
        self.metrics = {}

    def fit(self, df, confidence=0.95):
        self.history_df = df.copy()
        self.model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=confidence,
        )
        self.model.fit(df)

    def predict(self, horizon, confidence=0.95):
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        self.model.interval_width = confidence
        future = self.model.make_future_dataframe(periods=horizon)
        forecast = self.model.predict(future)
        self.forecast_df = forecast.iloc[-horizon:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        return self.forecast_df

    def evaluate(self, df_train, df_test, confidence=0.95):
        self.fit(df_train, confidence)
        forecast = self.model.predict(df_test[['ds']])
        y_true = df_test['y'].values
        y_pred = forecast['yhat'].values
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        self.metrics = {'MAE': mae, 'RMSE': rmse}
        return self.metrics



# ─────────────────────────── Hybrid ML ───────────────────────────
class HybridMLForecaster:
    """
    Two-stage model: ElasticNetCV captures the linear trend component;
    XGBoost models the non-linear residuals. Recursive multi-step forecasting
    with a rolling price buffer so features stay valid at each future step.
    """

    def __init__(self, lags=[1, 2, 3, 7], windows=[7, 14], max_window=30):
        self.lags = lags
        self.windows = windows
        self.max_window = max_window
        self.trend_model = None
        self.residual_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.last_features = None
        self.last_date = None
        self.price_buffer = None
        self.y_train_ = None
        self.X_train_ = None
        self.residual_std_ = None
        self.metrics = {}

    def _create_features_from_buffer(self, date, price_buffer):
        prices = np.array(price_buffer)
        n = len(prices)
        features = {}

        features['dayofweek'] = date.dayofweek
        features['dayofyear'] = date.dayofyear
        features['month'] = date.month
        features['quarter'] = date.quarter
        features['is_weekend'] = 1 if date.dayofweek >= 5 else 0
        features['dayofweek_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)
        features['month_sin'] = np.sin(2 * np.pi * date.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * date.month / 12)

        for lag in self.lags:
            features[f'lag_{lag}'] = prices[-lag] if n >= lag else prices[-1]

        for w in self.windows:
            window = prices[-w:] if n >= w else prices
            features[f'rolling_mean_{w}'] = np.mean(window)
            features[f'rolling_std_{w}'] = np.std(window)

        features['return_1'] = (prices[-1] / prices[-2] - 1) if n >= 2 else 0
        features['return_7'] = (prices[-1] / prices[-8] - 1) if n >= 8 else 0

        if n >= 7:
            rets = np.diff(prices[-7:]) / prices[-7:-1]
            features['volatility_7'] = np.std(rets)
        else:
            features['volatility_7'] = 0

        mean7 = features.get('rolling_mean_7', np.mean(prices[-min(7, n):]))
        features['mom_7'] = prices[-1] - mean7

        return pd.Series(features)

    def fit(self, df_features, target='y'):
        self.original_prices = df_features[target].copy()
        X = df_features.drop(target, axis=1)
        y = df_features[target]
        self.feature_columns = X.columns.tolist()

        X_scaled = self.scaler.fit_transform(X)
        self.X_train_ = X
        self.y_train_ = y

        self.trend_model = ElasticNetCV(
            l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
            cv=5,
            max_iter=2000,
            random_state=42,
        )
        self.trend_model.fit(X_scaled, y)
        trend_pred = self.trend_model.predict(X_scaled)

        residuals = y - trend_pred
        self.residual_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
        )
        self.residual_model.fit(X_scaled, residuals)

        final_pred = trend_pred + self.residual_model.predict(X_scaled)
        self.residual_std_ = np.std(y - final_pred)

        buffer_size = max(max(self.lags), max(self.windows)) + 1
        self.price_buffer = deque(
            self.original_prices.iloc[-buffer_size:].tolist(), maxlen=buffer_size
        )
        self.last_features = X.iloc[-1:].copy()
        self.last_date = df_features.index[-1]

    def predict(self, horizon, confidence=0.95):
        if self.trend_model is None or self.residual_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions, lower_bound, upper_bound, dates = [], [], [], []
        current_date = self.last_date
        buffer = self.price_buffer.copy()

        z_score = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96, 0.99: 2.58}.get(confidence, 1.96)

        for _ in range(horizon):
            next_date = current_date + pd.Timedelta(days=1)
            features_series = self._create_features_from_buffer(next_date, buffer)
            features_df = pd.DataFrame([features_series])[self.feature_columns]
            X_scaled = self.scaler.transform(features_df)

            trend = self.trend_model.predict(X_scaled)[0]
            residual = self.residual_model.predict(X_scaled)[0]
            y_pred = trend + residual

            predictions.append(y_pred)
            dates.append(next_date)
            margin = z_score * self.residual_std_
            lower_bound.append(y_pred - margin)
            upper_bound.append(y_pred + margin)

            buffer.append(y_pred)
            current_date = next_date

        return pd.DataFrame({
            'ds': dates,
            'yhat': predictions,
            'yhat_lower': lower_bound,
            'yhat_upper': upper_bound,
        })

    def evaluate(self, df_train_features, df_test_features, target='y'):
        self.fit(df_train_features, target)
        X_test = df_test_features.drop(target, axis=1)
        y_test = df_test_features[target]
        X_test_scaled = self.scaler.transform(X_test)
        trend_test = self.trend_model.predict(X_test_scaled)
        residual_test = self.residual_model.predict(X_test_scaled)
        y_pred = trend_test + residual_test
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        self.metrics = {'MAE': mae, 'RMSE': rmse}
        return self.metrics
