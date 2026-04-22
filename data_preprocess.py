
import pandas as pd
import numpy as np

def load_and_prepare_btc_data(uploaded_file, price_col='Close'):
    """
    Load minute-level BTC data, resample to daily, and prepare for forecasting.
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # --- Detect Timestamp Column ---
        date_col = None
        possible_names = ['open time', 'Open time', 'date', 'timestamp', 'time', 'datetime', 'timestamp']
        for col in df.columns:
            if col.lower() in [name.lower() for name in possible_names]:
                date_col = col
                break
        
        if date_col is None:
            # Fallback: try first column
            date_col = df.columns[0]
            try:
                pd.to_datetime(df[date_col])
            except:
                return None, "Could not identify a timestamp column."
        
        # Parse datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # --- Price Column Validation ---
        if price_col not in df.columns:
            return None, f"Price column '{price_col}' not found. Available: {', '.join(df.columns)}"
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        
        # Drop NaN in timestamp or price
        df = df.dropna(subset=[date_col, price_col])
        
        # Sort chronologically
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # --- Resample to Daily ---
        # Set timestamp as index
        df = df.set_index(date_col)

        df_daily = df[price_col].resample('D').last().dropna()
        
        # Create a complete daily date range (no gaps)
        full_range = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
        
        df_daily = df_daily.reindex(full_range)
        # Forward fill missing days (crypto trades 24/7, so ffill is safe)
        df_daily = df_daily.ffill()
        # Drop any leading NaN
        df_daily = df_daily.dropna()
        
        # Prepare output for Prophet (ds, y)
        result_df = pd.DataFrame({
            'ds': df_daily.index,
            'y': df_daily.values
        })
        
        return result_df, None
        
    except Exception as e:
        return None, f"Error processing file: {str(e)}"
    


def create_ml_features(df, target='y', lags=[1,2,3,7], windows=[7,14]):
    """
    Add time series features for machine learning forecasting.
    Assumes df has datetime index and a column named `target`.
    Returns a new DataFrame with features and target (no NaN rows).
    """
    data = df.copy()
    
    # Ensure datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'ds' in data.columns:
            data['ds'] = pd.to_datetime(data['ds'])
            data = data.set_index('ds')
        else:
            raise ValueError("DataFrame must have a datetime index or 'ds' column.")
    
    # Target series
    y = data[target]
    
    # 1. Calendar features (cyclical encoding for linear models)
    data['dayofweek'] = data.index.dayofweek
    data['dayofyear'] = data.index.dayofyear
    data['month'] = data.index.month
    data['quarter'] = data.index.quarter
    data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    
    # Cyclical encoding for day of week (0-6)
    data['dayofweek_sin'] = np.sin(2 * np.pi * data['dayofweek'] / 7)
    data['dayofweek_cos'] = np.cos(2 * np.pi * data['dayofweek'] / 7)
    
    # Cyclical encoding for month (1-12)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    
    # 2. Lag features (past prices)
    for lag in lags:
        data[f'lag_{lag}'] = y.shift(lag)
    
    # 3. Rolling statistics (computed on past data only)
    # We use shift(1) so that today's feature doesn't include today's price
    shifted_y = y.shift(1)
    for w in windows:
        data[f'rolling_mean_{w}'] = shifted_y.rolling(window=w, min_periods=1).mean()
        data[f'rolling_std_{w}'] = shifted_y.rolling(window=w, min_periods=1).std()
    
    # 4. Returns (percentage change) – often more stationary
    data['return_1'] = y.pct_change(1)
    data['return_7'] = y.pct_change(7)
    
    # 5. Volatility (rolling std of returns)
    data['volatility_7'] = data['return_1'].rolling(7).std()
    
    # 6. Price momentum (difference from moving average)
    data['mom_7'] = y - data['rolling_mean_7']
    
    # Drop rows with NaN created by lags/rolling
    data = data.dropna()
    
    return data