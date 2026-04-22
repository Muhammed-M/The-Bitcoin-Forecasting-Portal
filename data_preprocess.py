
import pandas as pd
import numpy as np

KNOWN_DATE_COLUMNS = ['open time', 'date', 'timestamp', 'time', 'datetime']
KNOWN_PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close']

def detect_available_price_columns(uploaded_file):
    """
    Peek at the CSV and return which of the standard price columns exist.
    Returns (list_of_available_cols, error_string_or_None)
    """
    try:
        df = pd.read_csv(uploaded_file)
        available = [col for col in KNOWN_PRICE_COLUMNS if col in df.columns]
        if not available:
            return [], (
                f"No standard price columns found. "
                f"Expected one of: {', '.join(KNOWN_PRICE_COLUMNS)}. "
                f"Columns in your file: {', '.join(df.columns.tolist()[:10])}."
            )
        return available, None
    except Exception as e:
        return [], f"Could not read CSV: {str(e)}"


def load_and_prepare_btc_data(uploaded_file, price_col='Close'):
    """
    Load minute-level or daily BTC data, resample to daily, and prepare for forecasting.
    Returns (df, error_string_or_None)
    """
    try:
        df = pd.read_csv(uploaded_file)

        # --- Detect Timestamp Column ---
        date_col = None
        for col in df.columns:
            if col.strip().lower() in KNOWN_DATE_COLUMNS:
                date_col = col
                break

        if date_col is None:
            # Fallback: try first column
            date_col = df.columns[0]
            try:
                pd.to_datetime(df[date_col].iloc[:5])
            except Exception:
                return None, (
                    "Could not identify a date/time column. "
                    f"Expected one of: {', '.join(KNOWN_DATE_COLUMNS)} (case-insensitive). "
                    f"First column tried: '{date_col}'. "
                    "Please ensure your CSV has a recognisable timestamp column."
                )

        # --- Parse datetime ---
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception:
            return None, (
                f"Could not parse '{date_col}' as dates. "
                "Ensure the column contains valid date/time values (e.g., '2021-01-01' or Unix timestamps)."
            )

        # --- Price Column Validation ---
        if price_col not in df.columns:
            available_cols = ', '.join(df.columns.tolist()[:15])
            return None, (
                f"Price column '{price_col}' not found in your file. "
                f"Available columns: {available_cols}. "
                f"Please select a valid price column from the sidebar."
            )

        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        nan_count = df[price_col].isna().sum()
        if nan_count == len(df):
            return None, (
                f"Column '{price_col}' contains no numeric values. "
                "Please select a column with numeric price data."
            )

        # Drop NaN in timestamp or price
        df = df.dropna(subset=[date_col, price_col])

        if len(df) < 30:
            return None, (
                f"Not enough data after cleaning — only {len(df)} valid rows found. "
                "At least 30 rows are required for forecasting."
            )

        # Sort chronologically
        df = df.sort_values(date_col).reset_index(drop=True)

        # --- Resample to Daily ---
        df = df.set_index(date_col)
        df_daily = df[price_col].resample('D').last().dropna()

        # Create a complete daily date range (no gaps)
        full_range = pd.date_range(
            start=df_daily.index.min(),
            end=df_daily.index.max(),
            freq='D'
        )
        df_daily = df_daily.reindex(full_range)
        # Forward fill missing days (crypto trades 24/7)
        df_daily = df_daily.ffill().dropna()

        if len(df_daily) < 30:
            return None, (
                "After resampling to daily frequency, fewer than 30 days of data remain. "
                "Please upload a file with a longer date range."
            )

        # Prepare output for Prophet (ds, y)
        result_df = pd.DataFrame({
            'ds': df_daily.index,
            'y': df_daily.values
        })

        return result_df, None

    except Exception as e:
        return None, (
            f"Unexpected error while processing file: {str(e)}. "
            "Please check that your file is a valid CSV with numeric price data and a date column."
        )


def create_ml_features(df, target='y', lags=[1, 2, 3, 7], windows=[7, 14]):
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

    y = data[target]

    # 1. Calendar features (cyclical encoding for linear models)
    data['dayofweek'] = data.index.dayofweek
    data['dayofyear'] = data.index.dayofyear
    data['month'] = data.index.month
    data['quarter'] = data.index.quarter
    data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)

    data['dayofweek_sin'] = np.sin(2 * np.pi * data['dayofweek'] / 7)
    data['dayofweek_cos'] = np.cos(2 * np.pi * data['dayofweek'] / 7)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    # 2. Lag features
    for lag in lags:
        data[f'lag_{lag}'] = y.shift(lag)

    # 3. Rolling statistics (shift(1) so today's feature doesn't include today)
    shifted_y = y.shift(1)
    for w in windows:
        data[f'rolling_mean_{w}'] = shifted_y.rolling(window=w, min_periods=1).mean()
        data[f'rolling_std_{w}'] = shifted_y.rolling(window=w, min_periods=1).std()

    # 4. Returns
    data['return_1'] = y.pct_change(1)
    data['return_7'] = y.pct_change(7)

    # 5. Volatility
    data['volatility_7'] = data['return_1'].rolling(7).std()

    # 6. Momentum
    data['mom_7'] = y - data['rolling_mean_7']

    data = data.dropna()
    return data
