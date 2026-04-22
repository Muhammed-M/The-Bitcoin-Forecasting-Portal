# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data_preprocess import load_and_prepare_btc_data, create_ml_features, detect_available_price_columns
from models import ProphetForecaster, ARIMAForecaster, HybridMLForecaster

# ─────────────────────────── PAGE CONFIG ───────────────────────────
st.set_page_config(
    page_title="Bitcoin Price Forecaster",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# ─────────────────────────── SESSION STATE ───────────────────────────
defaults = {
    'data_loaded': False,
    'df_daily': None,
    'df_features': None,
    'model_trained': False,
    'forecast': None,
    'metrics': None,
    'available_price_cols': [],
    'last_price_col': None,
    'last_uploaded_name': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────── SIDEBAR ───────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bitcoin--v1.png", width=60)
    st.title("₿ BTC Forecaster")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload Bitcoin CSV",
        type=["csv"],
        help="Upload a Kaggle-style Bitcoin historical CSV (minute-level or daily).",
    )

    # ── Price column selector (shown after file upload) ──
    price_col = 'Close'  # default fallback
    if uploaded_file is not None:
        file_name = uploaded_file.name

        # Only re-scan columns when a new file is uploaded
        if file_name != st.session_state.last_uploaded_name:
            available_cols, col_error = detect_available_price_columns(uploaded_file)
            st.session_state.available_price_cols = available_cols
            st.session_state.last_uploaded_name = file_name
            st.session_state.data_loaded = False  # force reload
            uploaded_file.seek(0)  # reset file pointer after peek

        if st.session_state.available_price_cols:
            price_col = st.selectbox(
                "💲 Price Column",
                options=st.session_state.available_price_cols,
                index=st.session_state.available_price_cols.index('Close')
                      if 'Close' in st.session_state.available_price_cols else 0,
                help="Select which OHLC price column to use for forecasting.",
            )
        else:
            st.error("No standard price columns (Open/High/Low/Close) found in this file.")

    # ── Load data when file + price col are ready ──
    if uploaded_file is not None and st.session_state.available_price_cols:
        needs_reload = (
            not st.session_state.data_loaded
            or price_col != st.session_state.last_price_col
        )

        if needs_reload:
            uploaded_file.seek(0)

            @st.cache_data(show_spinner=False)
            def load_data(file_bytes, file_name, price_col):
                import io
                return load_and_prepare_btc_data(io.BytesIO(file_bytes), price_col)

            file_bytes = uploaded_file.read()
            with st.spinner("Processing data..."):
                df_daily, error = load_data(file_bytes, file_name, price_col)

            if error:
                st.error(f"❌ {error}")
                with st.expander("📋 Expected CSV format"):
                    st.markdown("""
**Required columns:**
- A date/time column named one of: `Date`, `Timestamp`, `Open time`, `time`, `datetime`
- At least one price column: `Open`, `High`, `Low`, `Close`

**Example:**
```
Open time,Open,High,Low,Close
2021-01-01,29000,29500,28800,29300
2021-01-02,29300,30100,29100,30000
```
                    """)
                st.session_state.data_loaded = False
            else:
                st.session_state.df_daily = df_daily
                st.session_state.data_loaded = True
                st.session_state.last_price_col = price_col
                st.session_state.model_trained = False  # reset on new data
                st.success(f"✅ Loaded {len(df_daily):,} days of data")

                with st.expander("🔍 Data Preview"):
                    st.dataframe(df_daily.head(10), use_container_width=True)
                    st.caption(
                        f"Date range: {df_daily['ds'].min().date()} → {df_daily['ds'].max().date()}"
                    )
    elif uploaded_file is None:
        st.info("👆 Upload a CSV file to begin")
        st.session_state.data_loaded = False

    st.markdown("---")

    # ── Forecast Settings ──
    st.subheader("⚙️ Forecast Settings")

    data_ready = st.session_state.data_loaded

    model_choice = st.selectbox(
        "Model",
        ["Prophet", "ARIMA", "Hybrid ML (ElasticNet + XGBoost)"],
        disabled=not data_ready,
        help=(
            "**Prophet** – robust to seasonality and trend shifts.\n"
            "**ARIMA** – classic statistical model, good for short-term patterns.\n"
            "**Hybrid ML** – combines linear trend (ElasticNet) with gradient boosting (XGBoost)."
        ),
    )

    horizon = st.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1,
        disabled=not data_ready,
    )

    confidence = st.select_slider(
        "Confidence Interval",
        options=[0.80, 0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x * 100)}%",
        disabled=not data_ready,
        help="Width of the uncertainty band around the forecast.",
    )

    show_ma = st.checkbox(
        "Show 7-day Moving Average",
        value=False,
        disabled=not data_ready,
    )

    st.markdown("---")

    generate_btn = st.button(
        "🚀 Generate Forecast",
        type="primary",
        disabled=not data_ready,
        use_container_width=True,
    )

# ─────────────────────────── MAIN PANEL ───────────────────────────
st.title("Bitcoin Price Forecast")
st.markdown("Analyze historical trends and predict future prices with confidence intervals.")

if not st.session_state.data_loaded:
    st.info("📊 Upload your Bitcoin CSV file in the sidebar to start.")
    with st.expander("📋 Expected CSV Format"):
        st.markdown("""
Your CSV should contain at least:
- A timestamp column (e.g., `Open time`, `Date`, `Timestamp`)
- Price columns: `Open`, `High`, `Low`, `Close` (at least one)
        """)
    st.stop()

# ─────────────────────────── DATA PREP ───────────────────────────
df_daily = st.session_state.df_daily.copy()

split_idx = int(len(df_daily) * 0.8)
df_train = df_daily.iloc[:split_idx].copy()
df_test = df_daily.iloc[split_idx:].copy()

@st.cache_data(show_spinner=False)
def get_ml_features(df_json):
    from io import StringIO
    df = pd.read_json(StringIO(df_json))
    df['ds'] = pd.to_datetime(df['ds'])
    df_feat = create_ml_features(df.set_index('ds'))
    return df_feat

if "Hybrid ML" in model_choice:
    df_features = get_ml_features(df_daily.to_json())
    train_features = df_features.iloc[:split_idx].copy()
    test_features = df_features.iloc[split_idx:].copy()

# ─────────────────────────── FORECASTING ───────────────────────────
if generate_btn:
    with st.spinner("Training model and generating forecast…"):
        metrics = {}
        forecast = None

        try:
            if model_choice == "Prophet":
                forecaster = ProphetForecaster()
                metrics = forecaster.evaluate(df_train, df_test, confidence)
                forecaster.fit(df_daily, confidence)
                forecast = forecaster.predict(horizon, confidence)

            elif model_choice == "ARIMA":
                forecaster = ARIMAForecaster(order=(5, 1, 0))
                metrics = forecaster.evaluate(df_train, df_test, confidence)
                # Re-fit on full data
                forecaster.fit(df_daily)
                # Override dates to continue from last known date
                raw = forecaster.predict(horizon, confidence)
                last_date = df_daily['ds'].iloc[-1]
                raw['ds'] = pd.date_range(
                    start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D'
                )
                forecast = raw

            else:  # Hybrid ML
                forecaster = HybridMLForecaster()
                metrics = forecaster.evaluate(train_features, test_features, target='y')
                forecaster.fit(df_features, target='y')
                forecast = forecaster.predict(horizon, confidence)

            st.session_state.forecast = forecast
            st.session_state.metrics = metrics
            st.session_state.model_trained = True

        except Exception as e:
            st.error(f"❌ Forecasting failed: {str(e)}")
            st.session_state.model_trained = False

# ─────────────────────────── VISUALIZATION ───────────────────────────
if st.session_state.get('model_trained', False):
    forecast = st.session_state.forecast
    metrics = st.session_state.metrics

    fig = go.Figure()

    # Historical price
    fig.add_trace(go.Scatter(
        x=df_daily['ds'], y=df_daily['y'],
        mode='lines', name='Historical Price',
        line=dict(color='#3b82f6', width=2),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>',
    ))

    # 7-day moving average
    if show_ma:
        ma7 = df_daily['y'].rolling(7).mean()
        fig.add_trace(go.Scatter(
            x=df_daily['ds'], y=ma7,
            mode='lines', name='7-day MA',
            line=dict(color='#f59e0b', width=1.5, dash='dot'),
            hovertemplate='MA7: $%{y:.2f}<extra></extra>',
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', name='Forecast',
        line=dict(color='#ef4444', width=3),
        hovertemplate='Forecast: $%{y:.2f}<extra></extra>',
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'][::-1].tolist(),
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(239, 68, 68, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        name=f"{int(confidence * 100)}% Confidence",
        showlegend=True,
    ))

    # Forecast start marker (manual trace — avoids Plotly add_vline date bug)
    forecast_start = str(df_daily["ds"].iloc[-1].date())
    y_min = float(df_daily["y"].min())
    y_max = float(df_daily["y"].max())
    fig.add_trace(go.Scatter(
        x=[forecast_start, forecast_start],
        y=[y_min, y_max],
        mode="lines+text",
        line=dict(color="#64748b", width=2, dash="dash"),
        text=["", "Forecast Start"],
        textposition="top right",
        textfont=dict(size=12, color="#64748b"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        template='plotly_white',
        height=550,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            bgcolor='rgba(255,255,255,0.8)', bordercolor='#e2e8f0', borderwidth=1,
        ),
        font=dict(family='Inter, sans-serif', color='#1e293b'),
        paper_bgcolor='#f8fafc',
        plot_bgcolor='#ffffff',
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', tickprefix='$')

    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics ──
    st.markdown("### 📈 Backtest Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:,.2f}")
    with col2:
        st.metric("Root Mean Squared Error (RMSE)", f"${metrics['RMSE']:,.2f}")
    with col3:
        st.metric("Last Known Price", f"${df_daily['y'].iloc[-1]:,.2f}")

    # ── Forecast table ──
    with st.expander("📋 Forecast Data Table"):
        forecast_display = forecast.copy()
        forecast_display['ds'] = forecast_display['ds'].dt.date
        forecast_display = forecast_display.rename(columns={
            'ds': 'Date', 'yhat': 'Forecast',
            'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound',
        })
        st.dataframe(
            forecast_display.style.format({
                'Forecast': '${:,.2f}',
                'Lower Bound': '${:,.2f}',
                'Upper Bound': '${:,.2f}',
            }),
            use_container_width=True,
            hide_index=True,
        )
        csv = forecast_display.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Forecast CSV",
            data=csv,
            file_name=f"btc_forecast_{horizon}d.csv",
            mime="text/csv",
        )

else:
    st.info("👈 Configure settings in the sidebar and click **Generate Forecast** to see predictions.")
    fig_preview = px.line(df_daily, x='ds', y='y', title='Historical Bitcoin Prices (Preview)')
    fig_preview.update_layout(
        template='plotly_white', height=400,
        paper_bgcolor='#f8fafc', plot_bgcolor='#ffffff',
    )
    fig_preview.update_xaxes(title='Date', gridcolor='#e2e8f0')
    fig_preview.update_yaxes(title='Price (USD)', gridcolor='#e2e8f0', tickprefix='$')
    st.plotly_chart(fig_preview, use_container_width=True)

# ─────────────────────────── FOOTER ───────────────────────────
st.markdown("---")
st.caption("Built with Streamlit · Prophet · ARIMA · XGBoost · Plotly")
