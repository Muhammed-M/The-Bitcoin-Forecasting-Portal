# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO, BytesIO

from data_preprocess import load_and_prepare_btc_data, create_ml_features, detect_available_price_columns
from models import ProphetForecaster, HybridMLForecaster

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
    'available_price_cols': [],
    'last_price_col': None,
    'last_uploaded_name': None,
    'model_trained': False,
    'trained_model': None,
    'trained_model_choice': None,
    'trained_confidence': None,
    'train_metrics': None,
    'forecast_ready': False,
    'forecast': None,
    'forecast_horizon': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────── HELPERS ───────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name, price_col):
    return load_and_prepare_btc_data(BytesIO(file_bytes), price_col)

@st.cache_data(show_spinner=False)
def get_ml_features(df_json):
    df = pd.read_json(StringIO(df_json))
    df['ds'] = pd.to_datetime(df['ds'])
    return create_ml_features(df.set_index('ds'))

def make_chart(df_daily, forecast, confidence, show_ma):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_daily['ds'], y=df_daily['y'],
        mode='lines', name='Historical Price',
        line=dict(color='#3b82f6', width=1.8),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>',
    ))
    if show_ma:
        ma7 = df_daily['y'].rolling(7).mean()
        fig.add_trace(go.Scatter(
            x=df_daily['ds'], y=ma7,
            mode='lines', name='7-day MA',
            line=dict(color='#d97706', width=1.5, dash='dot'),
            hovertemplate='MA7: $%{y:.2f}<extra></extra>',
        ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', name='Forecast',
        line=dict(color='#fbbf24', width=2.5),
        hovertemplate='Forecast: $%{y:.2f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'][::-1].tolist(),
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
        fill='toself', fillcolor='rgba(251,191,36,0.07)',
        line=dict(color='rgba(251,191,36,0.15)'),
        hoverinfo='skip', name=f"{int(confidence * 100)}% Confidence", showlegend=True,
    ))
    forecast_start = str(df_daily["ds"].iloc[-1].date())
    y_min = float(df_daily["y"].min())
    y_max = float(df_daily["y"].max())
    fig.add_trace(go.Scatter(
        x=[forecast_start, forecast_start], y=[y_min, y_max],
        mode="lines+text",
        line=dict(color="rgba(251,191,36,0.4)", width=1.5, dash="dash"),
        text=["", "Forecast Start"], textposition="top right",
        textfont=dict(size=11, color="rgba(251,191,36,0.6)", family="DM Mono"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Price (USD)",
        hovermode='x unified', template='plotly_dark', height=560,
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            bgcolor='rgba(20,21,25,0.8)', bordercolor='rgba(251,191,36,0.2)',
            borderwidth=1, font=dict(color='#9a9a8e', size=12),
        ),
        font=dict(family='DM Mono, monospace', color='#9a9a8e', size=11),
        paper_bgcolor='#0d0e12', plot_bgcolor='#0f1015',
        hoverlabel=dict(
            bgcolor='#1a1b22', bordercolor='rgba(251,191,36,0.3)',
            font=dict(family='DM Mono, monospace', color='#e8e8e0', size=12),
        ),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.04)',
                     zeroline=False, tickfont=dict(color='#5a5a4e', size=10),
                     linecolor='rgba(255,255,255,0.06)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.04)',
                     zeroline=False, tickprefix='$', tickfont=dict(color='#5a5a4e', size=10),
                     linecolor='rgba(255,255,255,0.06)')
    return fig

# ─────────────────────────── SIDEBAR ───────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bitcoin--v1.png", width=52)
    st.markdown('<p class="sb-brand">BTC Forecaster</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── STEP 1: Data ──
    st.markdown('<p class="sb-step-label">① Data</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV", type=["csv"],
        help="Kaggle-style Bitcoin CSV (minute or daily).",
        label_visibility="collapsed",
    )

    price_col = 'Close'
    if uploaded_file is not None:
        file_name = uploaded_file.name
        if file_name != st.session_state.last_uploaded_name:
            available_cols, _ = detect_available_price_columns(uploaded_file)
            st.session_state.available_price_cols = available_cols
            st.session_state.last_uploaded_name = file_name
            st.session_state.data_loaded = False
            st.session_state.model_trained = False
            st.session_state.forecast_ready = False
            uploaded_file.seek(0)

        if st.session_state.available_price_cols:
            price_col = st.selectbox(
                "Price Column",
                options=st.session_state.available_price_cols,
                index=st.session_state.available_price_cols.index('Close')
                      if 'Close' in st.session_state.available_price_cols else 0,
                help="Which OHLC column to forecast.",
            )
        else:
            st.error("No price columns found (Open/High/Low/Close).")

    if uploaded_file is not None and st.session_state.available_price_cols:
        needs_reload = (
            not st.session_state.data_loaded
            or price_col != st.session_state.last_price_col
        )
        if needs_reload:
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            with st.spinner("Loading..."):
                df_daily, error = load_data(file_bytes, file_name, price_col)
            if error:
                st.error(f"❌ {error}")
                st.session_state.data_loaded = False
            else:
                st.session_state.df_daily = df_daily
                st.session_state.data_loaded = True
                st.session_state.last_price_col = price_col
                st.session_state.model_trained = False
                st.session_state.forecast_ready = False
                date_min = df_daily['ds'].min().strftime('%b %Y')
                date_max = df_daily['ds'].max().strftime('%b %Y')
                st.success(f"✅ {len(df_daily):,} days · {date_min} → {date_max}")
    elif uploaded_file is None:
        st.markdown('<p class="sb-hint">Upload a CSV to begin</p>', unsafe_allow_html=True)
        st.session_state.data_loaded = False

    st.markdown("---")

    # ── STEP 2: Train ──
    data_ready = st.session_state.data_loaded
    st.markdown('<p class="sb-step-label">② Train Model</p>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Algorithm",
        ["Prophet", "Hybrid ML (ElasticNet + XGBoost)"],
        disabled=not data_ready,
        help=(
            "**Prophet** – robust to seasonality and trend shifts.\n"
            "**Hybrid ML** – ElasticNet for trend + XGBoost for residuals."
        ),
    )

    confidence = st.select_slider(
        "Confidence Interval",
        options=[0.80, 0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x * 100)}%",
        disabled=not data_ready,
        help="Width of the uncertainty band.",
    )

    if st.session_state.model_trained:
        settings_changed = (
            model_choice != st.session_state.trained_model_choice or
            confidence != st.session_state.trained_confidence
        )
        if settings_changed:
            st.markdown('<p class="sb-warn">⚠ Settings changed — retrain model</p>', unsafe_allow_html=True)

    train_btn = st.button(
        "⚡ Train Model",
        disabled=not data_ready,
        use_container_width=True,
        type="primary",
    )

    st.markdown("---")

    # ── STEP 3: Forecast ──
    model_ready = st.session_state.model_trained
    st.markdown('<p class="sb-step-label">③ Forecast</p>', unsafe_allow_html=True)

    horizon = st.slider(
        "Horizon (days)",
        min_value=7, max_value=90, value=30, step=1,
        disabled=not model_ready,
    )

    show_ma = st.checkbox(
        "Show 7-day Moving Average",
        value=False, disabled=not model_ready,
    )

    if model_ready:
        st.markdown(
            f'<p class="sb-trained-info">✓ Trained · {st.session_state.trained_model_choice}</p>',
            unsafe_allow_html=True,
        )

    predict_btn = st.button(
        "📈 Generate Forecast",
        disabled=not model_ready,
        use_container_width=True,
    )

# ─────────────────────────── MAIN PANEL ───────────────────────────
st.markdown("""
<h1 class="main-title">Bitcoin Price Forecast</h1>
<p class="main-subtitle">Analyze historical trends · Predict future prices with confidence intervals</p>
""", unsafe_allow_html=True)

if not st.session_state.data_loaded:
    st.info("📊 Upload your Bitcoin CSV in the sidebar to begin.")
    with st.expander("Expected CSV Format"):
        st.markdown("""
- Date column: `Date`, `Open time`, `Timestamp`, `datetime`
- Price column: `Open`, `High`, `Low`, or `Close`
        """)
    st.stop()

df_daily = st.session_state.df_daily.copy()
split_idx = int(len(df_daily) * 0.8)
df_train = df_daily.iloc[:split_idx].copy()
df_test  = df_daily.iloc[split_idx:].copy()

# ─────────────────────────── TRAIN ───────────────────────────
if train_btn:
    with st.spinner(f"Training {model_choice}…"):
        try:
            if model_choice == "Prophet":
                forecaster = ProphetForecaster()
                metrics = forecaster.evaluate(df_train, df_test, confidence)
                forecaster.fit(df_daily, confidence)
            else:
                df_features = get_ml_features(df_daily.to_json())
                train_features = df_features.iloc[:split_idx].copy()
                test_features  = df_features.iloc[split_idx:].copy()
                forecaster = HybridMLForecaster()
                metrics = forecaster.evaluate(train_features, test_features, target='y')
                forecaster.fit(df_features, target='y')

            st.session_state.trained_model = forecaster
            st.session_state.trained_model_choice = model_choice
            st.session_state.trained_confidence = confidence
            st.session_state.train_metrics = metrics
            st.session_state.model_trained = True
            st.session_state.forecast_ready = False
            st.success(f"✅ Training complete · MAE: ${metrics['MAE']:,.0f} · RMSE: ${metrics['RMSE']:,.0f}")

        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            st.session_state.model_trained = False

# ─────────────────────────── PREDICT ───────────────────────────
if predict_btn and st.session_state.model_trained:
    with st.spinner("Generating forecast…"):
        try:
            forecast = st.session_state.trained_model.predict(
                horizon, st.session_state.trained_confidence
            )
            st.session_state.forecast = forecast
            st.session_state.forecast_horizon = horizon
            st.session_state.forecast_ready = True
        except Exception as e:
            st.error(f"❌ Forecast failed: {str(e)}")
            st.session_state.forecast_ready = False

# ─────────────────────────── VISUALIZATION ───────────────────────────
if st.session_state.forecast_ready:
    forecast   = st.session_state.forecast
    metrics    = st.session_state.train_metrics
    confidence = st.session_state.trained_confidence

    fig = make_chart(df_daily, forecast, confidence, show_ma)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Backtest Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE", f"${metrics['MAE']:,.2f}")
    with col2:
        st.metric("RMSE", f"${metrics['RMSE']:,.2f}")
    with col3:
        st.metric("Last Known Price", f"${df_daily['y'].iloc[-1]:,.2f}")

    with st.expander("Forecast Data Table"):
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
            use_container_width=True, hide_index=True,
        )
        st.download_button(
            label="⬇️ Download Forecast CSV",
            data=forecast_display.to_csv(index=False),
            file_name=f"btc_forecast_{st.session_state.forecast_horizon}d.csv",
            mime="text/csv",
        )
else:
    if not st.session_state.model_trained:
        st.info("👈 **Step ①** Train a model using the sidebar.")
    else:
        st.info("👈 **Step ③** Set the horizon and click Generate Forecast.")

    fig_preview = px.line(df_daily, x='ds', y='y')
    fig_preview.update_traces(line=dict(color='#3b82f6', width=1.8))
    fig_preview.update_layout(
        template='plotly_dark', height=420,
        paper_bgcolor='#0d0e12', plot_bgcolor='#0f1015',
        font=dict(family='DM Mono, monospace', color='#9a9a8e', size=11),
        hoverlabel=dict(bgcolor='#1a1b22', font=dict(color='#e8e8e0')),
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis_title="Date", yaxis_title="Price (USD)",
    )
    fig_preview.update_xaxes(gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#5a5a4e'))
    fig_preview.update_yaxes(gridcolor='rgba(255,255,255,0.04)', tickprefix='$', tickfont=dict(color='#5a5a4e'))
    st.plotly_chart(fig_preview, use_container_width=True)

# ─────────────────────────── FOOTER ───────────────────────────
st.markdown("---")
st.caption("Built with Streamlit · Prophet · XGBoost · Plotly")
