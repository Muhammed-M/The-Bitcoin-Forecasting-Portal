# app.py

import os
os.environ["SKLEARN_DISABLE_ARROW"] = "1"


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Import custom modules
from data_preprocess import load_and_prepare_btc_data, create_ml_features
from models import ProphetForecaster, HybridMLForecaster

# ----------------------------- PAGE CONFIGURATION -----------------------------
st.set_page_config(
    page_title="Bitcoin Price Forecaster",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- CUSTOM CSS (Light Modern Theme) -----------------------------
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8fafc;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    h1 {
        color: #0f172a;
        font-weight: 700;
        letter-spacing: -0.02em;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #1e293b;
        font-weight: 600;
        letter-spacing: -0.01em;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #334155;
        font-weight: 500;
        font-size: 1.1rem;
    }
    
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    div[data-testid="metric-container"] label {
        color: #64748b !important;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 700;
        font-size: 2rem;
    }
    
    .stButton > button {
        background-color: #0f172a;
        color: white;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #1e293b;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-1px);
    }
    
    .stDataFrame {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .stFileUploader > div > div {
        border-radius: 12px;
        border: 2px dashed #cbd5e1;
        background-color: #ffffff;
    }
    
    .stSlider > div > div > div {
        background-color: #0f172a;
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    hr {
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------- SESSION STATE INITIALIZATION -----------------------------
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_daily' not in st.session_state:
    st.session_state.df_daily = None
if 'df_features' not in st.session_state:
    st.session_state.df_features = None

# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bitcoin--v1.png", width=60)
    st.title("₿ BTC Forecaster")
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📂 Upload Bitcoin CSV",
        type=["csv"],
        help="Upload a Kaggle-style Bitcoin historical CSV file (minute or daily)."
    )
    
    if uploaded_file is not None:
        # Default price column is 'Close' (user cannot change per request)
        price_col = 'Close'
        
        # Load and prepare data with caching
        @st.cache_data(show_spinner=False)
        def load_data(file, price_col):
            return load_and_prepare_btc_data(file, price_col)
        
        with st.spinner("Processing data..."):
            df_daily, error = load_data(uploaded_file, price_col)
        
        if error:
            st.error(f"❌ {error}")
            st.session_state.data_loaded = False
        else:
            st.session_state.df_daily = df_daily
            st.session_state.data_loaded = True
            st.success(f"✅ Data loaded: {len(df_daily):,} days")
            
            # Show data preview in expander
            with st.expander("🔍 Data Preview"):
                st.dataframe(df_daily.head(10), use_container_width=True)
                st.caption(f"Date range: {df_daily['ds'].min().date()} → {df_daily['ds'].max().date()}")
    else:
        st.info("👆 Upload a CSV file to begin")
        st.session_state.data_loaded = False
    
    st.markdown("---")
    
    # Forecasting controls (only enabled if data loaded)
    st.subheader("⚙️ Forecast Settings")
    
    model_choice = st.selectbox(
        "Model",
        ["Prophet", "Hybrid ML (ElasticNet + XGBoost)"],
        disabled=not st.session_state.data_loaded,
        help="Prophet: robust to seasonality. Hybrid ML: combines linear trend with gradient boosting."
    )
    
    horizon = st.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1,
        disabled=not st.session_state.data_loaded
    )
    
    # Fixed confidence interval (95%)
    confidence = 0.95
    
    # Optional: toggle for moving averages
    show_ma = st.checkbox("Show 7-day Moving Average", value=False, disabled=not st.session_state.data_loaded)
    
    st.markdown("---")
    
    # Generate button
    generate_btn = st.button(
        "🚀 Generate Forecast",
        type="primary",
        disabled=not st.session_state.data_loaded,
        use_container_width=True
    )

# ----------------------------- MAIN PANEL -----------------------------
st.title("Bitcoin Price Forecast")
st.markdown("Analyze historical trends and predict future prices with confidence intervals.")

if not st.session_state.data_loaded:
    # Show placeholder when no data
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 Upload your Bitcoin CSV file in the sidebar to start.")
    with st.expander("📋 Expected CSV Format"):
        st.markdown("""
        Your CSV should contain at least:
        - A timestamp column (e.g., `Open time`, `Date`, `Timestamp`)
        - Price columns: `Open`, `High`, `Low`, `Close` (at least one)
        """)
    st.stop()

# ----------------------------- DATA PREPARATION FOR FORECASTING -----------------------------
df_daily = st.session_state.df_daily.copy()

# Split data chronologically (last 20% for backtesting)
split_idx = int(len(df_daily) * 0.8)
df_train = df_daily.iloc[:split_idx].copy()
df_test = df_daily.iloc[split_idx:].copy()

# Create ML features if needed (cached)
@st.cache_data(show_spinner=False)
def get_ml_features(df):
    df_feat = create_ml_features(df.set_index('ds'))
    return df_feat

if "Hybrid ML" in model_choice:
    df_features = get_ml_features(df_daily)
    # Split features as well
    train_features = df_features.iloc[:split_idx].copy()
    test_features = df_features.iloc[split_idx:].copy()

# ----------------------------- FORECASTING LOGIC -----------------------------
if generate_btn:
    with st.spinner("Training model and generating forecast..."):
        metrics = {}
        forecast = None
        
        if model_choice == "Prophet":
            forecaster = ProphetForecaster()
            # Evaluate on test set
            metrics = forecaster.evaluate(df_train, df_test)
            # Fit on full data and forecast
            forecaster.fit(df_daily)
            forecast = forecaster.predict(horizon, confidence)
            
        else:  # Hybrid ML
            forecaster = HybridMLForecaster()
            # Evaluate on test features
            metrics = forecaster.evaluate(train_features, test_features, target='y')
            # Fit on full features and forecast
            forecaster.fit(df_features, target='y')
            forecast = forecaster.predict(horizon, confidence)
        
        # Store in session state for plotting
        st.session_state.forecast = forecast
        st.session_state.metrics = metrics
        st.session_state.model_trained = True

# ----------------------------- VISUALIZATION -----------------------------
if st.session_state.get('model_trained', False):
    forecast = st.session_state.forecast
    metrics = st.session_state.metrics
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Historical actual prices
    fig.add_trace(go.Scatter(
        x=df_daily['ds'],
        y=df_daily['y'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#3b82f6', width=2),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Optionally add moving average
    if show_ma:
        ma7 = df_daily['y'].rolling(7).mean()
        fig.add_trace(go.Scatter(
            x=df_daily['ds'],
            y=ma7,
            mode='lines',
            name='7-day MA',
            line=dict(color='#f59e0b', width=1.5, dash='dot'),
            hovertemplate='MA7: $%{y:.2f}<extra></extra>'
        ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#ef4444', width=3),
        hovertemplate='Forecast: $%{y:.2f}<extra></extra>'
    ))
    
    # Confidence interval (uncertainty zone)
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'][::-1].tolist(),
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(239, 68, 68, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        name=f'{int(confidence*100)}% Confidence',
        showlegend=True
    ))
    
    # Vertical line at forecast start
    forecast_start = df_daily['ds'].iloc[-1]
    fig.add_vline(
        x=forecast_start,
        line_width=2,
        line_dash="dash",
        line_color="#64748b",
        annotation_text="Forecast Start",
        annotation_position="top left",
        annotation_font_size=12,
        annotation_font_color="#64748b"
    )
    
    # Layout styling
    fig.update_layout(
        title=None,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        template='plotly_white',
        height=550,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e2e8f0',
            borderwidth=1
        ),
        font=dict(family='Inter, sans-serif', color='#1e293b'),
        paper_bgcolor='#f8fafc',
        plot_bgcolor='#ffffff',
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', tickprefix='$')
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics display
    st.markdown("### 📈 Backtest Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:,.2f}")
    with col2:
        st.metric("Root Mean Squared Error (RMSE)", f"${metrics['RMSE']:,.2f}")
    with col3:
        last_price = df_daily['y'].iloc[-1]
        st.metric("Last Price", f"${last_price:,.2f}")
    
    # Forecast table in expander
    with st.expander("📋 Forecast Data Table"):
        forecast_display = forecast.copy()
        forecast_display['ds'] = forecast_display['ds'].dt.date
        forecast_display = forecast_display.rename(columns={
            'ds': 'Date',
            'yhat': 'Forecast',
            'yhat_lower': 'Lower Bound',
            'yhat_upper': 'Upper Bound'
        })
        st.dataframe(
            forecast_display.style.format({
                'Forecast': '${:,.2f}',
                'Lower Bound': '${:,.2f}',
                'Upper Bound': '${:,.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        csv = forecast_display.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Forecast CSV",
            data=csv,
            file_name=f"btc_forecast_{horizon}d.csv",
            mime="text/csv"
        )

else:
    # Show placeholder before generation
    st.info("👈 Configure settings in the sidebar and click **Generate Forecast** to see predictions.")
    
    # Show a simple line chart of historical data as preview
    fig_preview = px.line(df_daily, x='ds', y='y', title='Historical Bitcoin Prices (Preview)')
    fig_preview.update_layout(
        template='plotly_white',
        height=400,
        paper_bgcolor='#f8fafc',
        plot_bgcolor='#ffffff'
    )
    fig_preview.update_xaxes(title='Date', gridcolor='#e2e8f0')
    fig_preview.update_yaxes(title='Price (USD)', gridcolor='#e2e8f0', tickprefix='$')
    st.plotly_chart(fig_preview, use_container_width=True)

# ----------------------------- FOOTER -----------------------------
st.markdown("---")
st.caption("Built with Streamlit · Prophet · XGBoost · Plotly")