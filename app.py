# =====================================
# app.py — FINAL STABLE LIVE VERSION (CoinGecko)
# =====================================

# -------------------------------
# Imports & Path Setup
# -------------------------------
import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# ✅ Must be first Streamlit command
st.set_page_config(
    page_title="📊 Real-Time Crypto Market Prediction Dashboard",
    layout="wide"
)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from src.api_fetch import fetch_live_data
from src.feature_engineering import create_features
from src.visualization import plot_predictions, plot_cumulative, plot_indicators

# -------------------------------
# Page Header
# -------------------------------
st.title("📊 Real-Time Crypto Market Prediction Dashboard")
st.write("""
Predict cryptocurrency prices using **live CoinGecko data** and a trained machine learning model 
enhanced with technical indicators (**EMA**, **RSI**, **MACD**, **Bollinger Bands**, etc.).
""")

# -------------------------------
# Load Model & Scalers
# -------------------------------
with st.spinner("Loading trained model and scalers..."):
    try:
        model = joblib.load(os.path.join(project_root, "models", "trained_model.pkl"))
        X_scaler = joblib.load(os.path.join(project_root, "models", "X_scaler.pkl"))
        y_scaler = joblib.load(os.path.join(project_root, "models", "y_scaler.pkl"))
        st.success("✅ Model and scalers loaded successfully.")
    except Exception as e:
        st.error(f"❌ Could not load model or scalers: {e}")
        st.stop()

# -------------------------------
# Helper: Feature Alignment
# -------------------------------
def align_features(df_feat, model):
    """Ensure features match model training columns."""
    X = df_feat.select_dtypes(include=[np.number]).fillna(0)
    if hasattr(model, "feature_names_in_"):
        model_feats = list(model.feature_names_in_)
        aligned = pd.DataFrame(index=X.index)
        for f in model_feats:
            aligned[f] = X[f] if f in X.columns else 0
        return aligned[model_feats]
    return X

# -------------------------------
# Prepare Features
# -------------------------------
def prepare_features(df, model):
    """Compute indicators and align features correctly."""
    df_feat = create_features(df.copy())
    df_feat.columns = df_feat.columns.map(str)

    # Drop columns not used in model
    removed = [c for c in ["Close", "Close_raw", "pred"] if c in df_feat.columns]
    if removed:
        st.info(f"⚙️ Dropped unused columns before prediction: {removed}")

    df_feat = df_feat.drop(columns=[c for c in ["Close", "Close_raw", "pred"] if c in df_feat.columns])
    df_aligned = df.tail(len(df_feat)).copy()
    X = align_features(df_feat, model)
    return df_aligned, X

# -------------------------------
# Forecast Function
# -------------------------------
def forecast_future(df, model, X_scaler, y_scaler, days=7):
    """Forecast future prices using dynamic volatility."""
    if df.empty:
        st.error("❌ No data available for forecasting.")
        return pd.DataFrame()

    df_copy = df.copy().reset_index(drop=True)
    forecasts = []

    # Estimate volatility
    volatility = df_copy["Close"].pct_change().rolling(10).std().iloc[-1]
    if np.isnan(volatility) or volatility == 0:
        volatility = 0.005  # fallback ±0.5%

    for _ in range(days):
        df_features = create_features(df_copy.copy())

        for col in ["Close", "pred", "Close_raw"]:
            if col in df_features.columns:
                df_features = df_features.drop(columns=[col])

        X = align_features(df_features, model)
        X_scaled = X_scaler.transform(X)
        scaled_pred = model.predict(X_scaled)
        pred = y_scaler.inverse_transform(scaled_pred.reshape(-1, 1)).flatten()[0]

        fluctuation = np.random.normal(0, volatility)
        pred = pred * (1 + fluctuation)

        next_date = df_copy["Date"].iloc[-1] + timedelta(days=1)
        forecasts.append({"Date": next_date, "Predicted_Close": pred})

        new_row = {
            "Date": next_date,
            "Open": df_copy["Close"].iloc[-1],
            "High": pred * (1 + abs(fluctuation / 2)),
            "Low": pred * (1 - abs(fluctuation / 2)),
            "Close": pred,
            "Volume": df_copy["Volume"].iloc[-1],
        }
        df_copy = pd.concat([df_copy, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(forecasts)

# -------------------------------
# Fetch Live Data (CoinGecko)
# -------------------------------
st.subheader("📡 Fetch Live Crypto Data (from CoinGecko)")

symbol = st.text_input("Enter symbol (e.g., BTCUSDT, ETHUSDT):", "BTCUSDT")
interval = st.selectbox("Interval:", ["daily"], index=0)
lookback = st.selectbox("Lookback:", ["3 months ago UTC", "6 months ago UTC", "1 year ago UTC"], index=2)

if st.button("🔄 Fetch Live Data"):
    with st.spinner("Fetching live market data..."):
        try:
            df = fetch_live_data(symbol, interval, lookback)
            if df.empty:
                st.error(f"⚠️ No live data found for {symbol}. Try another symbol.")
            else:
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df)} rows of {symbol} data from CoinGecko.")
                st.dataframe(df.tail())
        except Exception as e:
            st.error(f"❌ Failed to fetch live data: {e}")

# -------------------------------
# Predictions & Visualization
# -------------------------------
df_aligned = None

if "df" in st.session_state and st.session_state.df is not None:
    df = st.session_state.df.copy()

    # --- Feature Preparation ---
    try:
        df_aligned, X = prepare_features(df, model)
        if df_aligned is None or df_aligned.empty:
            st.warning("⚠️ Not enough data for prediction. Try a longer lookback.")
            st.stop()
        st.success(f"✅ Features computed and aligned ({len(df_aligned)} rows).")
    except Exception as e:
        st.error(f"❌ Feature preparation failed: {e}")
        st.stop()

    # --- Predictions ---
    try:
        X_scaled = X_scaler.transform(X)
        preds = model.predict(X_scaled)
        preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        df_aligned["pred"] = preds

        st.subheader("🤖 Predictions on Live Data")
        st.dataframe(df_aligned[["Date", "Close", "pred"]].tail(10))
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # --- Visuals ---
    st.subheader("📈 Visual Analysis")
    try:
        plot_predictions(df_aligned)
        plot_cumulative(df_aligned)

        df_indicators = create_features(df_aligned.copy())
        if any(ind in df_indicators.columns for ind in ["EMA_10", "EMA_30", "RSI", "MACD"]):
            plot_indicators(df_indicators)
        else:
            st.info("ℹ️ No technical indicators available for plotting. Try a longer lookback.")
    except Exception as e:
        st.warning(f"⚠️ Some charts could not be rendered: {e}")

    # --- Market Summary ---
    if not df_aligned.empty:
        latest_close = float(df_aligned["Close"].iloc[-1])
        latest_pred = float(df_aligned["pred"].iloc[-1])
        st.subheader("📊 Market Summary")
        st.markdown(f"**Last Actual Close:** ${latest_close:,.2f}")
        st.markdown(f"**Predicted Next Close:** ${latest_pred:,.2f}")

    # --- Forecast ---
    st.subheader("📅 Forecast Future Prices (Starting Tomorrow)")
    forecast_days = st.slider("Days to forecast:", 3, 14, 7)

    if st.button("🔮 Generate Forecast") and df_aligned is not None:
        with st.spinner("Generating forecast... please wait ⏳"):
            forecast_df = forecast_future(df_aligned, model, X_scaler, y_scaler, days=forecast_days)
            if forecast_df.empty:
                st.error("⚠️ Forecast not generated.")
            else:
                st.success(f"✅ Forecast generated for {len(forecast_df)} days.")
                st.dataframe(forecast_df)

                combined = pd.concat(
                    [
                        df_aligned[["Date", "Close"]].tail(60).rename(columns={"Close": "Predicted_Close"}),
                        forecast_df,
                    ],
                    ignore_index=True,
                )
                st.line_chart(combined.set_index("Date")["Predicted_Close"])
else:
    st.info("📂 Please fetch live data to begin.")


































