# =====================================
# src/feature_engineering.py
# =====================================
import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate technical indicators and features for crypto price prediction."""
    df = df.copy()
    df = df.sort_values("Date")

    # --- Basic features ---
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(window=10).std()

    # --- Moving Averages ---
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_30"] = df["Close"].ewm(span=30, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # --- RSI (Relative Strength Index) ---
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- MACD (Moving Average Convergence Divergence) ---
    short_ema = df["Close"].ewm(span=12, adjust=False).mean()
    long_ema = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema

    # --- Bollinger Bands (Upper & Lower) ---
    sma = df["Close"].rolling(window=20).mean()
    std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = sma + (2 * std)
    df["BB_Lower"] = sma - (2 * std)

    # Drop NaN rows after indicator calculations
    df = df.dropna().reset_index(drop=True)

    return df






