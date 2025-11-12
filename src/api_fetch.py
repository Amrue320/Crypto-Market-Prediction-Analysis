# =====================================
# src/api_fetch.py — Fixed CoinGecko Fetch (no 401, works globally)
# =====================================

import pandas as pd
import requests
import streamlit as st

# Map Binance-style tickers to CoinGecko coin IDs
COIN_MAP = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "BNBUSDT": "binancecoin",
    "DOGEUSDT": "dogecoin",
    "SOLUSDT": "solana"
}

def fetch_live_data(symbol="BTCUSDT", interval="daily", lookback="1 year ago UTC"):
    """
    Fetch OHLCV-like data for a cryptocurrency from CoinGecko.
    Example: fetch_live_data("BTCUSDT", "daily", "1 year ago UTC")
    """

    # Convert Binance-like symbol to CoinGecko ID
    coin_id = COIN_MAP.get(symbol.upper(), "bitcoin")

    # Convert human-readable lookback to number of days
    lookback_days = 365
    lookback = str(lookback).lower()
    if "month" in lookback:
        lookback_days = 30
    elif "3 month" in lookback or "90" in lookback:
        lookback_days = 90
    elif "6 month" in lookback or "180" in lookback:
        lookback_days = 180
    elif "week" in lookback:
        lookback_days = 7
    elif "2 year" in lookback:
        lookback_days = 730
    elif "3 year" in lookback:
        lookback_days = 1095

    try:
        # ✅ Correct CoinGecko endpoint
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"

        # Add headers to avoid 401 Unauthorized error
        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        params = {
            "vs_currency": "usd",
            "days": lookback_days,
            "interval": "daily"
        }

        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        if not prices:
            st.error(f"❌ No price data found for {symbol}.")
            return pd.DataFrame()

        df = pd.DataFrame(prices, columns=["Timestamp", "Close"])
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")

        if volumes:
            vol_df = pd.DataFrame(volumes, columns=["Timestamp", "Volume"])
            df = df.merge(vol_df, on="Timestamp", how="left")
        else:
            df["Volume"] = 0.0

        # Create pseudo OHLC for model compatibility
        df["Open"] = df["Close"].shift(1)
        df["High"] = df["Close"].rolling(window=2).max()
        df["Low"] = df["Close"].rolling(window=2).min()

        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()

        if df.empty:
            st.warning(f"⚠️ No rows of {symbol} data loaded from CoinGecko.")
        else:
            st.success(f"✅ Loaded {len(df)} rows of {symbol} data.")

        return df

    except Exception as e:
        st.error(f"❌ Failed to fetch live data for {symbol}: {e}")
        return pd.DataFrame()






