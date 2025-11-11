# =====================================
# src/api_fetch.py — Fetch Live Crypto Data
# =====================================
import pandas as pd
from binance.client import Client

def fetch_live_data(symbol="BTCUSDT", interval="1d", lookback="1 year ago UTC"):
    """
    Fetch OHLCV data from Binance API.
    Example: fetch_live_data("ETHUSDT", "1d", "1 year ago UTC")
    """
    client = Client()  # no API key required for public endpoints
    klines = client.get_historical_klines(symbol, interval, lookback)

    df = pd.DataFrame(klines, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])

    df["Date"] = pd.to_datetime(df["Open time"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    return df

