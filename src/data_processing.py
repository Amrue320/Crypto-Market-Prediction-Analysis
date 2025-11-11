import pandas as pd
import numpy as np
import os
import yfinance as yf

def download_data(ticker='BTC-USD', start='2018-01-01', end=None):
    """Fallback: download Bitcoin data if no custom dataset exists"""
    print(f"📥 Downloading {ticker} data from Yahoo Finance...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.reset_index(inplace=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_data.csv", index=False)
    print(f"✅ Data saved to data/raw_data.csv ({len(df)} rows)")
    return df


def load_custom_data(path="data/custom_dataset.csv"):
    """Load and clean custom crypto dataset."""
    print(f"📂 Loading custom dataset from {path} ...")

    df = pd.read_csv(path)
    print(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    # --- Normalize column names ---
    df.columns = [str(col).strip().lower() for col in df.columns]

    # --- Rename common variations ---
    rename_map = {
        "timestamp": "Date",
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "price": "Close",
        "adj close": "Close",
        "volume": "Volume",
        "vol": "Volume"
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # --- Handle timestamp/date conversion ---
    if "Date" in df.columns:
        if np.issubdtype(df["Date"].dtype, np.number):
            df["Date"] = pd.to_datetime(df["Date"], unit="s", errors="coerce")
        else:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(subset=["Date"], inplace=True)
    else:
        raise ValueError("❌ Dataset must contain a 'timestamp' or 'date' column!")

    # --- Ensure numeric columns exist ---
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            print(f"⚠️ Missing expected column: {col}. Filling with 0s.")
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Sort by Date ---
    df = df.sort_values("Date").reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/cleaned_data.csv", index=False)
    print("✅ Data cleaned and saved to data/cleaned_data.csv")

    return df

