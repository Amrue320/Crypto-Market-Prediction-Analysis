# =====================================
# src/model_training.py — Final Version
# =====================================
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Import project modules
from src.api_fetch import fetch_live_data
from src.feature_engineering import create_features

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(symbol="BTCUSDT", interval="1d", lookback="1 year ago UTC"):
    print(f"📥 Fetching {symbol} data from Binance...")
    df = fetch_live_data(symbol, interval, lookback)
    print(f"✅ Data fetched: {len(df)} rows")

    print("🔧 Generating features...")
    df_feat = create_features(df).dropna()

    # Prepare training data
    X = df_feat.select_dtypes("number").drop(columns=["Close"], errors="ignore")
    y = df_feat["Close"].values.reshape(-1, 1)

    print(f"📊 Training on {X.shape[0]} samples with {X.shape[1]} features...")

    # Scale features and target
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    # Train model
    print("🤖 Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=15
    )
    model.fit(X_scaled, y_scaled.ravel())

    # Save artifacts
    joblib.dump(model, os.path.join(MODEL_DIR, "trained_model.pkl"))
    joblib.dump(X_scaler, os.path.join(MODEL_DIR, "X_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(MODEL_DIR, "y_scaler.pkl"))
    print("✅ Model and scalers saved successfully.")

    print(f"📦 Model path: {MODEL_DIR}/trained_model.pkl")
    print("🎉 Training complete!")

if __name__ == "__main__":
    train_model()





