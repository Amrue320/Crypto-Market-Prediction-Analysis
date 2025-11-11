import streamlit as st
import altair as alt
import pandas as pd

def plot_predictions(df):
    """Plot Actual vs Predicted Close prices."""
    df = df.copy()
    df.columns = df.columns.map(str)  # Ensure simple column names

    if "Date" not in df.columns or "Close" not in df.columns or "pred" not in df.columns:
        st.warning("⚠️ Missing columns for prediction plot.")
        return

    chart = alt.Chart(df).mark_line(color="steelblue").encode(
        x="Date:T",
        y=alt.Y("Close:Q", title="Actual Price (USD)"),
        tooltip=["Date", "Close"]
    )

    pred_line = alt.Chart(df).mark_line(color="red").encode(
        x="Date:T",
        y=alt.Y("pred:Q", title="Predicted Price (USD)"),
        tooltip=["Date", "pred"]
    )

    st.altair_chart(chart + pred_line, use_container_width=True)

def plot_cumulative(df):
    """Plot cumulative return of actual prices."""
    df = df.copy()
    df.columns = df.columns.map(str)
    if "Date" not in df.columns or "Close" not in df.columns:
        st.warning("⚠️ Missing columns for cumulative chart.")
        return

    df["Cumulative_Return"] = (df["Close"] / df["Close"].iloc[0]) - 1
    st.line_chart(df.set_index("Date")[["Cumulative_Return"]], use_container_width=True)

def plot_indicators(df):
    """Plot technical indicators (EMA, RSI, MACD)."""
    df = df.copy()
    df.columns = df.columns.map(str)  # ✅ Force single-level columns

    if "Date" not in df.columns:
        st.warning("⚠️ Missing Date column for indicator plot.")
        return

    indicators = ["EMA_10", "EMA_30", "EMA_50", "RSI", "MACD"]
    available = [col for col in indicators if col in df.columns]

    if not available:
        st.warning("⚠️ No indicators available for plotting.")
        return

    # ✅ Reset index to avoid multi-index or tuple errors
    df_reset = df.reset_index(drop=True)
    df_reset["Date"] = pd.to_datetime(df_reset["Date"], errors="coerce")

    try:
        st.line_chart(df_reset.set_index("Date")[available], use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error plotting indicators: {e}")






