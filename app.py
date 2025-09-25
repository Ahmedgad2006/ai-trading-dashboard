import pandas as pd
import asyncio
import os
import logging
import streamlit as st
from main_pipeline import add_indicators, prepare_features  # Correct import

# Adjust paths for Codespaces/local environment
RAW_DATA_DIR = "./data/raw"
PROCESSED_DATA_DIR = "./data/processed"
MODEL_DIR = "./models"

logger = logging.getLogger(__name__)

def run_pipeline_for_list(tickers):
    """
    Reads CSVs → adds indicators → loads GPU-trained model → predicts.
    Returns DataFrame exactly like fake_signals().
    """
    results = []
    for symbol in tickers:
        try:
            raw = {}
            for tf in ["5min", "15min", "1h", "1d"]:
                path = os.path.join(RAW_DATA_DIR, f"{symbol.replace('/', '-')}_{tf}.csv")
                raw[tf] = pd.read_csv(path, parse_dates=["datetime"]) if os.path.exists(path) else pd.DataFrame()

            if all(df.empty for df in raw.values()):
                continue

            # ---- add indicators ----
            processed = {}
            for tf, df in raw.items():
                if not df.empty:
                    df = add_indicators(df)
                    processed[tf] = df
            if not processed:
                continue

            # ---- load GPU-trained model → predict ----
            preds = {}
            for tf in ["5min", "15min", "1h", "1d"]:
                model_path = os.path.join(MODEL_DIR, f"{tf}_model.pkl")
                if not os.path.exists(model_path):
                    preds[tf] = ("HOLD", 0.0)
                    continue
                model = pd.read_pickle(model_path)
                X = prepare_features(processed[tf].iloc[[-1]])
                if X.empty:
                    preds[tf] = ("HOLD", 0.0)
                    continue
                pred = model.predict(X)[0]
                prob = model.predict_proba(X)[0][pred]
                preds[tf] = (["BUY", "HOLD", "SELL"][pred], round(prob * 100, 1))

            # ---- same shape as fake_signals ----
            results.append({
                "symbol": symbol,
                "signal": preds.get("1d", ("HOLD", 0.0))[0],  # use 1-day as final vote
                "confidence": preds.get("1d", ("HOLD", 0.0))[1],
                "entry": round(float(processed["1d"]["close"].iloc[-1]), 2),
                "tp": round(float(processed["1d"]["close"].iloc[-1]) + 1.5 * float(processed["1d"]["atr"].iloc[-1]), 2),
                "sl": round(float(processed["1d"]["close"].iloc[-1]) - 1.5 * float(processed["1d"]["atr"].iloc[-1]), 2),
                "atr": round(float(processed["1d"]["atr"].iloc[-1]), 2),
                "kelly": round(float(processed["1d"]["close"].iloc[-1]) * 0.25, 2),  # placeholder
                "s1": round(float(processed["1d"]["low"].tail(20).min()), 2),
                "r1": round(float(processed["1d"]["high"].tail(20).max()), 2),
            })
        except Exception as e:
            results.append({
                "symbol": symbol, "signal": "ERROR", "confidence": 0.0,
                "entry": 0, "tp": 0, "sl": 0, "atr": 0, "kelly": 0, "s1": 0, "r1": 0
            })
    return pd.DataFrame(results)

def main():
    st.title("Trading Dashboard")
    tickers = st.text_input("Enter tickers (comma-separated)", "AAPL,GOOGL").split(",")
    if st.button("Run Pipeline"):
        with st.spinner("Processing..."):
            results = run_pipeline_for_list([t.strip() for t in tickers])
            st.write(results)

if __name__ == "__main__":
    main()
