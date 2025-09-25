import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

# Adjust paths for Codespaces/local environment
RAW_DATA_DIR = "./data/raw"
PROCESSED_DATA_DIR = "./data/processed"
MODEL_DIR = "./models"

logger = logging.getLogger(__name__)

def add_indicators(df):
    """Add technical indicators to the DataFrame."""
    # Calculate basic indicators (expand as needed)
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['rsi'] = 50  # Placeholder: Implement RSI calculation
    df['atr'] = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()  # Simple ATR placeholder
    return df

def prepare_features(df):
    cols = ['rsi', 'ema_9', 'ema_21', 'macd', 'macd_signal', 'vwap', 'volume_ma_ratio',
            'engulfing_signal', 'hammer', 'hanging_man', 'doji', 'morning_star', 'evening_star',
            'stoch_k', 'stoch_d', 'cci', 'adx', 'atr', 'bb_pos', 'ema_spread', 'ret_1', 'ret_3', 'vol_10', 'trend_regime']
    feat = df.reindex(columns=cols).fillna(0).replace([np.inf, -np.inf], 0).clip(-1e6, 1e6)
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(feat), columns=feat.columns, index=feat.index)

def run_pipeline_for_list(tickers):
    """Reads CSVs → adds indicators → loads GPU-trained model → predicts."""
    results = []
    for symbol in tickers:
        try:
            raw = {}
            for tf in ["5min", "15min", "1h", "1d"]:
                path = os.path.join(RAW_DATA_DIR, f"{symbol.replace('/', '-')}_{tf}.csv")
                raw[tf] = pd.read_csv(path, parse_dates=["datetime"]) if os.path.exists(path) else pd.DataFrame()
            if all(df.empty for df in raw.values()): continue

            # Add indicators
            processed = {}
            for tf, df in raw.items():
                if not df.empty:
                    df = add_indicators(df)
                    processed[tf] = df
            if not processed: continue

            # Load model and predict
            preds = {}
            for tf in ["5min", "15min", "1h", "1d"]:
                model_path = os.path.join(MODEL_DIR, f"{tf}_model.pkl")
                if not os.path.exists(model_path):
                    preds[tf] = ("HOLD", 0.0)
                    continue
                try:
                    model = pd.read_pickle(model_path)
                    X = prepare_features(processed[tf].iloc[[-1]])
                    if X.empty or X.shape[1] != len(prepare_features.__annotations__.get('return', {}).get('columns', [])):
                        preds[tf] = ("HOLD", 0.0)
                        continue
                    pred = model.predict(X)[0]
                    prob = model.predict_proba(X)[0][pred] if hasattr(model, 'predict_proba') else 0.0
                    preds[tf] = (["BUY", "HOLD", "SELL"][pred], round(prob * 100, 1))
                except Exception as e:
                    logger.error(f"Model prediction failed for {tf}: {e}")
                    preds[tf] = ("HOLD", 0.0)

            # Ensure required columns exist, use defaults if missing
            default_values = {"close": 0.0, "atr": 0.0, "low": 0.0, "high": 0.0}
            tf_1d = processed.get("1d", pd.DataFrame(default_values))
            results.append({
                "symbol": symbol,
                "signal": preds.get("1d", ("HOLD", 0.0))[0],
                "confidence": preds.get("1d", ("HOLD", 0.0))[1],
                "entry": round(float(tf_1d["close"].iloc[-1]) if not tf_1d["close"].empty else 0.0, 2),
                "tp": round(float(tf_1d["close"].iloc[-1]) + 1.5 * float(tf_1d["atr"].iloc[-1]) if not tf_1d["atr"].empty else 0.0, 2),
                "sl": round(float(tf_1d["close"].iloc[-1]) - 1.5 * float(tf_1d["atr"].iloc[-1]) if not tf_1d["atr"].empty else 0.0, 2),
                "atr": round(float(tf_1d["atr"].iloc[-1]) if not tf_1d["atr"].empty else 0.0, 2),
                "kelly": round(float(tf_1d["close"].iloc[-1]) * 0.25 if not tf_1d["close"].empty else 0.0, 2),
                "s1": round(float(tf_1d["low"].tail(20).min()) if not tf_1d["low"].empty else 0.0, 2),
                "r1": round(float(tf_1d["high"].tail(20).max()) if not tf_1d["high"].empty else 0.0, 2),
            })
        except Exception as e:
            logger.error(f"Pipeline failed for {symbol}: {e}")
            results.append({"symbol": symbol, "signal": "ERROR", "confidence": 0.0, "entry": 0, "tp": 0, "sl": 0, "atr": 0, "kelly": 0, "s1": 0, "r1": 0})
    return pd.DataFrame(results)
