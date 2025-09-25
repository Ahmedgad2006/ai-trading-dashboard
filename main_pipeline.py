import os, pandas as pd, logging
from sklearn.preprocessing import StandardScaler

RAW_DATA_DIR   = "/content/drive/MyDrive/trading_ai/data/raw"
PROCESSED_DATA_DIR = "/content/drive/MyDrive/trading_ai/data/processed"
MODEL_DIR      = "/content/drive/MyDrive/trading_ai/models"

logger = logging.getLogger(__name__)

def prepare_features(df):
    cols = ['rsi','ema_9','ema_21','macd','macd_signal','vwap','volume_ma_ratio',
            'engulfing_signal','hammer','hanging_man','doji','morning_star','evening_star',
            'stoch_k','stoch_d','cci','adx','atr','bb_pos','ema_spread','ret_1','ret_3','vol_10','trend_regime']
    feat = df.reindex(columns=cols).fillna(0).replace([pd.np.inf, -pd.np.inf], 0).clip(-1e6, 1e6)
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

            # ---- add indicators (reuse your Cell-4 logic) ----
            processed = {}
            for tf, df in raw.items():
                if not df.empty:
                    df = add_indicators(df)   # inline below
                    processed[tf] = df
            if not processed: continue

            # ---- load GPU-trained model → predict ----
            preds = {}
            for tf in ["5min", "15min", "1h", "1d"]:
                model_path = os.path.join(MODEL_DIR, f"{tf}_model.pkl")
                if not os.path.exists(model_path):
                    preds[tf] = ("HOLD", 0.0); continue
                model = pd.read_pickle(model_path)
                X = prepare_features(processed[tf].iloc[[-1]])
                if X.empty: preds[tf] = ("HOLD", 0.0); continue
                pred = model.predict(X)[0]
                prob = model.predict_proba(X)[0][pred]
                preds[tf] = (["BUY", "HOLD", "SELL"][pred], round(prob * 100, 1))

            # ---- same shape as fake_signals ----
            results.append({
                "symbol": symbol,
                "signal": preds.get("1d", ("HOLD", 0.0))[0],
                "confidence": preds.get("1d", ("HOLD", 0.0))[1],
                "entry": round(float(processed["1d"]["close"].iloc[-1]), 2),
                "tp": round(float(processed["1d"]["close"].iloc[-1]) + 1.5 * float(processed["1d"]["atr"].iloc[-1]), 2),
                "sl": round(float(processed["1d"]["close"].iloc[-1]) - 1.5 * float(processed["1d"]["atr"].iloc[-1]), 2),
                "atr": round(float(processed["1d"]["atr"].iloc[-1]), 2),
                "kelly": round(float(processed["1d"]["close"].iloc[-1]) * 0.25, 2),
                "s1": round(float(processed["1d"]["low"].tail(20).min()), 2),
                "r1": round(float(processed["1d"]["high"].tail(20).max()), 2),
            })
        except Exception as e:
            results.append({"symbol": symbol, "signal": "ERROR", "confidence": 0.0, "entry": 0, "tp": 0, "sl": 0, "atr": 0, "kelly": 0, "s1": 0, "r1": 0})
    return pd.DataFrame(results)
