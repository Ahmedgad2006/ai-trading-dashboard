import os, pandas as pd, asyncio, logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

RAW_DATA_DIR = "/content/drive/MyDrive/trading_ai/data/raw"   # same as Colab
PROCESSED_DATA_DIR = "/content/drive/MyDrive/trading_ai/data/processed"
MODEL_DIR = "/content/drive/MyDrive/trading_ai/models"

logger = logging.getLogger(__name__)

def load_model(tf):
    path = os.path.join(MODEL_DIR, f"{tf}_model.pkl")
    return pd.read_pickle(path) if os.path.exists(path) else None

def prepare_features(df):
    cols = ['rsi','ema_9','ema_21','macd','macd_signal','vwap','volume_ma_ratio',
            'engulfing_signal','hammer','hanging_man','doji','morning_star','evening_star',
            'stoch_k','stoch_d','cci','adx','atr','bb_pos','ema_spread','ret_1','ret_3','vol_10','trend_regime']
    feat = df.reindex(columns=cols).fillna(0).replace([np.inf, -np.inf], 0).clip(-1e6, 1e6)
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(feat), columns=feat.columns, index=feat.index)

def run_pipeline_for_list(tickers):
    """Lightweight inference – reads CSVs + predicts (no training)."""
    results = []
    for symbol in tickers:
        try:
            raw = {}
            for tf in ["5min", "15min", "1h", "1d"]:
                path = os.path.join(RAW_DATA_DIR, f"{symbol.replace('/', '-')}_{tf}.csv")
                raw[tf] = pd.read_csv(path, parse_dates=["datetime"]) if os.path.exists(path) else pd.DataFrame()
            if all(df.empty for df in raw.values()): continue

            # add indicators (reuse your Cell-4 logic)
            from main_indicators import add_indicators   # inline below
            processed = {}
            for tf, df in raw.items():
                if not df.empty:
                    df = add_indicators(df)
                    processed[tf] = df
            if not processed: continue

            # load model & predict
            preds = {}
            for tf, df in processed.items():
                model = load_model(tf)
                if model is None: continue
                X = prepare_features(df.iloc[[-1]])
                if X.empty: continue
                pred = model.predict(X)[0]
                prob = model.predict_proba(X)[0][pred]
                preds[tf] = (["BUY", "HOLD", "SELL"][pred], round(prob * 100, 1))
            if preds:
                results.append({"symbol": symbol, **{f"{tf}": preds.get(tf, ("HOLD", 0.0))[0] for tf in ["5min", "15min", "1h", "1d"]}})
        except Exception as e:
            results.append({"symbol": symbol, "5min": "ERROR", "15min": "ERROR", "1h": "ERROR", "1d": "ERROR"})
    return pd.DataFrame(results)
