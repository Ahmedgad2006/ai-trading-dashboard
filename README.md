# 🤖 AI Trading Signal Dashboard  
&gt; Free GPU training (Google Colab) + beautiful web UI (Streamlit Cloud) + instant Telegram alerts

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/YOUR_USER/ai-trading-dashboard/main/app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

## 📊  What it does
1. **GPU train** Random-Forest models on **TA-Lib features** (RSI, MACD, ATR, candle patterns, sector breadth)  
2. **Predict** BUY / SELL / HOLD on **any ticker** (stocks, crypto, ETFs)  
3. **Auto-alert** via Telegram (rich card: price, TP/SL, Kelly size, S/R, patterns)  
4. **Dashboard** → public web UI → download CSV, KPI cards, confidence bars  

**All free tiers** – **no paid data, no server cost**.

## 🚀  Quick start (2 min)
1.  **Open the dashboard**  
    👉  https://share.streamlit.io/YOUR_USER/ai-trading-dashboard/main/app.py  
2.  Type any tickers → **Analyze** → download signals  
3.  **Telegram alerts** arrive automatically (bot stays in Colab)

## 🧠  How it works
| Step | Where | Free Limit |
|------|-------|------------|
| **GPU training** | Google Colab | T4 16 GB, 12 h/session |
| **Data fetch** | TwelveData / FMP / CryptoCompare | 800/day, 250/day, 10 k/day |
| **ML model** | Random-Forest (GPU) | scikit-learn |
| **Alerts** | Telegram bot | unlimited |
| **Dashboard** | Streamlit Community Cloud | 1 GB RAM, always-on |

## 🔧  Local / Docker (optional)
```bash
git clone https://github.com/YOUR_USER/ai-trading-dashboard.git
cd ai-trading-dashboard
pip install -r requirements.txt
streamlit run app.py
