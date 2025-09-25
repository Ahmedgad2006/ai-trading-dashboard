import streamlit as st
import pandas as pd
import os, asyncio, logging, requests
from main_pipeline import run_pipeline_for_list   # reuse your Cell-7 logic

st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("📊 AI Trading Signal Dashboard")

# ---- user input ----
tickers = st.text_input("Enter tickers (comma or space):", "AMD,NVDA,SMH,BTC/USD").upper().split(",")
tickers = [t.strip() for t in tickers if t.strip()]

# ---- run analysis ----
if st.button("Analyze"):
    with st.spinner("Running AI pipeline …"):
        df = asyncio.run(run_pipeline_for_list(tickers))
    if df.empty:
        st.warning("No signals generated – check tickers or data.")
    else:
        st.success("Done!")
        st.dataframe(df)

        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "signals.csv", "text/csv")

# ---- last Telegram alert (optional) ----
if st.checkbox("Show last Telegram alert"):
    last = os.getenv("LAST_ALERT", "No alert yet")
    st.text(last)
