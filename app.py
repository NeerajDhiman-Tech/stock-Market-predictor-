import streamlit as st
import pandas as pd, joblib, os
from pathlib import Path

st.set_page_config(page_title="Stock Predictor Demo", layout="wide")
st.title("Stock Predictor — Demo")

DATA_DIR = Path("datasets")
MODEL_DIR = Path("models")

uploaded = st.file_uploader("Upload CSV with Date,Open,High,Low,Close,Volume", type="csv")
model_files = sorted([p.name for p in MODEL_DIR.glob("*.pkl")])
model_choice = st.selectbox("Choose model to use (train models first):", ["(none)"] + model_files)

def load_and_std(df):
    # same standardization as train script
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").dropna(subset=["Date"]).set_index("Date")
    lower = {c.lower():c for c in df.columns}
    def find(opts):
        for o in opts:
            if o in lower: return lower[o]
        return None
    openc = find(["open"]); highc = find(["high"]); lowc = find(["low"]); closec = find(["close","adj close","adjusted"]); volc = find(["volume","vol"])
    if closec is None:
        st.error("No Close column found in uploaded CSV")
        return None
    std = pd.DataFrame(index=df.index)
    std["Open"] = pd.to_numeric(df[openc]) if openc else None
    std["High"] = pd.to_numeric(df[highc]) if highc else None
    std["Low"] = pd.to_numeric(df[lowc]) if lowc else None
    std["Close"] = pd.to_numeric(df[closec])
    std["Volume"] = pd.to_numeric(df[volc]) if volc else 0
    return std.dropna(subset=["Close"])

if uploaded is not None and model_choice != "(none)":
    df = pd.read_csv(uploaded)
    std = load_and_std(df)
    if std is None:
        st.stop()
    model = joblib.load(MODEL_DIR / model_choice)
    last_row = std[["Open","High","Low","Close","Volume"]].iloc[-1:]
    pred = model.predict(last_row)[0]
    st.metric("Predicted next-day close", f"{pred:.2f}")
    st.line_chart(std["Close"].tail(200))
else:
    st.info("Upload a CSV and select a trained model. If you haven't trained models yet, run train_model.py.")