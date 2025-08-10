import pandas as pd
import numpy as np
import os, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

DATA_DIR = "datasets"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_df(df):
    # Basic standardization: ensure Date index and numeric columns
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").dropna(subset=["Date"])
        df = df.set_index("Date")
    # normalize column names
    cols = {c.lower():c for c in df.columns}
    def find(name_options):
        for opt in name_options:
            if opt in cols:
                return cols[opt]
        return None
    open_col = find(["open"])
    high_col = find(["high"])
    low_col = find(["low"])
    close_col = find(["close","adj close","adjusted"])
    vol_col = find(["volume","vol"])
    if close_col is None:
        raise ValueError("No close column found")
    df_std = pd.DataFrame(index=df.index)
    df_std["Open"] = pd.to_numeric(df[open_col]) if open_col else np.nan
    df_std["High"] = pd.to_numeric(df[high_col]) if high_col else np.nan
    df_std["Low"] = pd.to_numeric(df[low_col]) if low_col else np.nan
    df_std["Close"] = pd.to_numeric(df[close_col])
    df_std["Volume"] = pd.to_numeric(df[vol_col]) if vol_col else 0
    df_std = df_std.dropna(subset=["Close"])
    return df_std

def train_for_file(path):
    print("Training:", path)
    df = pd.read_csv(path)
    df = prepare_df(df)
    # simple features
    df["target"] = df["Close"].shift(-1)
    df = df.dropna()
    X = df[["Open","High","Low","Close","Volume"]]
    y = df["target"]
    if len(df) < 50:
        print("Too few rows, skipping:", path)
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Model saved. R2 score: {score:.4f}")
    model_name = os.path.splitext(os.path.basename(path))[0] + "_model.pkl"
    joblib.dump(model, os.path.join(MODEL_DIR, model_name))

if __name__ == '__main__':
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if not files:
        print("No CSV files found in datasets/. Put your CSV files there and re-run.")
    for f in files:
        train_for_file(os.path.join(DATA_DIR, f))
