"""Random Forest classifier: will tomorrow's close be higher (1) or lower (0)?
Trained per-ticker on 2y of daily data. Saves to model.pkl."""

import subprocess
import sys

# auto-install scikit-learn if needed
try:
    import sklearn
except ImportError:
    print("scikit-learn not found — installing…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])

import joblib
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, classification_report,
)

# config
TICKERS      = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "JNJ", "WMT", "KO", "V"]
MODEL_FILE   = "model.pkl"
TRAIN_RATIO  = 0.80
RF_PARAMS    = dict(n_estimators=300, max_depth=6, min_samples_leaf=5,
                    random_state=42, n_jobs=-1)

FEATURE_COLS = [
    "ma10_ratio",       # MA10 / close
    "ma30_ratio",       # MA30 / close
    "ma50_ratio",       # MA50 / close
    "ma10_ma30",        # MA10 / MA30 (crossover proximity)
    "ma30_ma50",        # MA30 / MA50
    "rsi14",            # RSI 14-period
    "volume_change",    # % change in volume vs yesterday
    "volume_ma10",      # volume / 10-day avg volume
    "momentum10",       # raw 10-day price momentum
    "roc5",             # 5-day rate of change (%)
    "roc10",            # 10-day rate of change (%)
    "roc20",            # 20-day rate of change (%)
    "volatility10",     # 10-day rolling std of daily returns
]

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; ml-bot/1.0)"})


# data fetching

def fetch_daily(ticker: str) -> pd.DataFrame:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&range=2y"
    )
    resp = _session.get(url, timeout=15)
    resp.raise_for_status()
    data   = resp.json()
    result = data["chart"]["result"][0]
    ts     = result["timestamp"]
    quote  = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "Close":  quote["close"],
        "Volume": quote["volume"],
    }, index=pd.to_datetime(ts, unit="s", utc=True).tz_localize(None).normalize())
    df.dropna(subset=["Close"], inplace=True)
    return df


# feature engineering

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"]
    volume = df["Volume"].replace(0, np.nan)

    f = pd.DataFrame(index=df.index)

    # MA ratios — normalized so they're comparable across tickers and time
    ma10 = close.rolling(10).mean()
    ma30 = close.rolling(30).mean()
    ma50 = close.rolling(50).mean()
    f["ma10_ratio"] = ma10 / close
    f["ma30_ratio"] = ma30 / close
    f["ma50_ratio"] = ma50 / close
    f["ma10_ma30"]  = ma10 / ma30
    f["ma30_ma50"]  = ma30 / ma50

    # RSI-14 (Wilder smoothing via EWM)
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, 1e-10)
    f["rsi14"] = 100 - (100 / (1 + rs))

    # volume
    f["volume_change"] = volume.pct_change()
    f["volume_ma10"]   = volume / volume.rolling(10).mean()

    # momentum and rate of change
    f["momentum10"] = close - close.shift(10)
    f["roc5"]       = close.pct_change(5)  * 100
    f["roc10"]      = close.pct_change(10) * 100
    f["roc20"]      = close.pct_change(20) * 100

    # volatility
    f["volatility10"] = close.pct_change().rolling(10).std()

    # target: 1 if tomorrow's close > today's
    f["target"] = (close.shift(-1) > close).astype(int)

    return f


# training

def train_ticker(ticker: str, raw: pd.DataFrame) -> tuple:
    feat = build_features(raw).dropna()

    X = feat[FEATURE_COLS]
    y = feat["target"]

    # chronological split — no shuffling, preserves time order
    split    = int(len(X) * TRAIN_RATIO)
    X_train  = X.iloc[:split]
    X_test   = X.iloc[split:]
    y_train  = y.iloc[:split]
    y_test   = y.iloc[split:]

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred, feat


# reporting

def print_ticker_report(ticker, X_train, X_test, y_test, y_pred, model):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)

    W = 58
    print(f"\n  ┌{'─'*W}┐")
    print(f"  │  {ticker:<56}│")
    print(f"  ├{'─'*W}┤")
    print(f"  │  {'Training samples':<28} {len(X_train):>6}  days{' ':>16}│")
    print(f"  │  {'Test samples':<28} {len(X_test):>6}  days{' ':>16}│")
    print(f"  │  {'Accuracy':<28} {acc:>8.2%}{' ':>18}│")
    print(f"  │  {'Precision':<28} {prec:>8.2%}{' ':>18}│")
    print(f"  │  {'Recall':<28} {rec:>8.2%}{' ':>18}│")
    print(f"  └{'─'*W}┘")

    print(f"\n  Classification Report — {ticker}")
    report = classification_report(
        y_test, y_pred,
        target_names=["DOWN (0)", "UP   (1)"],
        digits=3,
    )
    for line in report.splitlines():
        print(f"    {line}")

    # top 5 features
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    top5        = importances.sort_values(ascending=False).head(5)
    print(f"\n  Top-5 features — {ticker}")
    for feat_name, imp in top5.items():
        bar = "█" * int(imp * 40)
        print(f"    {feat_name:<20}  {imp:.4f}  {bar}")


# entry point

def main():
    W = 62
    print("\n" + "═" * W)
    print("  ML MODEL TRAINING  —  Random Forest Direction Classifier")
    print("═" * W)
    print(f"  Tickers  : {', '.join(TICKERS)}")
    print(f"  Features : {len(FEATURE_COLS)}")
    print(f"  Split    : {int(TRAIN_RATIO*100)}% train / {int((1-TRAIN_RATIO)*100)}% test  (chronological)")
    print(f"  Model    : RandomForest  trees={RF_PARAMS['n_estimators']}"
          f"  max_depth={RF_PARAMS['max_depth']}")
    print("═" * W)

    print("\nFetching 2 years of daily data…")
    raw_data: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        print(f"  {ticker}…", end=" ", flush=True)
        df = fetch_daily(ticker)
        raw_data[ticker] = df
        print(f"{len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")

    print("\nTraining models…")
    trained_models: dict = {"features": FEATURE_COLS}

    for ticker in TICKERS:
        print(f"\n  [{ticker}] building features & fitting…", end=" ", flush=True)
        model, X_train, X_test, y_train, y_test, y_pred, feat = train_ticker(
            ticker, raw_data[ticker]
        )
        trained_models[ticker] = model
        print("done")
        print_ticker_report(ticker, X_train, X_test, y_test, y_pred, model)

    # summary table
    print(f"\n\n{'═'*W}")
    print("  SUMMARY")
    print(f"{'═'*W}")
    print(f"  {'Ticker':<8} {'Train':>7} {'Test':>7} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    print(f"  {'─'*6:<8} {'─'*5:>7} {'─'*5:>7} {'─'*8:>10} {'─'*8:>10} {'─'*6:>10}")

    for ticker in TICKERS:
        model, X_train, X_test, y_train, y_test, y_pred, feat = train_ticker(
            ticker, raw_data[ticker]
        )
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        print(f"  {ticker:<8} {len(X_train):>7} {len(X_test):>7} {acc:>10.2%} {prec:>10.2%} {rec:>10.2%}")

    print(f"{'═'*W}")

    # save
    joblib.dump(trained_models, MODEL_FILE)
    print(f"\n  Model saved → {MODEL_FILE}")
    print(f"  Load in bot.py with:  models = joblib.load('{MODEL_FILE}')")
    print(f"  Predict with:         models['AAPL'].predict(features_df)\n")


if __name__ == "__main__":
    main()
