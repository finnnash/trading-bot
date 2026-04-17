"""
bot.py — Three-signal paper trading bot
Signals : MA Crossover  →  ML Model  →  RSS Sentiment
Trade executes only when ALL THREE agree.
"""

import json
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, time as dtime
import zoneinfo

import requests
import pandas as pd
import schedule
from textblob import TextBlob

# ── ML model (loaded once at startup) ─────────────────────────────────────────
try:
    import joblib
    from ml_model import build_features, FEATURE_COLS
    _ml_models = joblib.load("model.pkl")
    _ml_ready  = True
except Exception as _ml_err:
    _ml_models = {}
    _ml_ready  = False
    print(f"  [ML] WARNING: could not load model.pkl ({_ml_err}). ML signal disabled.")

# ── Config ─────────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "JNJ", "WMT", "KO", "V"]

STARTING_CASH    = 100_000.0
SHORT_WINDOW     = 10          # fast MA period
LONG_WINDOW      = 30          # slow MA period
MAX_POSITION_PCT = 0.20        # max 20% of portfolio per stock
CASH_BUFFER_PCT  = 0.05        # keep 5% cash reserve at all times

SENTIMENT_BUY_THRESHOLD  =  0.2
SENTIMENT_SELL_THRESHOLD = -0.2
SENTIMENT_CACHE_TTL      = 30 * 60   # 30 minutes

PORTFOLIO_FILE = "portfolio.json"
TRADES_FILE    = "trades.csv"

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; trading-bot/1.0)"})
_sentiment_cache: dict[str, tuple[float, int, float]] = {}  # {ticker: (score, n_headlines, ts)}


# ── Market hours ───────────────────────────────────────────────────────────────

def market_is_open() -> bool:
    et  = zoneinfo.ZoneInfo("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5:
        return False
    return dtime(9, 30) <= now.time() < dtime(16, 0)


# ── Portfolio ──────────────────────────────────────────────────────────────────

def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {"cash": STARTING_CASH, "positions": {}}


def save_portfolio(portfolio: dict) -> None:
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)


def portfolio_value(portfolio: dict, prices: dict) -> float:
    equity = sum(shares * prices.get(t, 0) for t, shares in portfolio["positions"].items())
    return portfolio["cash"] + equity


def max_spend(portfolio: dict, prices: dict) -> float:
    total = portfolio_value(portfolio, prices)
    return max(portfolio["cash"] - total * CASH_BUFFER_PCT, 0.0)


# ── Trade log ──────────────────────────────────────────────────────────────────

def log_trade(action: str, ticker: str, shares: float, price: float, reason: str) -> None:
    exists = os.path.exists(TRADES_FILE)
    with open(TRADES_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "action", "ticker", "shares", "price", "value", "reason"])
        w.writerow([datetime.now().isoformat(), action, ticker,
                    round(shares, 6), round(price, 4),
                    round(shares * price, 4), reason])


# ── Signal 1: MA Crossover (5-minute bars) ─────────────────────────────────────

def fetch_5m(ticker: str) -> pd.DataFrame | None:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=5m&range=5d"
    )
    try:
        resp = _session.get(url, timeout=10)
        resp.raise_for_status()
        result = resp.json()["chart"]["result"][0]
        closes = result["indicators"]["quote"][0]["close"]
        df = pd.DataFrame(
            {"Close": closes},
            index=pd.to_datetime(result["timestamp"], unit="s", utc=True),
        ).dropna()
        return df if len(df) >= LONG_WINDOW else None
    except Exception as e:
        print(f"  [{ticker}] 5m fetch error: {e}")
        return None


def ma_signal(df: pd.DataFrame) -> tuple[float, float, float, str]:
    """
    Returns (price, ma_short, ma_long, direction).
    direction: 'BUY' | 'SELL' | 'HOLD'
    """
    close    = df["Close"].squeeze()
    ma_s     = float(close.rolling(SHORT_WINDOW).mean().iloc[-1])
    ma_l     = float(close.rolling(LONG_WINDOW).mean().iloc[-1])
    price    = float(close.iloc[-1])
    prev_s   = float(close.rolling(SHORT_WINDOW).mean().iloc[-2])
    prev_l   = float(close.rolling(LONG_WINDOW).mean().iloc[-2])

    # Require an actual crossover on the last bar (not just above/below)
    crossed_up = (prev_s <= prev_l) and (ma_s > ma_l)
    crossed_dn = (prev_s >= prev_l) and (ma_s < ma_l)

    if crossed_up:
        direction = "BUY"
    elif crossed_dn:
        direction = "SELL"
    else:
        direction = "HOLD"

    return price, ma_s, ma_l, direction


# ── Signal 2: ML Model (daily bars) ───────────────────────────────────────────

def fetch_daily_for_ml(ticker: str) -> pd.DataFrame | None:
    """90 days of daily Close+Volume needed to compute ML features."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&range=6mo"
    )
    try:
        resp   = _session.get(url, timeout=10)
        resp.raise_for_status()
        result = resp.json()["chart"]["result"][0]
        quote  = result["indicators"]["quote"][0]
        df = pd.DataFrame(
            {"Close": quote["close"], "Volume": quote["volume"]},
            index=pd.to_datetime(result["timestamp"], unit="s", utc=True)
                  .tz_localize(None).normalize(),
        ).dropna(subset=["Close"])
        return df if len(df) >= 55 else None   # need ≥55 rows for MA50 feature
    except Exception as e:
        print(f"  [{ticker}] daily fetch error: {e}")
        return None


def ml_signal(ticker: str, df_daily: pd.DataFrame) -> tuple[int | None, float | None]:
    """
    Returns (prediction, probability_up).
    prediction: 1=UP, 0=DOWN, None=unavailable
    """
    if not _ml_ready or ticker not in _ml_models:
        return None, None
    try:
        feat   = build_features(df_daily)[FEATURE_COLS]
        latest = feat.iloc[[-1]]
        if latest.isna().any().any():
            return None, None
        pred  = int(_ml_models[ticker].predict(latest)[0])
        proba = _ml_models[ticker].predict_proba(latest)[0]
        prob_up = float(proba[1]) if len(proba) > 1 else None
        return pred, prob_up
    except Exception as e:
        print(f"  [{ticker}] ML error: {e}")
        return None, None


# ── Signal 3: RSS Sentiment (Yahoo Finance headlines + TextBlob) ────────────────

def rss_sentiment(ticker: str) -> tuple[float, int]:
    """
    Fetch Yahoo Finance RSS headlines, score with TextBlob.
    Returns (avg_polarity, n_headlines). Cached for 30 minutes.
    """
    now = time.time()
    if ticker in _sentiment_cache:
        score, n, ts = _sentiment_cache[ticker]
        if now - ts < SENTIMENT_CACHE_TTL:
            return score, n

    url = (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={ticker}&region=US&lang=en-US"
    )
    try:
        resp = _session.get(url, timeout=10)
        resp.raise_for_status()
        root  = ET.fromstring(resp.content)
        items = root.findall(".//item")
        if not items:
            _sentiment_cache[ticker] = (0.0, 0, now)
            return 0.0, 0
        scores = []
        for item in items:
            text  = (item.findtext("title") or "") + " " + (item.findtext("description") or "")
            scores.append(TextBlob(text.strip()).sentiment.polarity)
        avg = round(sum(scores) / len(scores), 4)
        _sentiment_cache[ticker] = (avg, len(scores), now)
        return avg, len(scores)
    except Exception as e:
        print(f"  [{ticker}] RSS error: {e}")
        _sentiment_cache[ticker] = (0.0, 0, now)
        return 0.0, 0


# ── Signal display helpers ─────────────────────────────────────────────────────

def _sent_label(score: float) -> str:
    if score >=  0.5: return "VERY POSITIVE"
    if score >=  0.2: return "positive"
    if score >  -0.2: return "neutral"
    if score >  -0.5: return "negative"
    return "VERY NEGATIVE"

def _ml_label(pred: int | None, prob: float | None) -> str:
    if pred is None:
        return "n/a"
    direction = "UP" if pred == 1 else "DOWN"
    prob_str  = f"  ({prob:.0%} confidence)" if prob is not None else ""
    return f"{direction}{prob_str}"

def _tick(ok: bool) -> str:
    return "✓" if ok else "✗"


# ── Core strategy loop ─────────────────────────────────────────────────────────

def run_strategy() -> None:
    W = 62
    print(f"\n{'═'*W}")
    print(f"  Trading cycle  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*W}")

    if not market_is_open():
        print("  Market is closed — next check in 5 minutes.")
        print(f"{'═'*W}\n")
        return

    portfolio      = load_portfolio()
    current_prices = {}

    # ── Fetch 5m data & compute MA signals for all tickers ────────────────────
    ma_data = {}   # ticker -> (price, ma_s, ma_l, direction, df_5m)
    for ticker in TICKERS:
        df = fetch_5m(ticker)
        if df is None:
            continue
        price, ma_s, ma_l, direction = ma_signal(df)
        current_prices[ticker] = price
        ma_data[ticker] = (price, ma_s, ma_l, direction, df)

    total = portfolio_value(portfolio, current_prices)

    # ── Process each ticker ───────────────────────────────────────────────────
    for ticker in TICKERS:
        if ticker not in ma_data:
            print(f"\n  {ticker:<6}  — no data")
            continue

        price, ma_s, ma_l, ma_dir, df_5m = ma_data[ticker]
        held_shares = portfolio["positions"].get(ticker, 0)
        held_value  = held_shares * price
        max_allowed = total * MAX_POSITION_PCT

        print(f"\n  {ticker}  ${price:.2f}  "
              f"MA{SHORT_WINDOW}={ma_s:.2f}  MA{LONG_WINDOW}={ma_l:.2f}  "
              f"held={held_shares} sh")
        print(f"  {'─'*58}")

        # ── HOLD: no crossover → skip signal evaluation ───────────────────────
        if ma_dir == "HOLD":
            bias = "above" if ma_s > ma_l else "below"
            print(f"  [1] MA         : HOLD  (MA{SHORT_WINDOW} {bias} MA{LONG_WINDOW}, no crossover)")
            print(f"  └─ No trade\n")
            continue

        # ── Crossover detected: evaluate all three signals ────────────────────
        ma_ok = (ma_dir == "BUY") or (ma_dir == "SELL" and held_shares > 0)

        # Signal 2: ML
        df_daily       = fetch_daily_for_ml(ticker)
        ml_pred, ml_prob = ml_signal(ticker, df_daily) if df_daily is not None else (None, None)
        ml_ok = (
            (ma_dir == "BUY"  and ml_pred == 1) or
            (ma_dir == "SELL" and ml_pred == 0)
        ) if ml_pred is not None else False
        ml_available = ml_pred is not None

        # Signal 3: Sentiment
        sent_score, n_headlines = rss_sentiment(ticker)
        cache_note = ""
        if ticker in _sentiment_cache:
            age = int((time.time() - _sentiment_cache[ticker][2]) // 60)
            cache_note = f"  [cached {age}m ago]" if age > 0 else ""
        sent_ok = (
            (ma_dir == "BUY"  and sent_score >  SENTIMENT_BUY_THRESHOLD) or
            (ma_dir == "SELL" and sent_score <  SENTIMENT_SELL_THRESHOLD)
        )

        # Print signal table
        print(f"  [1] MA Crossover : {ma_dir:<4}  {_tick(True)}  "
              f"MA{SHORT_WINDOW}={ma_s:.2f} {'>' if ma_dir=='BUY' else '<'} MA{LONG_WINDOW}={ma_l:.2f}")

        ml_status = _ml_label(ml_pred, ml_prob)
        print(f"  [2] ML Model     : {ml_status:<20}  {_tick(ml_ok) if ml_available else '–'}  "
              f"{'agrees' if ml_ok else 'disagrees' if ml_available else 'unavailable'}")

        need_sent = f"> {SENTIMENT_BUY_THRESHOLD}" if ma_dir == "BUY" else f"< {SENTIMENT_SELL_THRESHOLD}"
        print(f"  [3] Sentiment    : {sent_score:+.4f} {_sent_label(sent_score):<14}  "
              f"{_tick(sent_ok)}  {n_headlines} headlines{cache_note}  (need {need_sent})")

        all_ok = ma_ok and ml_ok and sent_ok

        # ── BUY ───────────────────────────────────────────────────────────────
        if ma_dir == "BUY" and all_ok and held_value < max_allowed:
            budget = min(max_allowed - held_value, max_spend(portfolio, current_prices))
            if budget >= price:
                n    = int(budget // price)
                cost = n * price
                portfolio["cash"] -= cost
                portfolio["positions"][ticker] = held_shares + n
                reason = (
                    f"MA{SHORT_WINDOW}({ma_s:.2f})>MA{LONG_WINDOW}({ma_l:.2f}) | "
                    f"ML=UP({ml_prob:.0%}) | sent={sent_score:+.4f}"
                )
                log_trade("BUY", ticker, n, price, reason)
                print(f"  └─ ► EXECUTE BUY   {n} sh @ ${price:.2f}  "
                      f"cost=${cost:,.2f}  (all 3 signals confirmed)")
            else:
                print(f"  └─ ✗ BUY confirmed but insufficient budget "
                      f"(${budget:.2f} < ${price:.2f})")

        # ── SELL ──────────────────────────────────────────────────────────────
        elif ma_dir == "SELL" and all_ok and held_shares > 0:
            proceeds = held_shares * price
            portfolio["cash"] += proceeds
            portfolio["positions"].pop(ticker, None)
            reason = (
                f"MA{SHORT_WINDOW}({ma_s:.2f})<MA{LONG_WINDOW}({ma_l:.2f}) | "
                f"ML=DOWN({ml_prob:.0%}) | sent={sent_score:+.4f}"
            )
            log_trade("SELL", ticker, held_shares, price, reason)
            print(f"  └─ ► EXECUTE SELL  {held_shares} sh @ ${price:.2f}  "
                  f"proceeds=${proceeds:,.2f}  (all 3 signals confirmed)")

        # ── BLOCKED ───────────────────────────────────────────────────────────
        else:
            blockers = []
            if not ml_ok:
                blockers.append(
                    f"ML={'unavailable' if not ml_available else ('DOWN' if ma_dir=='BUY' else 'UP')}"
                )
            if not sent_ok:
                blockers.append(f"sentiment {sent_score:+.4f} (need {need_sent})")
            if ma_dir == "SELL" and held_shares == 0:
                blockers.append("no position to sell")
            reason_str = " + ".join(blockers) if blockers else "position limit reached"
            print(f"  └─ ✗ SKIP {ma_dir:<4}  — {reason_str}")

        print()

    save_portfolio(portfolio)

    # ── Portfolio snapshot ────────────────────────────────────────────────────
    total_now = portfolio_value(portfolio, current_prices)
    pnl       = total_now - STARTING_CASH
    pnl_pct   = pnl / STARTING_CASH * 100

    print(f"{'─'*W}")
    print(f"  PORTFOLIO SNAPSHOT")
    print(f"{'─'*W}")
    print(f"  {'Cash':<22}  ${portfolio['cash']:>12,.2f}")
    for ticker, shares in portfolio["positions"].items():
        p   = current_prices.get(ticker, 0)
        val = shares * p
        print(f"  {ticker:<22}  {shares:>8.2f} sh @ ${p:>8.2f}  = ${val:>10,.2f}")
    print(f"{'─'*W}")
    print(f"  {'Total Value':<22}  ${total_now:>12,.2f}")
    pnl_sign = "+" if pnl >= 0 else ""
    print(f"  {'P&L':<22}  ${pnl_sign}{pnl:>11,.2f}  ({pnl_sign}{pnl_pct:.2f}%)")
    print(f"{'═'*W}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 62)
    print("  Paper Trading Bot  —  Three-Signal System")
    print("═" * 62)
    print(f"  Tickers   : {', '.join(TICKERS)}")
    print(f"  Signal 1  : MA Crossover  (MA{SHORT_WINDOW} / MA{LONG_WINDOW}, 5-min bars)")
    print(f"  Signal 2  : ML Model      ({'loaded ✓' if _ml_ready else 'unavailable ✗'})")
    print(f"  Signal 3  : RSS Sentiment (Yahoo Finance + TextBlob, 30-min cache)")
    print(f"  Thresholds: BUY sent > {SENTIMENT_BUY_THRESHOLD}  │  SELL sent < {SENTIMENT_SELL_THRESHOLD}")
    print(f"  Risk      : max {MAX_POSITION_PCT:.0%} per position  │  {CASH_BUFFER_PCT:.0%} cash buffer")
    print(f"  Schedule  : every 5 minutes")
    print("═" * 62 + "\n")

    run_strategy()
    schedule.every(5).minutes.do(run_strategy)

    while True:
        schedule.run_pending()
        time.sleep(30)
