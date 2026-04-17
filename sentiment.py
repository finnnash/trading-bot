"""Finnhub news sentiment. Returns -1 to +1 score for a ticker. Cached 30 min."""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# set FINNHUB_KEY in .env
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "YOUR_KEY_HERE")

# score thresholds used by bot.py
BUY_THRESHOLD  =  0.2
SELL_THRESHOLD = -0.2

# cache
_cache: dict[str, tuple[float, float]] = {}   # ticker -> (score, fetched_at)
CACHE_TTL = 30 * 60

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; sentiment-bot/1.0)"})


def _fetch_from_api(ticker: str) -> float:
    """Hit Finnhub /news-sentiment, return bullish% - bearish% as a -1 to +1 score."""
    if FINNHUB_KEY == "YOUR_KEY_HERE":
        print(f"  [sentiment:{ticker}] WARNING: No API key set — returning neutral 0.0")
        return 0.0

    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={FINNHUB_KEY}"

    try:
        resp = _session.get(url, timeout=10)
        print(f"  [sentiment:{ticker}] HTTP {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [sentiment:{ticker}] Request error: {e}")
        return 0.0

    # empty dict means Finnhub doesn't cover this symbol
    if not data or "sentiment" not in data:
        print(f"  [sentiment:{ticker}] No sentiment data — keys returned: {list(data.keys())}")
        return 0.0

    sentiment    = data["sentiment"]
    bullish      = float(sentiment.get("bullishPercent", 0.0))
    bearish      = float(sentiment.get("bearishPercent", 0.0))
    score        = round(bullish - bearish, 4)

    articles     = data.get("buzz", {}).get("articlesInLastWeek", "n/a")
    news_score   = data.get("companyNewsScore", "n/a")

    print(f"  [sentiment:{ticker}] bullish={bullish:.2%}  bearish={bearish:.2%}"
          f"  articles_7d={articles}  companyNewsScore={news_score}")

    return score


def get_sentiment(ticker: str) -> float:
    """Return cached or fresh sentiment score. Returns 0.0 on any error."""
    now = time.time()

    if ticker in _cache:
        cached_score, fetched_at = _cache[ticker]
        age = now - fetched_at
        if age < CACHE_TTL:
            mins = int(age // 60)
            secs = int(age % 60)
            print(f"  [sentiment:{ticker}] cached  score={cached_score:+.4f}  age={mins}m{secs:02d}s")
            return cached_score

    score = _fetch_from_api(ticker)
    _cache[ticker] = (score, now)
    print(f"  [sentiment:{ticker}] score={score:+.4f}  → {_label(score)}")
    return score


def _label(score: float) -> str:
    if score >=  0.5: return "BULLISH"
    if score >=  0.2: return "Somewhat-Bullish"
    if score >  -0.2: return "Neutral"
    if score >  -0.5: return "Somewhat-Bearish"
    return "BEARISH"


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "MSFT", "GOOGL", "NVDA"]
    print(f"Finnhub sentiment test — {', '.join(tickers)}\n")
    for t in tickers:
        score = get_sentiment(t)
        filled = int(abs(score) * 20)
        bar    = ("█" * filled).ljust(20)
        sign   = "+" if score >= 0 else ""
        print(f"  {t:<6}  {sign}{score:.4f}  [{bar}]  {_label(score)}\n")
