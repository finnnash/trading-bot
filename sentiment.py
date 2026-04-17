"""
sentiment.py — Finnhub News Sentiment
Returns a score from -1.0 (very bearish) to +1.0 (very bullish) for a ticker.

Endpoint : GET /api/v1/news-sentiment?symbol=AAPL&token=...
Free tier : 60 requests/minute — no daily cap.

Score formula: bullishPercent - bearishPercent
  e.g. 70% bullish, 30% bearish → 0.70 - 0.30 = +0.40

Results are cached for 30 minutes so the bot doesn't spam the API.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ── API key ────────────────────────────────────────────────────────────────────
# Set FINNHUB_KEY in .env (or as a shell environment variable).
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "YOUR_KEY_HERE")

# ── Thresholds (consumed by bot.py) ───────────────────────────────────────────
BUY_THRESHOLD  =  0.2   # sentiment must be above this to allow a BUY
SELL_THRESHOLD = -0.2   # sentiment must be below this to allow a SELL

# ── Cache ──────────────────────────────────────────────────────────────────────
_cache: dict[str, tuple[float, float]] = {}   # ticker -> (score, fetched_at)
CACHE_TTL = 30 * 60                           # 30 minutes

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; sentiment-bot/1.0)"})


# ── Core fetch ─────────────────────────────────────────────────────────────────

def _fetch_from_api(ticker: str) -> float:
    """
    Call Finnhub /news-sentiment and return a -1 to +1 score.

    Response fields we use:
      sentiment.bullishPercent  — fraction of bullish articles  (0–1)
      sentiment.bearishPercent  — fraction of bearish articles  (0–1)

    score = bullishPercent - bearishPercent
    """
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

    # Finnhub returns an empty object {} when the symbol isn't covered
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


# ── Public interface ───────────────────────────────────────────────────────────

def get_sentiment(ticker: str) -> float:
    """
    Return the sentiment score for ticker, served from a 30-minute cache.

    Score interpretation:
      >  0.2  → bullish  (BUY allowed)
      ± 0.2   → neutral  (no trade)
      < -0.2  → bearish  (SELL allowed)

    Returns 0.0 on any error so the bot fails safe.
    """
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


# ── Quick test ─────────────────────────────────────────────────────────────────

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
