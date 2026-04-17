import json
import csv
import os
import re
import math
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import requests
import pandas as pd
from flask import Flask, render_template_string, jsonify
from textblob import TextBlob

# ── ML model (optional, loaded once) ──────────────────────────────────────────
try:
    import joblib
    from ml_model import build_features, FEATURE_COLS
    _ml_models = joblib.load("model.pkl")
    ML_READY   = True
except Exception:
    _ml_models = {}
    ML_READY   = False

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TICKERS        = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "JNJ", "WMT", "KO", "V"]
STARTING_CASH  = 100_000.0
SHORT_WINDOW   = 10
LONG_WINDOW    = 30
PORTFOLIO_FILE = "portfolio.json"
TRADES_FILE    = "trades.csv"
RISK_FREE      = 0.04

_http = requests.Session()
_http.headers.update({"User-Agent": "Mozilla/5.0 (compatible; dashboard/1.0)"})
_sentiment_cache: dict[str, tuple[float, float]] = {}   # ticker -> (score, ts)


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {"cash": STARTING_CASH, "positions": {}}


def load_trades() -> list[dict]:
    if not os.path.exists(TRADES_FILE):
        return []
    with open(TRADES_FILE, newline="") as f:
        return list(csv.DictReader(f))


def parse_reason(reason: str) -> dict:
    """Break 'MA10(263.67)>MA30(261.20) | ML=UP(52%) | sent=+0.35' into parts."""
    out = {"ma": "—", "ml": "—", "sentiment": "—"}
    if not reason:
        return out
    if re.search(r"MA\d+.*?>", reason):
        out["ma"] = "BUY"
    elif re.search(r"MA\d+.*?<", reason):
        out["ma"] = "SELL"
    m = re.search(r"ML=(UP|DOWN)", reason, re.I)
    if m:
        out["ml"] = m.group(1).upper()
    m = re.search(r"sent(?:iment)?=([+-]?\d+\.\d+)", reason, re.I)
    if m:
        out["sentiment"] = f"{float(m.group(1)):+.3f}"
    return out


def reconstruct_history(trades: list[dict]) -> tuple[list[str], list[float]]:
    """
    Replay trades to get a (labels, portfolio_values) time series.
    Returns two parallel lists; first point is always "Start"/$100k.
    """
    cash       = STARTING_CASH
    positions  = {}   # ticker -> (shares, avg_cost)
    last_price = {}   # ticker -> last seen price

    labels = ["Start"]
    values = [STARTING_CASH]

    for tr in trades:
        ticker = tr["ticker"]
        shares = float(tr["shares"])
        price  = float(tr["price"])
        value  = float(tr["value"])
        last_price[ticker] = price

        if tr["action"] == "BUY":
            cash -= value
            prev_sh, prev_cb = positions.get(ticker, (0, 0))
            new_sh = prev_sh + shares
            new_cb = (prev_sh * prev_cb + value) / new_sh if new_sh else price
            positions[ticker] = (new_sh, new_cb)
        elif tr["action"] == "SELL":
            cash += value
            positions.pop(ticker, None)

        equity = sum(sh * last_price.get(t, 0) for t, (sh, _) in positions.items())
        pv = cash + equity

        try:
            dt  = datetime.fromisoformat(tr["timestamp"])
            lbl = dt.strftime("%b %d %H:%M")
        except Exception:
            lbl = tr["timestamp"][:16]

        labels.append(lbl)
        values.append(round(pv, 2))

    return labels, values


def compute_stats(trades: list[dict], portfolio: dict,
                  port_values: list[float]) -> dict:
    cash      = portfolio["cash"]
    positions = portfolio["positions"]

    last_price: dict[str, float] = {}
    for tr in trades:
        last_price[tr["ticker"]] = float(tr["price"])

    equity    = sum(sh * last_price.get(t, 0) for t, sh in positions.items())
    port_val  = cash + equity
    total_ret = (port_val - STARTING_CASH) / STARTING_CASH * 100

    # Sharpe & max drawdown from reconstructed history
    sharpe = max_dd = 0.0
    if len(port_values) >= 3:
        s   = pd.Series(port_values)
        ret = s.pct_change().dropna()
        rf  = RISK_FREE / 252
        exc = ret - rf
        if exc.std() > 0:
            sharpe = float(exc.mean() / exc.std() * math.sqrt(252))
        peak   = s.cummax()
        dd     = (s - peak) / peak
        max_dd = float(dd.min() * 100)

    sells = [t for t in trades if t["action"] == "SELL"]
    wins  = 0
    for tr in sells:
        pnl_raw = tr.get("pnl", "")
        try:
            if float(pnl_raw) > 0:
                wins += 1
        except (ValueError, TypeError):
            pass
    win_rate = wins / len(sells) * 100 if sells else 0.0

    return dict(
        portfolio_value = port_val,
        cash            = cash,
        total_ret       = total_ret,
        sharpe          = sharpe,
        max_dd          = max_dd,
        win_rate        = win_rate,
        wins            = wins,
        losses          = len(sells) - wins,
        total_trades    = len(trades),
        buys            = sum(1 for t in trades if t["action"] == "BUY"),
        sells           = len(sells),
    )


def fetch_spy_normalized(start_label: str | None,
                         port_labels: list[str],
                         port_values: list[float]) -> tuple[list[str], list[float]]:
    """
    Return (labels, spy_values) where spy is indexed to 100 at the first
    portfolio trade date.  Falls back to last-60-days if no trades.
    """
    try:
        url  = ("https://query1.finance.yahoo.com/v8/finance/chart/"
                "SPY?interval=1d&range=2y")
        resp = _http.get(url, timeout=10)
        resp.raise_for_status()
        res  = resp.json()["chart"]["result"][0]
        ts   = res["timestamp"]
        cl   = res["indicators"]["quote"][0]["close"]
        spy  = pd.DataFrame(
            {"Close": cl},
            index=pd.to_datetime(ts, unit="s", utc=True).tz_localize(None).normalize()
        ).dropna()

        # Anchor SPY to 100 at the date of the first trade (or 60 days ago)
        if len(port_labels) > 1:
            try:
                anchor_dt = datetime.strptime(port_labels[1], "%b %d %H:%M")
                # Year not in label — use current year, handle Dec/Jan wrap
                anchor_dt = anchor_dt.replace(year=datetime.now().year)
            except Exception:
                anchor_dt = datetime.now() - timedelta(days=60)
        else:
            anchor_dt = datetime.now() - timedelta(days=60)

        spy_after  = spy[spy.index >= pd.Timestamp(anchor_dt.date())]
        if spy_after.empty:
            spy_after = spy.tail(60)

        base         = float(spy_after.iloc[0]["Close"])
        spy_labels   = [d.strftime("%b %d") for d in spy_after.index]
        spy_norm     = [round(float(v) / base * 100, 4) for v in spy_after["Close"]]

        # Normalise portfolio to 100 at same anchor
        if len(port_values) > 1:
            port_base   = port_values[1]
            port_norm   = [round(v / port_base * 100, 4) for v in port_values[1:]]
            port_labels_norm = port_labels[1:]
        else:
            port_norm        = [100.0]
            port_labels_norm = ["Now"]

        return spy_labels, spy_norm, port_labels_norm, port_norm

    except Exception as e:
        return [], [], port_labels, port_values


# ── Live signal helpers ────────────────────────────────────────────────────────

def _fetch_5m(ticker: str) -> pd.DataFrame | None:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval=5m&range=5d")
    try:
        r   = _http.get(url, timeout=10)
        res = r.json()["chart"]["result"][0]
        cl  = res["indicators"]["quote"][0]["close"]
        df  = pd.DataFrame(
            {"Close": cl},
            index=pd.to_datetime(res["timestamp"], unit="s", utc=True)
        ).dropna()
        return df if len(df) >= LONG_WINDOW else None
    except Exception:
        return None


def _fetch_daily(ticker: str) -> pd.DataFrame | None:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval=1d&range=6mo")
    try:
        r   = _http.get(url, timeout=10)
        res = r.json()["chart"]["result"][0]
        q   = res["indicators"]["quote"][0]
        df  = pd.DataFrame(
            {"Close": q["close"], "Volume": q["volume"]},
            index=pd.to_datetime(res["timestamp"], unit="s", utc=True)
                  .tz_localize(None).normalize()
        ).dropna(subset=["Close"])
        return df if len(df) >= 55 else None
    except Exception:
        return None


def _ma_signal(df: pd.DataFrame) -> tuple[float, float, float, str]:
    close  = df["Close"].squeeze()
    ma_s   = float(close.rolling(SHORT_WINDOW).mean().iloc[-1])
    ma_l   = float(close.rolling(LONG_WINDOW).mean().iloc[-1])
    price  = float(close.iloc[-1])
    prev_s = float(close.rolling(SHORT_WINDOW).mean().iloc[-2])
    prev_l = float(close.rolling(LONG_WINDOW).mean().iloc[-2])
    if (prev_s <= prev_l) and (ma_s > ma_l):
        sig = "BUY"
    elif (prev_s >= prev_l) and (ma_s < ma_l):
        sig = "SELL"
    else:
        sig = "HOLD"
    return price, ma_s, ma_l, sig


def _ml_prediction(ticker: str, df_daily: pd.DataFrame) -> str:
    if not ML_READY or ticker not in _ml_models:
        return "n/a"
    try:
        feat   = build_features(df_daily)[FEATURE_COLS]
        latest = feat.iloc[[-1]]
        if latest.isna().any().any():
            return "n/a"
        pred  = int(_ml_models[ticker].predict(latest)[0])
        proba = _ml_models[ticker].predict_proba(latest)[0]
        prob  = f"{float(proba[1]):.0%}" if len(proba) > 1 else ""
        return f"{'UP' if pred == 1 else 'DOWN'} {prob}"
    except Exception:
        return "n/a"


def _rss_sentiment(ticker: str) -> tuple[float, int]:
    now = time.time()
    if ticker in _sentiment_cache:
        score, ts = _sentiment_cache[ticker]
        if now - ts < 1800:
            return score, -1   # -1 signals cached
    url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline"
           f"?s={ticker}&region=US&lang=en-US")
    try:
        r     = _http.get(url, timeout=8)
        root  = ET.fromstring(r.content)
        items = root.findall(".//item")
        if not items:
            return 0.0, 0
        scores = [TextBlob(
            (i.findtext("title") or "") + " " + (i.findtext("description") or "")
        ).sentiment.polarity for i in items]
        avg = round(sum(scores) / len(scores), 4)
        _sentiment_cache[ticker] = (avg, now)
        return avg, len(scores)
    except Exception:
        return 0.0, 0


# ── API routes ─────────────────────────────────────────────────────────────────

@app.route("/api/data")
def api_data():
    portfolio = load_portfolio()
    trades    = load_trades()

    port_labels, port_values = reconstruct_history(trades)
    stats = compute_stats(trades, portfolio, port_values)

    spy_labels, spy_norm, pn_labels, pn_values = fetch_spy_normalized(
        port_labels[1] if len(port_labels) > 1 else None,
        port_labels, port_values
    )

    trade_rows = []
    for tr in reversed(trades):
        parsed = parse_reason(tr.get("reason", ""))
        trade_rows.append({
            "timestamp": tr["timestamp"][:16].replace("T", " "),
            "action":    tr["action"],
            "ticker":    tr["ticker"],
            "shares":    tr["shares"],
            "price":     f"${float(tr['price']):,.2f}",
            "value":     f"${float(tr['value']):,.2f}",
            "ma":        parsed["ma"],
            "ml":        parsed["ml"],
            "sentiment": parsed["sentiment"],
            "reason":    tr.get("reason", ""),
        })

    open_positions = []
    last_price = {t["ticker"]: float(t["price"]) for t in trades}
    for ticker, shares in portfolio["positions"].items():
        price = last_price.get(ticker, 0)
        open_positions.append({
            "ticker": ticker,
            "shares": shares,
            "price":  price,
            "value":  round(shares * price, 2),
        })

    return jsonify(
        stats         = stats,
        port_labels   = pn_labels,
        port_values   = pn_values,
        spy_labels    = spy_labels,
        spy_values    = spy_norm,
        trades        = trade_rows,
        positions     = open_positions,
        updated       = datetime.now().strftime("%H:%M:%S"),
    )


@app.route("/api/signals")
def api_signals():
    results = []
    for ticker in TICKERS:
        df5   = _fetch_5m(ticker)
        if df5 is None:
            results.append({"ticker": ticker, "price": "—",
                            "ma_sig": "no data", "ma_s": "—", "ma_l": "—",
                            "ml": "—", "sentiment": "—", "n_headlines": 0,
                            "overall": "—"})
            continue

        price, ma_s, ma_l, ma_sig = _ma_signal(df5)
        df_daily = _fetch_daily(ticker)
        ml       = _ml_prediction(ticker, df_daily) if df_daily is not None else "n/a"
        sent, n  = _rss_sentiment(ticker)

        if ma_sig == "BUY":
            ml_ok   = ml.startswith("UP")
            sent_ok = sent > 0.2
        elif ma_sig == "SELL":
            ml_ok   = ml.startswith("DOWN")
            sent_ok = sent < -0.2
        else:
            ml_ok = sent_ok = False

        if ma_sig == "HOLD":
            overall = "HOLD"
        elif ml_ok and sent_ok:
            overall = ma_sig + " ✓"
        else:
            blockers = []
            if not ml_ok:
                blockers.append("ML")
            if not sent_ok:
                blockers.append("Sent")
            overall = f"BLOCKED ({', '.join(blockers)})"

        results.append({
            "ticker":      ticker,
            "price":       f"${price:,.2f}",
            "ma_sig":      ma_sig,
            "ma_s":        f"{ma_s:.2f}",
            "ma_l":        f"{ma_l:.2f}",
            "ml":          ml,
            "sentiment":   f"{sent:+.3f}",
            "n_headlines": abs(n),
            "cached":      n == -1,
            "overall":     overall,
        })

    return jsonify(signals=results, updated=datetime.now().strftime("%H:%M:%S"))


# ── HTML template ──────────────────────────────────────────────────────────────

TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Trading Bot Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:       #0d1117;
  --surface:  #161b22;
  --border:   #30363d;
  --border2:  #21262d;
  --text:     #e6edf3;
  --muted:    #8b949e;
  --green:    #3fb950;
  --red:      #f85149;
  --blue:     #58a6ff;
  --yellow:   #d29922;
  --purple:   #bc8cff;
}
body { background: var(--bg); color: var(--text);
       font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       min-height: 100vh; font-size: 14px; }

/* ── Header ── */
header {
  background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 14px 28px; display: flex; align-items: center; gap: 14px;
  position: sticky; top: 0; z-index: 100;
}
.pulse { width: 9px; height: 9px; border-radius: 50%; background: var(--green);
         box-shadow: 0 0 8px var(--green); animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
header h1 { font-size: 1rem; font-weight: 600; letter-spacing:.03em; }
.header-right { margin-left: auto; display: flex; align-items: center; gap: 20px;
                font-size: .75rem; color: var(--muted); }
.countdown { color: var(--blue); font-weight: 600; }
#refresh-btn {
  background: none; border: 1px solid var(--border); color: var(--muted);
  padding: 4px 12px; border-radius: 6px; cursor: pointer; font-size: .73rem;
  transition: border-color .2s, color .2s;
}
#refresh-btn:hover { border-color: var(--blue); color: var(--blue); }

/* ── Layout ── */
main { max-width: 1400px; margin: 0 auto; padding: 24px 20px; }
section { margin-bottom: 22px; }

/* ── Cards ── */
.cards { display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; }
@media(max-width:1100px){ .cards{grid-template-columns:repeat(3,1fr)} }
@media(max-width:640px){ .cards{grid-template-columns:repeat(2,1fr)} }
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 18px 20px; transition: border-color .2s;
}
.card:hover { border-color: #58a6ff44; }
.card .label { font-size: .68rem; text-transform: uppercase; letter-spacing: .08em;
               color: var(--muted); margin-bottom: 8px; }
.card .val { font-size: 1.45rem; font-weight: 700; letter-spacing: -.01em; }
.card .sub { font-size: .73rem; color: var(--muted); margin-top: 3px; }
.up   { color: var(--green); }
.down { color: var(--red);   }
.neu  { color: var(--blue);  }

/* ── Panel (generic surface box) ── */
.panel { background: var(--surface); border: 1px solid var(--border);
         border-radius: 10px; padding: 20px 22px; }
.panel-title { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em;
               color: var(--muted); margin-bottom: 16px; }

/* ── Chart ── */
.chart-wrap { position: relative; height: 280px; }

/* ── Two-column grid ── */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media(max-width:900px){ .two-col{grid-template-columns:1fr} }

/* ── Tables ── */
.tbl-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; white-space: nowrap; }
th {
  font-size: .68rem; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); text-align: left; padding: 7px 12px;
  border-bottom: 1px solid var(--border);
}
td { padding: 9px 12px; font-size: .82rem; border-bottom: 1px solid var(--border2); }
tr:last-child td { border-bottom: none; }
tbody tr:hover { background: #1c2128; }

/* ── Pills ── */
.pill { display:inline-block; padding:2px 9px; border-radius:20px;
        font-size:.68rem; font-weight:700; letter-spacing:.04em; }
.pill-buy    { background:#1a2f4e; color:var(--blue);   }
.pill-sell   { background:#2d1a1a; color:var(--red);    }
.pill-hold   { background:#1e1e2e; color:var(--muted);  }
.pill-ok     { background:#1f4e2e; color:var(--green);  }
.pill-block  { background:#2d2015; color:var(--yellow); }
.pill-up     { background:#1f4e2e; color:var(--green);  }
.pill-down   { background:#2d1a1a; color:var(--red);    }
.pill-na     { background:#1e1e2e; color:var(--muted);  }

/* ── Sentiment bar ── */
.sent-bar { display:flex; align-items:center; gap:6px; }
.bar-track { flex:1; height:4px; background:var(--border2); border-radius:2px; }
.bar-fill  { height:100%; border-radius:2px; }

/* ── Empty / loading ── */
.empty { text-align:center; padding:36px; color:var(--muted); font-size:.82rem; }
.loading { display:flex; align-items:center; justify-content:center;
           gap:8px; padding:20px; color:var(--muted); font-size:.82rem; }
.spinner { width:14px; height:14px; border:2px solid var(--border);
           border-top-color:var(--blue); border-radius:50%;
           animation:spin .8s linear infinite; }
@keyframes spin{to{transform:rotate(360deg)}}

footer { text-align:center; padding:20px; color:#484f58; font-size:.72rem; }
</style>
</head>
<body>

<header>
  <div class="pulse"></div>
  <h1>Trading Bot Dashboard</h1>
  <div class="header-right">
    <span>Last updated: <strong id="last-updated">—</strong></span>
    <span>Next refresh: <span class="countdown" id="countdown">5:00</span></span>
    <button id="refresh-btn" onclick="refreshAll()">↻ Refresh now</button>
  </div>
</header>

<main>

<!-- ── Stats cards ── -->
<section>
  <div class="cards">
    <div class="card">
      <div class="label">Portfolio Value</div>
      <div class="val neu" id="stat-value">$100,000</div>
      <div class="sub">Starting: $100,000.00</div>
    </div>
    <div class="card">
      <div class="label">Total Return</div>
      <div class="val" id="stat-return">+0.00%</div>
      <div class="sub" id="stat-return-dollar">$0.00</div>
    </div>
    <div class="card">
      <div class="label">Sharpe Ratio</div>
      <div class="val" id="stat-sharpe">—</div>
      <div class="sub">Risk-adjusted return</div>
    </div>
    <div class="card">
      <div class="label">Max Drawdown</div>
      <div class="val" id="stat-drawdown">—</div>
      <div class="sub">Worst peak-to-trough</div>
    </div>
    <div class="card">
      <div class="label">Win Rate</div>
      <div class="val" id="stat-winrate">—</div>
      <div class="sub" id="stat-wl">— W / — L</div>
    </div>
    <div class="card">
      <div class="label">Total Trades</div>
      <div class="val" id="stat-trades">0</div>
      <div class="sub" id="stat-buysell">0 buys · 0 sells</div>
    </div>
  </div>
</section>

<!-- ── Chart ── -->
<section class="panel">
  <div class="panel-title">Portfolio vs S&amp;P 500 — normalized to 100 at first trade</div>
  <div class="chart-wrap">
    <canvas id="perfChart"></canvas>
  </div>
</section>

<!-- ── Signals + Positions ── -->
<section class="two-col">

  <div class="panel">
    <div class="panel-title">Live Signal Status
      <span style="font-size:.68rem;color:var(--muted);margin-left:8px" id="sig-updated"></span>
    </div>
    <div id="signals-body">
      <div class="loading"><div class="spinner"></div>Fetching live signals…</div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-title">Open Positions</div>
    <div class="tbl-wrap" id="positions-body">
      <div class="empty">No open positions — all capital in cash.</div>
    </div>
  </div>

</section>

<!-- ── Trade History ── -->
<section class="panel">
  <div class="panel-title">Trade History
    <span style="font-size:.68rem;color:var(--muted);margin-left:8px" id="trade-count"></span>
  </div>
  <div class="tbl-wrap" id="trades-body">
    <div class="empty">No trades yet — bot will populate this once the market opens.</div>
  </div>
</section>

</main>
<footer>Paper trading only &nbsp;·&nbsp; Not financial advice &nbsp;·&nbsp; Data: Yahoo Finance</footer>

<script>
let perfChart = null;

// ── Chart setup ───────────────────────────────────────────────────────────────
function buildChart(portLabels, portValues, spyLabels, spyValues) {
  const ctx = document.getElementById("perfChart").getContext("2d");

  const portUp  = (portValues[portValues.length-1] || 100) >= 100;
  const portCol = portUp ? "#3fb950" : "#f85149";

  const portGrad = ctx.createLinearGradient(0,0,0,280);
  portGrad.addColorStop(0, portUp ? "rgba(63,185,80,.22)" : "rgba(248,81,73,.22)");
  portGrad.addColorStop(1, "rgba(0,0,0,0)");

  if (perfChart) perfChart.destroy();

  perfChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: spyLabels.length ? spyLabels : portLabels,
      datasets: [
        {
          label: "Portfolio",
          data: (() => {
            // Align portfolio points to SPY labels (sparse join by index)
            if (!spyLabels.length) return portValues;
            // Just overlay — both series use their own x-axis via separate datasets
            return portValues;
          })(),
          borderColor: portCol,
          backgroundColor: portGrad,
          borderWidth: 2,
          pointRadius: portValues.length > 80 ? 0 : 3,
          pointHoverRadius: 5,
          fill: true,
          tension: 0.35,
          yAxisID: "y",
          xAxisID: "x",
          parsing: false,
          // Use raw index alignment — works when both series cover same period
        },
        {
          label: "SPY",
          data: spyValues,
          borderColor: "#58a6ff",
          backgroundColor: "transparent",
          borderWidth: 1.5,
          borderDash: [4,3],
          pointRadius: 0,
          fill: false,
          tension: 0.3,
          yAxisID: "y",
          xAxisID: "x",
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          display: true,
          labels: { color: "#8b949e", boxWidth: 12, font: { size: 11 } }
        },
        tooltip: {
          backgroundColor: "#1c2128",
          borderColor: "#30363d",
          borderWidth: 1,
          titleColor: "#8b949e",
          bodyColor: "#e6edf3",
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} (indexed)`
          }
        }
      },
      scales: {
        x: {
          ticks: { color:"#8b949e", font:{size:10}, maxTicksLimit:10, maxRotation:0 },
          grid:  { color:"#21262d" }
        },
        y: {
          ticks: {
            color:"#8b949e", font:{size:10},
            callback: v => v.toFixed(0)
          },
          grid: { color:"#21262d" }
        }
      }
    }
  });
}

// ── Render stats ──────────────────────────────────────────────────────────────
function renderStats(s) {
  const fmt  = v => (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
  const fmtD = v => (v >= 0 ? "+" : "-") + "$" + Math.abs(v).toLocaleString("en-US",{minimumFractionDigits:2});

  document.getElementById("stat-value").textContent =
    "$" + s.portfolio_value.toLocaleString("en-US",{minimumFractionDigits:2});

  const retEl = document.getElementById("stat-return");
  retEl.textContent = fmt(s.total_ret);
  retEl.className   = "val " + (s.total_ret >= 0 ? "up" : "down");

  document.getElementById("stat-return-dollar").textContent =
    fmtD(s.portfolio_value - 100000);

  const shrEl = document.getElementById("stat-sharpe");
  shrEl.textContent = s.sharpe !== 0 ? s.sharpe.toFixed(3) : "—";
  shrEl.className   = "val " + (s.sharpe >= 1 ? "up" : s.sharpe < 0 ? "down" : "neu");

  const ddEl = document.getElementById("stat-drawdown");
  ddEl.textContent = s.max_dd !== 0 ? s.max_dd.toFixed(2) + "%" : "—";
  ddEl.className   = "val " + (s.max_dd < -15 ? "down" : s.max_dd < -5 ? "neu" : "up");

  const wrEl = document.getElementById("stat-winrate");
  wrEl.textContent = s.win_rate !== 0 ? s.win_rate.toFixed(1) + "%" : "—";
  wrEl.className   = "val " + (s.win_rate >= 50 ? "up" : s.win_rate > 0 ? "neu" : "down");
  document.getElementById("stat-wl").textContent = `${s.wins}W / ${s.losses}L`;

  document.getElementById("stat-trades").textContent = s.total_trades;
  document.getElementById("stat-buysell").textContent =
    `${s.buys} buys · ${s.sells} sells`;
}

// ── Render trade table ────────────────────────────────────────────────────────
function renderTrades(trades) {
  const el = document.getElementById("trades-body");
  document.getElementById("trade-count").textContent =
    trades.length ? `(${trades.length} trades, newest first)` : "";

  if (!trades.length) {
    el.innerHTML = `<div class="empty">No trades yet — bot will populate this once the market opens.</div>`;
    return;
  }

  const mlPill = v => {
    if (v === "UP" || v.startsWith("UP"))
      return `<span class="pill pill-up">${v}</span>`;
    if (v === "DOWN" || v.startsWith("DOWN"))
      return `<span class="pill pill-down">${v}</span>`;
    return `<span class="pill pill-na">${v}</span>`;
  };

  const sentColor = v => {
    const n = parseFloat(v);
    if (isNaN(n)) return "var(--muted)";
    if (n >  0.2) return "var(--green)";
    if (n < -0.2) return "var(--red)";
    return "var(--muted)";
  };

  const rows = trades.map((t,i) => `
    <tr>
      <td style="color:var(--muted)">${trades.length - i}</td>
      <td style="color:var(--muted);font-size:.76rem">${t.timestamp}</td>
      <td><span class="pill ${t.action==="BUY"?"pill-buy":"pill-sell"}">${t.action}</span></td>
      <td><strong>${t.ticker}</strong></td>
      <td>${t.shares}</td>
      <td>${t.price}</td>
      <td>${t.value}</td>
      <td>${t.ma !== "—"
        ? `<span class="pill ${t.ma==="BUY"?"pill-buy":"pill-sell"}">${t.ma}</span>`
        : "<span style='color:var(--muted)'>—</span>"}</td>
      <td>${mlPill(t.ml)}</td>
      <td style="color:${sentColor(t.sentiment)}">${t.sentiment}</td>
    </tr>`).join("");

  el.innerHTML = `
    <table>
      <thead><tr>
        <th>#</th><th>Time</th><th>Action</th><th>Ticker</th>
        <th>Shares</th><th>Price</th><th>Value</th>
        <th>MA</th><th>ML</th><th>Sentiment</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ── Render positions ──────────────────────────────────────────────────────────
function renderPositions(positions) {
  const el = document.getElementById("positions-body");
  if (!positions.length) {
    el.innerHTML = `<div class="empty">No open positions — all capital in cash.</div>`;
    return;
  }
  const rows = positions.map(p => `
    <tr>
      <td><strong>${p.ticker}</strong></td>
      <td>${Number(p.shares).toFixed(2)} sh</td>
      <td>$${Number(p.price).toFixed(2)}</td>
      <td class="up">$${Number(p.value).toLocaleString("en-US",{minimumFractionDigits:2})}</td>
    </tr>`).join("");
  el.innerHTML = `
    <table>
      <thead><tr><th>Ticker</th><th>Shares</th><th>Price</th><th>Value</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ── Render signal table ───────────────────────────────────────────────────────
function renderSignals(signals) {
  const el = document.getElementById("signals-body");
  document.getElementById("sig-updated").textContent =
    signals.updated ? `· refreshed ${signals.updated}` : "";

  const maPill = v => {
    if (v === "BUY")  return `<span class="pill pill-buy">BUY</span>`;
    if (v === "SELL") return `<span class="pill pill-sell">SELL</span>`;
    return `<span class="pill pill-hold">HOLD</span>`;
  };

  const mlPill = v => {
    if (v.startsWith("UP"))   return `<span class="pill pill-up">${v}</span>`;
    if (v.startsWith("DOWN")) return `<span class="pill pill-down">${v}</span>`;
    return `<span class="pill pill-na">${v}</span>`;
  };

  const overallPill = v => {
    if (v.includes("✓"))   return `<span class="pill pill-ok">${v}</span>`;
    if (v === "HOLD")       return `<span class="pill pill-hold">HOLD</span>`;
    return `<span class="pill pill-block">${v}</span>`;
  };

  const sentColor = v => {
    const n = parseFloat(v);
    if (isNaN(n)) return "var(--muted)";
    if (n >  0.2) return "var(--green)";
    if (n < -0.2) return "var(--red)";
    return "var(--muted)";
  };

  const rows = signals.signals.map(s => `
    <tr>
      <td><strong>${s.ticker}</strong></td>
      <td>${s.price}</td>
      <td>${maPill(s.ma_sig)}</td>
      <td style="font-size:.75rem;color:var(--muted)">${s.ma_s} / ${s.ma_l}</td>
      <td>${mlPill(s.ml)}</td>
      <td style="color:${sentColor(s.sentiment)}">${s.sentiment}
        ${s.cached ? "<span style='font-size:.65rem;color:var(--muted)'>(cached)</span>" : ""}
      </td>
      <td>${overallPill(s.overall)}</td>
    </tr>`).join("");

  el.innerHTML = `
    <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>Ticker</th><th>Price</th><th>MA Signal</th>
        <th>MA10/MA30</th><th>ML Model</th><th>Sentiment</th><th>Overall</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
}

// ── Refresh orchestration ─────────────────────────────────────────────────────
async function refreshData() {
  try {
    const res  = await fetch("/api/data");
    const data = await res.json();

    renderStats(data.stats);
    renderTrades(data.trades);
    renderPositions(data.positions);
    buildChart(data.port_labels, data.port_values,
               data.spy_labels,  data.spy_values);

    document.getElementById("last-updated").textContent = data.updated;
  } catch(e) {
    console.error("Data refresh error:", e);
  }
}

async function refreshSignals() {
  document.getElementById("signals-body").innerHTML =
    `<div class="loading"><div class="spinner"></div>Fetching live signals…</div>`;
  try {
    const res  = await fetch("/api/signals");
    const data = await res.json();
    renderSignals(data);
  } catch(e) {
    document.getElementById("signals-body").innerHTML =
      `<div class="empty">Signal fetch failed — check bot logs.</div>`;
  }
}

async function refreshAll() {
  await refreshData();
  refreshSignals();   // fire and forget — slow external calls
  resetCountdown();
}

// ── Countdown timer ───────────────────────────────────────────────────────────
let _secsLeft = 300;
let _countdownTimer = null;

function resetCountdown() {
  _secsLeft = 300;
}

function tickCountdown() {
  _secsLeft = Math.max(0, _secsLeft - 1);
  const m = Math.floor(_secsLeft / 60);
  const s = String(_secsLeft % 60).padStart(2, "0");
  document.getElementById("countdown").textContent = `${m}:${s}`;
  if (_secsLeft === 0) refreshAll();
}

// ── Boot ──────────────────────────────────────────────────────────────────────
refreshAll();
_countdownTimer = setInterval(tickCountdown, 1000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(TEMPLATE)


if __name__ == "__main__":
    print("Dashboard → http://localhost:5000")
    app.run(debug=False, port=5000)
