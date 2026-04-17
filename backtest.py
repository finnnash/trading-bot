"""Runs three scenarios side-by-side over 2 years: MA-only, MA+ML, MA+ML+slippage."""

import csv
import math
import os
import sys

import requests
import pandas as pd
import numpy as np

# config
TICKERS          = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "JNJ", "WMT", "KO", "V"]
BENCHMARK        = "SPY"
STARTING_CASH    = 100_000.0
SHORT_WINDOW     = 10
LONG_WINDOW      = 30
MAX_POSITION_PCT = 0.20
CASH_BUFFER_PCT  = 0.05
RISK_FREE_ANNUAL = 0.04
SLIPPAGE         = 0.001   # 0.1% per fill
MODEL_FILE       = "model.pkl"
CSV_MA           = "backtest_results.csv"
CSV_COMBINED     = "backtest_results_combined.csv"
CSV_SLIP         = "backtest_results_combined_slippage.csv"

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; backtest/1.0)"})


# data fetching

def fetch_daily(ticker: str) -> pd.DataFrame:
    """2 years of daily Close + Volume from Yahoo Finance."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&range=2y"
    )
    resp   = _session.get(url, timeout=15)
    resp.raise_for_status()
    data   = resp.json()
    result = data["chart"]["result"][0]
    ts     = result["timestamp"]
    quote  = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"Close": quote["close"], "Volume": quote["volume"]},
        index=pd.to_datetime(ts, unit="s", utc=True).tz_localize(None).normalize(),
    )
    df.dropna(subset=["Close"], inplace=True)
    return df


# MA signals

def add_ma_signals(df: pd.DataFrame) -> pd.DataFrame:
    df         = df.copy()
    df["MA_s"] = df["Close"].rolling(SHORT_WINDOW).mean()
    df["MA_l"] = df["Close"].rolling(LONG_WINDOW).mean()
    above      = df["MA_s"] > df["MA_l"]
    prev       = above.shift(1).fillna(False)
    df["buy"]  = (~prev) & above     # crossed up
    df["sell"] = prev & (~above)     # crossed down
    return df


# ML predictions

def load_ml_predictions(all_data: dict) -> dict[str, pd.Series]:
    """Load model.pkl and return per-ticker prediction series (1=UP, 0=DOWN)."""
    if not os.path.exists(MODEL_FILE):
        print(f"  WARNING: {MODEL_FILE} not found — run ml_model.py first.")
        print("  Continuing with MA-only backtest.\n")
        return {}

    try:
        import joblib
        from ml_model import build_features, FEATURE_COLS
    except ImportError as e:
        print(f"  WARNING: Could not import ML dependencies ({e}).")
        print("  Continuing with MA-only backtest.\n")
        return {}

    models = joblib.load(MODEL_FILE)
    preds  = {}

    for ticker in TICKERS:
        if ticker not in models:
            print(f"  [{ticker}] not found in model.pkl — skipping ML for this ticker.")
            continue

        feat  = build_features(all_data[ticker])[FEATURE_COLS]
        valid = feat.dropna()
        raw   = models[ticker].predict(valid)
        preds[ticker] = pd.Series(raw, index=valid.index, dtype=int)

    return preds


# simulation engine

def simulate(
    all_data: dict,
    common_dates: list,
    ml_preds: dict | None = None,
    slippage: float = 0.0,
) -> tuple[float, list, list, float]:
    """
    Run one backtest pass.

    ml_preds=None  → MA-only
    ml_preds=dict  → trade only when MA and ML agree
    slippage=0.001 → 0.1% worse fill on every BUY and SELL

    BUY fill  = price * (1 + slippage)  — pay more
    SELL fill = price * (1 - slippage)  — receive less
    Portfolio value always uses raw market price, not fill price.

    Returns (final_value, trade_log, daily_values, total_slippage_cost).
    """
    use_ml          = bool(ml_preds)
    cash            = STARTING_CASH
    positions       = {}
    cost_basis      = {}
    trade_log       = []
    daily_vals      = []
    total_slip_cost = 0.0

    for date in common_dates:
        prices = {
            t: float(all_data[t].at[date, "Close"])
            for t in TICKERS if date in all_data[t].index
        }
        # value positions at market price (not fill price)
        equity = sum(positions.get(t, 0) * prices.get(t, 0) for t in TICKERS)
        total  = cash + equity

        for ticker in TICKERS:
            if ticker not in prices:
                continue
            price = prices[ticker]
            row   = all_data[ticker].loc[date]

            if pd.isna(row["MA_s"]) or pd.isna(row["MA_l"]):
                continue

            held = positions.get(ticker, 0)
            hval = held * price

            ml_up   = True
            ml_down = True
            if use_ml:
                ticker_preds = ml_preds.get(ticker, pd.Series(dtype=int))
                if date in ticker_preds.index:
                    pred    = int(ticker_preds.at[date])
                    ml_up   = (pred == 1)
                    ml_down = (pred == 0)
                else:
                    ml_up = ml_down = False

            # BUY
            if row["buy"] and hval < total * MAX_POSITION_PCT and ml_up:
                budget = min(
                    total * MAX_POSITION_PCT - hval,
                    cash - total * CASH_BUFFER_PCT,
                )
                fill_price  = price * (1 + slippage)
                slip_cost   = price * slippage
                if budget >= fill_price:
                    n           = int(budget // fill_price)
                    cost        = n * fill_price
                    slip_total  = n * slip_cost
                    total_slip_cost += slip_total
                    cash       -= cost
                    prev        = positions.get(ticker, 0)
                    new_total   = prev + n
                    prev_cb     = cost_basis.get(ticker, 0.0)
                    cost_basis[ticker] = (prev * prev_cb + cost) / new_total
                    positions[ticker]  = new_total
                    trade_log.append({
                        "date":         date.date(),
                        "action":       "BUY",
                        "ticker":       ticker,
                        "shares":       n,
                        "price":        round(price, 4),
                        "fill_price":   round(fill_price, 4),
                        "value":        round(cost, 2),
                        "slippage_cost": round(slip_total, 4),
                        "ma_short":     round(float(row["MA_s"]), 4),
                        "ma_long":      round(float(row["MA_l"]),  4),
                        "pnl":          "",
                    })

            # SELL
            elif row["sell"] and held > 0 and ml_down:
                fill_price  = price * (1 - slippage)
                slip_cost   = price * slippage
                proceeds    = held * fill_price
                slip_total  = held * slip_cost
                total_slip_cost += slip_total
                avg_cb      = cost_basis.get(ticker, price)
                pnl         = round((fill_price - avg_cb) * held, 2)
                cash       += proceeds
                positions.pop(ticker, None)
                cost_basis.pop(ticker, None)
                trade_log.append({
                    "date":         date.date(),
                    "action":       "SELL",
                    "ticker":       ticker,
                    "shares":       held,
                    "price":        round(price, 4),
                    "fill_price":   round(fill_price, 4),
                    "value":        round(proceeds, 2),
                    "slippage_cost": round(slip_total, 4),
                    "ma_short":     round(float(row["MA_s"]), 4),
                    "ma_long":      round(float(row["MA_l"]),  4),
                    "pnl":          pnl,
                })

        equity_now = sum(positions.get(t, 0) * prices.get(t, 0) for t in TICKERS)
        daily_vals.append(cash + equity_now)

    # liquidate at last close (no slippage — notional)
    end_date    = common_dates[-1]
    last_prices = {
        t: float(all_data[t].at[end_date, "Close"])
        for t in TICKERS if end_date in all_data[t].index
    }
    final_value = cash + sum(positions.get(t, 0) * last_prices.get(t, 0) for t in TICKERS)

    return final_value, trade_log, daily_vals, round(total_slip_cost, 2)


# metrics

def compute_metrics(final_value: float, trade_log: list, daily_vals: list,
                    years: float, slippage_cost: float = 0.0) -> dict:
    total_ret  = (final_value - STARTING_CASH) / STARTING_CASH * 100
    ann_ret    = ((final_value / STARTING_CASH) ** (1 / years) - 1) * 100

    dvs        = pd.Series(daily_vals)
    daily_rets = dvs.pct_change().dropna()
    rf_daily   = RISK_FREE_ANNUAL / 252
    excess     = daily_rets - rf_daily
    sharpe     = (excess.mean() / excess.std() * math.sqrt(252)) if excess.std() > 0 else 0.0

    peak       = dvs.cummax()
    max_dd     = float(((dvs - peak) / peak).min()) * 100

    sells      = [t for t in trade_log if t["action"] == "SELL" and t["pnl"] != ""]
    wins       = sum(1 for t in sells if float(t["pnl"]) > 0)
    losses     = sum(1 for t in sells if float(t["pnl"]) <= 0)
    win_rate   = wins / len(sells) * 100 if sells else 0.0

    buys_n     = sum(1 for t in trade_log if t["action"] == "BUY")
    sells_n    = sum(1 for t in trade_log if t["action"] == "SELL")

    # per-ticker P&L
    ticker_pnl = {}
    for ticker in TICKERS:
        t_sells = [t for t in sells if t["ticker"] == ticker]
        ticker_pnl[ticker] = sum(float(t["pnl"]) for t in t_sells)

    return dict(
        final_value=final_value,
        total_ret=total_ret,
        ann_ret=ann_ret,
        sharpe=sharpe,
        max_dd=max_dd,
        win_rate=win_rate,
        wins=wins,
        losses=losses,
        total_trades=len(trade_log),
        buys=buys_n,
        sells=sells_n,
        ticker_pnl=ticker_pnl,
        slippage_cost=slippage_cost,
    )


# reporting

def print_comparison(ma: dict, combo: dict, combo_slip: dict,
                     spy_ret: float, spy_final: float,
                     start_date, end_date, years: float) -> None:
    W  = 84
    HR = "═" * W
    hr = "─" * W

    def fp(v):
        return f"{'+' if v >= 0 else ''}{v:.2f}%"

    def fd(v):
        return f"${v:,.2f}"

    def row(label, c1, c2, c3, c4="—"):
        print(f"  {label:<26}  {c1:>13}  {c2:>13}  {c3:>13}  {c4:>11}")

    print(f"\n{HR}")
    print(f"  STRATEGY COMPARISON  ·  {start_date} → {end_date}  ({years:.2f} years)")
    print(HR)
    row("Metric", "MA Only", "MA + ML", "MA+ML +slip", "SPY (B&H)")
    print(hr)

    row("Final Value",
        fd(ma["final_value"]),
        fd(combo["final_value"]),
        fd(combo_slip["final_value"]),
        fd(spy_final))

    row("Total Return",
        fp(ma["total_ret"]),
        fp(combo["total_ret"]),
        fp(combo_slip["total_ret"]),
        fp(spy_ret))

    row("Annualized Return",
        fp(ma["ann_ret"]),
        fp(combo["ann_ret"]),
        fp(combo_slip["ann_ret"]),
        "—")

    row("Sharpe Ratio",
        f"{ma['sharpe']:.3f}",
        f"{combo['sharpe']:.3f}",
        f"{combo_slip['sharpe']:.3f}",
        "—")

    row("Max Drawdown",
        fp(ma["max_dd"]),
        fp(combo["max_dd"]),
        fp(combo_slip["max_dd"]),
        "—")

    row("Win Rate",
        f"{ma['win_rate']:.1f}% ({ma['wins']}W/{ma['losses']}L)",
        f"{combo['win_rate']:.1f}% ({combo['wins']}W/{combo['losses']}L)",
        f"{combo_slip['win_rate']:.1f}% ({combo_slip['wins']}W/{combo_slip['losses']}L)",
        "—")

    row("Total Trades",
        f"{ma['total_trades']} ({ma['buys']}B·{ma['sells']}S)",
        f"{combo['total_trades']} ({combo['buys']}B·{combo['sells']}S)",
        f"{combo_slip['total_trades']} ({combo_slip['buys']}B·{combo_slip['sells']}S)",
        "1")

    print(hr)

    ma_alpha      = ma["total_ret"]         - spy_ret
    combo_alpha   = combo["total_ret"]      - spy_ret
    slip_alpha    = combo_slip["total_ret"] - spy_ret
    row("Alpha vs SPY", fp(ma_alpha), fp(combo_alpha), fp(slip_alpha), "+0.00%")

    # slippage impact
    print(hr)
    print(f"  SLIPPAGE IMPACT  (0.1% per fill)")
    print(hr)

    ret_drag      = combo["total_ret"] - combo_slip["total_ret"]
    value_drag    = combo["final_value"] - combo_slip["final_value"]
    slip_cost     = combo_slip["slippage_cost"]
    n_trades      = combo_slip["total_trades"]
    avg_per_trade = slip_cost / n_trades if n_trades else 0
    ann_drag      = combo["ann_ret"] - combo_slip["ann_ret"]
    sharpe_drag   = combo["sharpe"] - combo_slip["sharpe"]

    print(f"  {'Total slippage paid':<34}  ${slip_cost:>10,.2f}")
    print(f"  {'Return drag':<34}  {ret_drag:>+10.2f}%")
    print(f"  {'Annualized return drag':<34}  {ann_drag:>+10.2f}%")
    print(f"  {'Final value drag':<34}  ${value_drag:>10,.2f}")
    print(f"  {'Sharpe drag':<34}  {sharpe_drag:>+10.3f}")
    print(f"  {'Avg slippage per trade':<34}  ${avg_per_trade:>10,.2f}")
    print(f"  {'Avg cost per round trip (2 fills)':<34}  ${avg_per_trade*2:>10,.2f}")

    # does slippage flip our alpha?
    print(hr)
    still_beats_spy = combo_slip["total_ret"] > spy_ret
    print(f"  MA+ML still beats SPY after slippage?  "
          f"{'YES  (' + fp(slip_alpha) + ' alpha)' if still_beats_spy else 'NO   (' + fp(slip_alpha) + ' alpha)'}")

    # per-ticker P&L comparison
    print(hr)
    print(f"  PER-TICKER  —  MA+ML  (no slippage vs 0.1% slippage)")
    print(f"  {'Ticker':<8}  {'No Slip':>14}  {'With Slip':>14}  {'Slip Cost':>12}  {'Drag':>8}")
    print(f"  {'─'*6:<8}  {'─'*10:>14}  {'─'*10:>14}  {'─'*9:>12}  {'─'*6:>8}")

    for ticker in TICKERS:
        pnl_c  = combo["ticker_pnl"].get(ticker, 0)
        pnl_s  = combo_slip["ticker_pnl"].get(ticker, 0)
        drag   = pnl_c - pnl_s
        s1 = f"+${pnl_c:,.2f}" if pnl_c >= 0 else f"-${abs(pnl_c):,.2f}"
        s2 = f"+${pnl_s:,.2f}" if pnl_s >= 0 else f"-${abs(pnl_s):,.2f}"
        s3 = f"${drag:,.2f}"
        print(f"  {ticker:<8}  {s1:>14}  {s2:>14}  {s3:>12}  {'':>8}")

    print(HR)
    print(f"\n  NOTE: ML model trained on same 2-year period (first 80% in-sample).")
    print(f"        Use walk-forward validation for production-grade evaluation.\n")


# CSV export

def save_csv(trade_log: list, path: str) -> None:
    fields = ["date", "action", "ticker", "shares", "price", "fill_price",
              "value", "slippage_cost", "ma_short", "ma_long", "pnl"]
    # back-fill keys for logs that predate slippage columns
    for row in trade_log:
        row.setdefault("fill_price",    row.get("price", ""))
        row.setdefault("slippage_cost", 0.0)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trade_log)
    print(f"  Saved {len(trade_log):>3} trades → {path}")


# entry point

def run_backtest():
    print("\nFetching 2 years of daily data…")
    all_data = {}
    for ticker in TICKERS:
        print(f"  {ticker:<6}", end="", flush=True)
        df = fetch_daily(ticker)
        all_data[ticker] = add_ma_signals(df)
        print(f"  {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")

    print(f"  {BENCHMARK:<6}", end="", flush=True)
    spy = fetch_daily(BENCHMARK)
    print(f"  {len(spy)} bars  ({spy.index[0].date()} → {spy.index[-1].date()})")

    common_dates = sorted(
        set.intersection(*[set(df.index) for df in all_data.values()])
    )
    start_date, end_date = common_dates[0], common_dates[-1]
    years = (end_date - start_date).days / 365.25

    print(f"\n  Period : {start_date.date()} → {end_date.date()}  ({years:.2f} yrs, {len(common_dates)} trading days)")

    # SPY benchmark
    spy_dates   = spy.index
    spy_s       = float(spy.loc[spy_dates[spy_dates >= start_date][0], "Close"])
    spy_e       = float(spy.loc[spy_dates[spy_dates <= end_date][-1],  "Close"])
    spy_ret_pct = (spy_e - spy_s) / spy_s * 100
    spy_final   = STARTING_CASH * (spy_e / spy_s)

    print("\nLoading ML model…")
    ml_preds = load_ml_predictions(all_data)
    if ml_preds:
        coverage = {t: len(ml_preds[t]) for t in ml_preds}
        print(f"  Loaded predictions — coverage: {coverage}")
    else:
        print("  Running MA-only (no ML model available).")

    # run all three scenarios
    print("\nRunning MA-only            (0% slip)…", end="  ", flush=True)
    ma_final, ma_trades, ma_daily, _ = simulate(
        all_data, common_dates, ml_preds=None, slippage=0.0)
    print(f"done  ({len(ma_trades)} trades)")

    print("Running MA + ML            (0% slip)…", end="  ", flush=True)
    combo_final, combo_trades, combo_daily, _ = simulate(
        all_data, common_dates, ml_preds=ml_preds, slippage=0.0)
    print(f"done  ({len(combo_trades)} trades)")

    print(f"Running MA + ML  (0.1% slippage)…", end="  ", flush=True)
    slip_final, slip_trades, slip_daily, slip_cost = simulate(
        all_data, common_dates, ml_preds=ml_preds, slippage=SLIPPAGE)
    print(f"done  ({len(slip_trades)} trades)  total slippage=${slip_cost:,.2f}")

    # compute metrics
    ma_m    = compute_metrics(ma_final,    ma_trades,    ma_daily,    years)
    combo_m = compute_metrics(combo_final, combo_trades, combo_daily, years)
    slip_m  = compute_metrics(slip_final,  slip_trades,  slip_daily,  years,
                              slippage_cost=slip_cost)

    print_comparison(ma_m, combo_m, slip_m, spy_ret_pct, spy_final,
                     start_date.date(), end_date.date(), years)

    # save CSVs
    save_csv(ma_trades,   CSV_MA)
    save_csv(combo_trades, CSV_COMBINED)
    save_csv(slip_trades,  CSV_SLIP)
    print()


if __name__ == "__main__":
    run_backtest()
