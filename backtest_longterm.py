"""MA10/MA30 crossover backtest over 10 years vs SPY buy-and-hold."""

import csv
import math

import requests
import pandas as pd
import numpy as np

# config
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "JNJ", "WMT", "KO", "PG", "V"]
BENCHMARK        = "SPY"
STARTING_CASH    = 100_000.0
SHORT_WINDOW     = 10
LONG_WINDOW      = 30
MAX_POSITION_PCT = 0.20
CASH_BUFFER_PCT  = 0.05
RISK_FREE_ANNUAL = 0.04
OUTPUT_CSV       = "backtest_longterm_results.csv"

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; backtest-lt/1.0)"})


# data fetching

def fetch_daily(ticker: str, period: str = "10y") -> pd.DataFrame:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&range={period}"
    )
    resp = _session.get(url, timeout=20)
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
    df["buy"]  = (~prev) & above
    df["sell"] = prev & (~above)
    return df


# simulation

def simulate(all_data: dict, common_dates: list) -> tuple[float, list, list]:
    cash       = STARTING_CASH
    positions  = {}
    cost_basis = {}
    trade_log  = []
    daily_vals = []

    for date in common_dates:
        prices = {
            t: float(all_data[t].at[date, "Close"])
            for t in TICKERS if date in all_data[t].index
        }
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

            # BUY
            if row["buy"] and hval < total * MAX_POSITION_PCT:
                budget = min(
                    total * MAX_POSITION_PCT - hval,
                    cash - total * CASH_BUFFER_PCT,
                )
                if budget >= price:
                    n    = int(budget // price)
                    cost = n * price
                    cash -= cost
                    prev_n  = positions.get(ticker, 0)
                    new_n   = prev_n + n
                    prev_cb = cost_basis.get(ticker, 0.0)
                    cost_basis[ticker] = (prev_n * prev_cb + cost) / new_n
                    positions[ticker]  = new_n
                    trade_log.append({
                        "date":     date.date(),
                        "action":   "BUY",
                        "ticker":   ticker,
                        "shares":   n,
                        "price":    round(price, 4),
                        "value":    round(cost, 2),
                        "ma_short": round(float(row["MA_s"]), 4),
                        "ma_long":  round(float(row["MA_l"]), 4),
                        "pnl":      "",
                    })

            # SELL
            elif row["sell"] and held > 0:
                proceeds = held * price
                avg_cb   = cost_basis.get(ticker, price)
                pnl      = round((price - avg_cb) * held, 2)
                cash    += proceeds
                positions.pop(ticker, None)
                cost_basis.pop(ticker, None)
                trade_log.append({
                    "date":     date.date(),
                    "action":   "SELL",
                    "ticker":   ticker,
                    "shares":   held,
                    "price":    round(price, 4),
                    "value":    round(proceeds, 2),
                    "ma_short": round(float(row["MA_s"]), 4),
                    "ma_long":  round(float(row["MA_l"]), 4),
                    "pnl":      pnl,
                })

        equity_now = sum(positions.get(t, 0) * prices.get(t, 0) for t in TICKERS)
        daily_vals.append(cash + equity_now)

    # liquidate at last close
    end_date    = common_dates[-1]
    last_prices = {
        t: float(all_data[t].at[end_date, "Close"])
        for t in TICKERS if end_date in all_data[t].index
    }
    final_value = cash + sum(positions.get(t, 0) * last_prices.get(t, 0) for t in TICKERS)
    return final_value, trade_log, daily_vals


# metrics

def compute_metrics(final_value: float, trade_log: list,
                    daily_vals: list, years: float) -> dict:
    total_ret  = (final_value - STARTING_CASH) / STARTING_CASH * 100
    ann_ret    = ((final_value / STARTING_CASH) ** (1 / years) - 1) * 100

    dvs        = pd.Series(daily_vals)
    daily_rets = dvs.pct_change().dropna()
    rf_daily   = RISK_FREE_ANNUAL / 252
    excess     = daily_rets - rf_daily
    sharpe     = (excess.mean() / excess.std() * math.sqrt(252)) if excess.std() > 0 else 0.0

    peak   = dvs.cummax()
    max_dd = float(((dvs - peak) / peak).min()) * 100

    sells  = [t for t in trade_log if t["action"] == "SELL" and t["pnl"] != ""]
    wins   = sum(1 for t in sells if float(t["pnl"]) > 0)
    losses = sum(1 for t in sells if float(t["pnl"]) <= 0)
    win_rt = wins / len(sells) * 100 if sells else 0.0

    return dict(
        final_value=final_value, total_ret=total_ret, ann_ret=ann_ret,
        sharpe=sharpe, max_dd=max_dd, win_rate=win_rt,
        wins=wins, losses=losses,
        total_trades=len(trade_log),
        buys=sum(1 for t in trade_log if t["action"] == "BUY"),
        sells=sum(1 for t in trade_log if t["action"] == "SELL"),
    )


# per-ticker breakdown

def ticker_breakdown(trade_log: list, all_data: dict, common_dates: list) -> list:
    rows = []
    for ticker in TICKERS:
        trades = [t for t in trade_log if t["ticker"] == ticker]
        sells  = [t for t in trades if t["action"] == "SELL" and t["pnl"] != ""]
        wins   = sum(1 for t in sells if float(t["pnl"]) > 0)
        pnl    = sum(float(t["pnl"]) for t in sells)

        # buy-and-hold return for this ticker over the same period
        start = common_dates[0]
        end   = common_dates[-1]
        df    = all_data[ticker]
        try:
            price_start = float(df.loc[df.index[df.index >= start][0],  "Close"])
            price_end   = float(df.loc[df.index[df.index <= end][-1],   "Close"])
            bah_ret     = (price_end - price_start) / price_start * 100
        except Exception:
            bah_ret = float("nan")

        rows.append({
            "ticker":   ticker,
            "trades":   len(trades),
            "wins":     wins,
            "losses":   len(sells) - wins,
            "pnl":      pnl,
            "bah_ret":  bah_ret,
        })
    return rows


# report

def print_report(m: dict, spy_ret: float, spy_final: float,
                 breakdown: list, start_date, end_date, years: float) -> None:
    W  = 66
    HR = "═" * W
    hr = "─" * W

    def fp(v):
        return f"{'+' if v >= 0 else ''}{v:.2f}%"

    def fd(v):
        return f"${v:>12,.2f}"

    def col(strat, bench):
        return f"  {strat:>28}  {bench:>28}"

    print(f"\n{HR}")
    print(f"  LONG-TERM BACKTEST  ·  MA{SHORT_WINDOW}/MA{LONG_WINDOW} Crossover  ·  10 Years")
    print(HR)
    print(f"  Period        : {start_date} → {end_date}  ({years:.1f} years)")
    print(f"  Universe      : {', '.join(TICKERS)}")
    print(f"  Starting Cash : {fd(STARTING_CASH)}")
    print(hr)

    print(f"  {'Metric':<24}  {'Strategy':>20}  {'SPY Buy & Hold':>20}")
    print(f"  {'─'*22:<24}  {'─'*18:>20}  {'─'*14:>20}")

    def row(label, sv, bv):
        print(f"  {label:<24}  {sv:>20}  {bv:>20}")

    row("Final Value",       fd(m["final_value"]),   fd(spy_final))
    row("Total Return",      fp(m["total_ret"]),      fp(spy_ret))
    row("Annualized Return", fp(m["ann_ret"]),        "—")
    row("Sharpe Ratio",      f"{m['sharpe']:.3f}",   "—")
    row("Max Drawdown",      fp(m["max_dd"]),         "—")
    row("Win Rate",
        f"{m['win_rate']:.1f}%  ({m['wins']}W / {m['losses']}L)", "—")
    row("Total Trades",
        f"{m['total_trades']}  ({m['buys']}B · {m['sells']}S)", "1")

    print(hr)
    alpha = m["total_ret"] - spy_ret
    row("Alpha vs SPY", fp(alpha), "+0.00%")
    verdict = "OUTPERFORMED" if alpha >= 0 else "UNDERPERFORMED"
    print(f"  Verdict       : {verdict} SPY by {abs(alpha):.2f}%")
    print(hr)

    # per-ticker table
    print(f"  {'PER-TICKER BREAKDOWN':}")
    print(f"  {'Ticker':<6}  {'Trades':>6}  {'W':>4}  {'L':>4}  "
          f"{'Realized P&L':>14}  {'B&H Return':>12}  {'Strategy vs B&H':>16}")
    print(f"  {'─'*6:<6}  {'─'*5:>6}  {'─'*3:>4}  {'─'*3:>4}  "
          f"{'─'*12:>14}  {'─'*10:>12}  {'─'*15:>16}")

    for r in breakdown:
        pnl_s  = f"+${r['pnl']:,.2f}" if r["pnl"] >= 0 else f"-${abs(r['pnl']):,.2f}"
        bah_s  = fp(r["bah_ret"]) if not math.isnan(r["bah_ret"]) else "—"
        if not math.isnan(r["bah_ret"]):
            bah_dollar = STARTING_CASH / len(TICKERS) * (1 + r["bah_ret"] / 100)
            equal_share_pnl = bah_dollar - STARTING_CASH / len(TICKERS)
            edge = r["pnl"] - equal_share_pnl
            edge_s = f"{'+' if edge >= 0 else ''}{edge:,.0f}"
        else:
            edge_s = "—"
        print(f"  {r['ticker']:<6}  {r['trades']:>6}  {r['wins']:>4}  {r['losses']:>4}  "
              f"{pnl_s:>14}  {bah_s:>12}  {edge_s:>16}")

    print(HR)


# CSV export

def save_csv(trade_log: list) -> None:
    fields = ["date", "action", "ticker", "shares", "price",
              "value", "ma_short", "ma_long", "pnl"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trade_log)
    print(f"  Saved {len(trade_log)} trades → {OUTPUT_CSV}\n")


# entry point

def run():
    print("\nFetching 10 years of daily data…")
    all_data = {}

    for ticker in TICKERS:
        print(f"  {ticker:<6}", end="", flush=True)
        try:
            df = fetch_daily(ticker, period="10y")
            all_data[ticker] = add_ma_signals(df)
            print(f"  {len(df):>5} bars  ({df.index[0].date()} → {df.index[-1].date()})")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"  {BENCHMARK:<6}", end="", flush=True)
    spy = fetch_daily(BENCHMARK, period="10y")
    print(f"  {len(spy):>5} bars  ({spy.index[0].date()} → {spy.index[-1].date()})")

    loaded = [t for t in TICKERS if t in all_data]
    if not loaded:
        print("No data loaded. Exiting.")
        return

    common_dates = sorted(
        set.intersection(*[set(all_data[t].index) for t in loaded])
    )
    start_date, end_date = common_dates[0], common_dates[-1]
    years = (end_date - start_date).days / 365.25

    print(f"\n  Common period : {start_date.date()} → {end_date.date()}"
          f"  ({years:.1f} yrs, {len(common_dates):,} trading days)")

    # SPY benchmark
    spy_dates   = spy.index
    spy_s       = float(spy.loc[spy_dates[spy_dates >= start_date][0], "Close"])
    spy_e       = float(spy.loc[spy_dates[spy_dates <= end_date][-1],  "Close"])
    spy_ret_pct = (spy_e - spy_s) / spy_s * 100
    spy_final   = STARTING_CASH * (spy_e / spy_s)

    print(f"\nRunning backtest over {len(common_dates):,} days × {len(loaded)} tickers…", flush=True)
    final_value, trade_log, daily_vals = simulate(all_data, common_dates)
    print(f"  Done — {len(trade_log)} total trades executed\n")

    m         = compute_metrics(final_value, trade_log, daily_vals, years)
    breakdown = ticker_breakdown(trade_log, all_data, common_dates)

    print_report(m, spy_ret_pct, spy_final, breakdown,
                 start_date.date(), end_date.date(), years)
    save_csv(trade_log)


if __name__ == "__main__":
    run()
