"""
Liquidity Trap — Backtest
==========================
Micro-Cap Gap Continuation: Inside Day → Day 3 Breakout

Setup Logic:
  DAY 1  — Catalyst / Big Move Day
    · Open  > Close[2] * 1.20           (gaps up 20%+ from 2 days prior)
    · Close > Close[2] * 1.10           (closes 10%+ above 2-day-ago close)
    · Close > Low + (High - Low) * 0.60 (strong close, upper 40% of range)
    · Volume > AvgVol(30) * 5           (5× average volume)
    · Volume > 20,000,000               (at least 20M shares traded)

  DAY 2  — Inside Day / Consolidation
    · High  < High[1]                   (inside day high)
    · Low   > Low[1]                    (inside day low)
    · Volume < 5,000,000                (quiet volume)
    · Volume < AvgVol(30) * 1.5         (below 1.5× average)
    · Close < Close[1] * 1.025          (not running away, ≤2.5% above D1 close)

  FILTERS (applied on Day 2 / entry evaluation)
    · Close   > 1.00                    (not sub-penny)
    · Close   < 30.00                   (low-priced stock)
    · AvgVol(30) > 500,000              (minimum liquidity)
    · Market Cap < $150M                (micro-cap)

  DAY 3  — Entry
    · Enter LONG when price breaks above Day 2 High
    · Entry price = Day 2 High (intraday breakout fill)

  EXIT OPTIONS (configurable below)
    · 'close'        — exit at Day 3 close
    · 'target_stop'  — fixed % profit target + stop loss from entry

Usage:
  1. Edit CONFIG below (tickers, dates, exit rules)
  2. pip install yfinance pandas numpy matplotlib
  3. python inside_day_backtest.py

Note on Market Cap:
  yfinance provides current market cap, not historical. For micro-cap
  screening this is used as a rough filter. For tickers that have grown
  significantly since your test period, consider removing this filter.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  CONFIG — Edit these to customize your backtest
# ─────────────────────────────────────────────────────────
CONFIG = {
    # Tickers to scan — swap for your watchlist or use a screener export
    "tickers": ["FFIE", "MULN", "CENN", "HPNN", "MRIN", "PROG", "ATER", "BBIG"],

    "start_date": "2021-01-01",
    "end_date": datetime.today().strftime("%Y-%m-%d"),

    # ── DAY 1 THRESHOLDS ──────────────────────────────────
    "d1_gap_min_pct":        20.0,   # Open must be > Close[2] * (1 + this/100)
    "d1_close_min_pct":      10.0,   # Close must be > Close[2] * (1 + this/100)
    "d1_close_range_pct":    60.0,   # Close must be in upper X% of High-Low range
    "d1_vol_mult":            5.0,   # Volume > AvgVol(30) * this
    "d1_vol_min":        20_000_000, # Absolute minimum volume on Day 1

    # ── DAY 2 THRESHOLDS ──────────────────────────────────
    "d2_vol_max":         5_000_000, # Volume < this
    "d2_vol_mult_max":        1.5,   # Volume < AvgVol(30) * this
    "d2_close_max_pct":       2.5,   # Close < Close[1] * (1 + this/100)

    # ── OPTIONAL FILTERS (applied on Day 2) ───────────────
    "filter_price_min":       1.00,  # Close > this
    "filter_price_max":      30.00,  # Close < this
    "filter_avg_vol_min":   500_000, # AvgVol(30) > this
    "filter_mktcap_max":  150_000_000, # Market cap < this (uses current cap)

    # ── EXIT RULES ────────────────────────────────────────
    # 'close'       → exit at Day 3 close
    # 'target_stop' → fixed % target + stop from entry price
    "exit_method":        "close",
    "profit_target_pct":    15.0,   # % gain target (used if exit_method='target_stop')
    "stop_loss_pct":         7.0,   # % loss stop  (used if exit_method='target_stop')

    # ── POSITION SIZING ───────────────────────────────────
    "initial_capital":    100_000,
    "shares_per_trade":     1_000,  # Fixed share size per trade
}
# ─────────────────────────────────────────────────────────


def fetch_data(ticker: str) -> tuple[pd.DataFrame, float]:
    """Download daily OHLCV data + current market cap from Yahoo Finance."""
    t = yf.Ticker(ticker)
    df = t.history(start=CONFIG["start_date"], end=CONFIG["end_date"], auto_adjust=True)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.tz_localize(None)

    info = t.info
    mktcap = info.get("marketCap", 0) or 0
    return df, mktcap


def avg_vol(df: pd.DataFrame, window: int = 30) -> pd.Series:
    return df["volume"].rolling(window).mean()


def identify_setups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scan for Day 1 + Day 2 setups and mark Day 3 entry signals.
    Returns df with extra columns for analysis.
    """
    df = df.copy()
    av = avg_vol(df, 30)
    df["avg_vol_30"] = av

    # ── DAY 1 FLAGS (evaluated on the Day 1 row itself) ──
    gap_pct     = CONFIG["d1_gap_min_pct"]   / 100
    close_pct   = CONFIG["d1_close_min_pct"] / 100
    range_pct   = CONFIG["d1_close_range_pct"] / 100

    d1_gap_up   = df["open"]   > df["close"].shift(2) * (1 + gap_pct)
    d1_strong   = df["close"]  > df["close"].shift(2) * (1 + close_pct)
    d1_hi_close = df["close"]  > df["low"] + (df["high"] - df["low"]) * range_pct
    d1_vol_mult = df["volume"] > av * CONFIG["d1_vol_mult"]
    d1_vol_min  = df["volume"] > CONFIG["d1_vol_min"]

    is_d1 = d1_gap_up & d1_strong & d1_hi_close & d1_vol_mult & d1_vol_min

    # ── DAY 2 FLAGS (evaluated on the Day 2 row, comparing to Day 1) ──
    close_max_pct = CONFIG["d2_close_max_pct"] / 100

    d2_inside_high = df["high"]   < df["high"].shift(1)
    d2_inside_low  = df["low"]    > df["low"].shift(1)
    d2_vol_abs     = df["volume"] < CONFIG["d2_vol_max"]
    d2_vol_mult    = df["volume"] < av * CONFIG["d2_vol_mult_max"]
    d2_close_calm  = df["close"]  < df["close"].shift(1) * (1 + close_max_pct)

    # Day 2 also requires that the PREVIOUS day was a valid Day 1
    is_d2 = (d2_inside_high & d2_inside_low & d2_vol_abs &
             d2_vol_mult & d2_close_calm & is_d1.shift(1))

    # ── OPTIONAL FILTERS (evaluated on Day 2 row) ──
    f_price_min  = df["close"] > CONFIG["filter_price_min"]
    f_price_max  = df["close"] < CONFIG["filter_price_max"]
    f_avg_vol    = av > CONFIG["filter_avg_vol_min"]

    is_d2_filtered = is_d2 & f_price_min & f_price_max & f_avg_vol

    # ── RECORD SETUP INFO ON THE DAY 3 ROW ──
    prev_is_d2   = is_d2_filtered.shift(1).fillna(False)

    df["setup_active"]  = prev_is_d2
    df["d1_date"]       = np.where(prev_is_d2, df.index.to_series().shift(2), pd.NaT)
    df["d2_date"]       = np.where(prev_is_d2, df.index.to_series().shift(1), pd.NaT)

    # Day 1 reference prices (for chart drawing)
    df["d1_high"]       = np.where(prev_is_d2, df["high"].shift(2),   np.nan)
    df["d1_low"]        = np.where(prev_is_d2, df["low"].shift(2),    np.nan)
    df["d1_close"]      = np.where(prev_is_d2, df["close"].shift(2),  np.nan)
    df["d1_volume"]     = np.where(prev_is_d2, df["volume"].shift(2), np.nan)

    # Day 2 reference prices (the inside day)
    df["d2_high"]       = np.where(prev_is_d2, df["high"].shift(1),   np.nan)
    df["d2_low"]        = np.where(prev_is_d2, df["low"].shift(1),    np.nan)
    df["d2_close"]      = np.where(prev_is_d2, df["close"].shift(1),  np.nan)
    df["d2_volume"]     = np.where(prev_is_d2, df["volume"].shift(1), np.nan)
    df["d2_avg_vol"]    = np.where(prev_is_d2, av.shift(1),           np.nan)

    # Day 3 entry: long when price trades up to / through Day 2 high
    df["entry_price"] = np.where(
        prev_is_d2 & (df["high"] >= df["d2_high"]),
        df["d2_high"],
        np.nan
    )

    return df


def calculate_trade_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute exit prices and P&L for each triggered Day 3 entry."""
    trades = df[df["entry_price"].notna()].copy()
    trades = trades[trades["close"] > 0]

    if CONFIG["exit_method"] == "close":
        trades["exit_price"] = trades["close"]
        trades["exit_reason"] = "EOD Close"
    else:
        target_m = 1 + CONFIG["profit_target_pct"] / 100
        stop_m   = 1 - CONFIG["stop_loss_pct"]     / 100

        def sim_exit(row):
            target = row["entry_price"] * target_m
            stop   = row["entry_price"] * stop_m
            # Assume worst case: if both levels hit, take the stop
            hit_stop   = row["low"]  <= stop
            hit_target = row["high"] >= target
            if hit_stop:
                return stop, "Stop Loss"
            elif hit_target:
                return target, "Target Hit"
            else:
                return row["close"], "EOD Close"

        trades[["exit_price", "exit_reason"]] = trades.apply(
            lambda r: pd.Series(sim_exit(r)), axis=1
        )

    shares = CONFIG["shares_per_trade"]
    trades["return_pct"] = (trades["exit_price"] - trades["entry_price"]) / trades["entry_price"] * 100
    trades["pnl_dollar"]  = (trades["exit_price"] - trades["entry_price"]) * shares
    trades["win"]         = trades["return_pct"] > 0
    return trades


def compute_metrics(trades: pd.DataFrame, ticker: str) -> dict:
    if len(trades) == 0:
        return {"ticker": ticker, "trades": 0}

    wins   = trades[trades["win"]]
    losses = trades[~trades["win"]]

    avg_w = wins["return_pct"].mean()   if len(wins)   > 0 else 0.0
    avg_l = losses["return_pct"].mean() if len(losses) > 0 else 0.0
    rr    = round(abs(avg_w / avg_l), 2) if avg_l != 0 else "N/A"

    # Equity curve
    capital = CONFIG["initial_capital"]
    equity  = [capital]
    for pnl in trades["pnl_dollar"]:
        equity.append(equity[-1] + pnl)
    eq = pd.Series(equity)
    peak  = eq.cummax()
    maxdd = ((eq - peak) / peak * 100).min()

    return {
        "ticker":       ticker,
        "trades":       len(trades),
        "wins":         int(trades["win"].sum()),
        "losses":       int((~trades["win"]).sum()),
        "win_rate":     round(trades["win"].mean() * 100, 1),
        "avg_return":   round(trades["return_pct"].mean(), 2),
        "avg_win":      round(avg_w, 2),
        "avg_loss":     round(avg_l, 2),
        "rr_ratio":     rr,
        "total_return": round(trades["return_pct"].sum(), 2),
        "total_pnl":    round(trades["pnl_dollar"].sum(), 2),
        "max_drawdown": round(maxdd, 2),
        "best_trade":   round(trades["return_pct"].max(), 2),
        "worst_trade":  round(trades["return_pct"].min(), 2),
        "equity_curve": eq,
        "trades_df":    trades,
    }


def plot_results(all_metrics: list):
    """Generate a multi-panel chart for all tickers with valid trades."""
    valid = [m for m in all_metrics if m["trades"] > 0]
    if not valid:
        print("  No trades to plot.")
        return

    n   = len(valid)
    fig = plt.figure(figsize=(15, 5.5 * n))
    fig.suptitle(
        "Liquidity Trap  |  Micro-Cap Inside Day → Day 3 Breakout",
        fontsize=14, fontweight="bold", y=1.01
    )

    for i, m in enumerate(valid):
        trades = m["trades_df"]
        equity = m["equity_curve"]
        cap    = CONFIG["initial_capital"]

        gs  = gridspec.GridSpec(n, 3, figure=fig)
        ax1 = fig.add_subplot(gs[i, 0:2])   # equity (wide)
        ax2 = fig.add_subplot(gs[i, 2])     # return distribution

        # — Equity curve —
        ax1.plot(equity.values, color="#2196F3", lw=1.8, zorder=3)
        ax1.fill_between(range(len(equity)), equity.values, cap,
                         where=equity.values >= cap, alpha=0.15, color="#2196F3")
        ax1.fill_between(range(len(equity)), equity.values, cap,
                         where=equity.values < cap, alpha=0.15, color="#f44336")
        ax1.axhline(cap, color="gray", lw=0.8, linestyle="--")
        ax1.set_title(
            f"{m['ticker']}  |  {m['trades']} trades  |  "
            f"Win rate: {m['win_rate']}%  |  Max DD: {m['max_drawdown']}%  |  "
            f"Total P&L: ${m['total_pnl']:+,.0f}",
            fontweight="bold", fontsize=10
        )
        ax1.set_xlabel("Trade #"); ax1.set_ylabel("Portfolio Value ($)")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.grid(alpha=0.25)

        # — Per-trade returns bar chart —
        colors = ["#4CAF50" if r > 0 else "#f44336" for r in trades["return_pct"]]
        ax2.bar(range(len(trades)), trades["return_pct"].values, color=colors, alpha=0.8)
        ax2.axhline(0, color="black", lw=0.8)
        ax2.axhline(trades["return_pct"].mean(), color="#FF9800", lw=1.2, linestyle="--",
                    label=f"Avg: {trades['return_pct'].mean():.1f}%")
        ax2.set_title("Per-Trade Returns (%)", fontweight="bold", fontsize=10)
        ax2.set_xlabel("Trade #"); ax2.set_ylabel("Return (%)")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.25)

    plt.tight_layout()
    out = "/sessions/upbeat-serene-mccarthy/mnt/Liquidity Trap Backtest/backtest_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Chart saved → backtest_results.png")


def print_summary(all_metrics: list):
    print("\n" + "=" * 100)
    print("  LIQUIDITY TRAP — MICRO-CAP INSIDE DAY → DAY 3 BREAKOUT")
    print(f"  Period : {CONFIG['start_date']}  →  {CONFIG['end_date']}")
    print(f"  Entry  : Break above Day 2 High  |  Exit: {CONFIG['exit_method'].upper()}", end="")
    if CONFIG["exit_method"] == "target_stop":
        print(f"  (+{CONFIG['profit_target_pct']}% target / -{CONFIG['stop_loss_pct']}% stop)", end="")
    print(f"\n  Size   : {CONFIG['shares_per_trade']:,} shares/trade  |  Capital: ${CONFIG['initial_capital']:,}")
    print("=" * 100)

    hdr = ("  {:<8}  {:>7}  {:>6}  {:>7}  {:>10}  {:>9}  {:>10}  {:>5}  "
           "{:>11}  {:>9}  {:>8}  {:>8}").format(
        "Ticker", "Trades", "Wins", "Losses", "Win Rate %",
        "Avg Ret %", "Avg Win %", "R:R", "Total P&L $", "Max DD %", "Best %", "Worst %"
    )
    print(hdr)
    print("  " + "-" * 96)

    for m in all_metrics:
        if m["trades"] == 0:
            print(f"  {m['ticker']:<8}  — no setups matched filters")
            continue
        row = ("  {:<8}  {:>7}  {:>6}  {:>7}  {:>9.1f}%  {:>8.2f}%  {:>9.2f}%  {:>5}  "
               "{:>+10,.0f}  {:>8.2f}%  {:>7.2f}%  {:>7.2f}%").format(
            m["ticker"], m["trades"], m["wins"], m["losses"],
            m["win_rate"], m["avg_return"], m["avg_win"],
            str(m["rr_ratio"]), m["total_pnl"],
            m["max_drawdown"], m["best_trade"], m["worst_trade"]
        )
        print(row)

    # Aggregate across all tickers
    valid = [m for m in all_metrics if m["trades"] > 0]
    if len(valid) > 1:
        total_trades = sum(m["trades"] for m in valid)
        total_wins   = sum(m["wins"]   for m in valid)
        total_pnl    = sum(m["total_pnl"] for m in valid)
        all_ret      = pd.concat([m["trades_df"]["return_pct"] for m in valid])
        print("  " + "-" * 96)
        print(f"  {'TOTAL':<8}  {total_trades:>7}  {total_wins:>6}  {total_trades-total_wins:>7}  "
              f"{total_wins/total_trades*100:>9.1f}%  {all_ret.mean():>8.2f}%  "
              f"{'':>9}  {'':>5}  {total_pnl:>+10,.0f}")

    print("=" * 100)


def save_trade_log(all_metrics: list):
    """Save full trade-by-trade log to CSV."""
    frames = []
    for m in all_metrics:
        if m["trades"] == 0:
            continue
        cols = [
            "d1_high", "d1_low", "d1_close", "d1_volume",
            "d2_high", "d2_low", "d2_close", "d2_volume", "d2_avg_vol",
            "entry_price", "exit_price", "return_pct", "pnl_dollar", "win"
        ]
        if "exit_reason" in m["trades_df"].columns:
            cols.append("exit_reason")
        t = m["trades_df"][cols].copy()
        t.insert(0, "ticker", m["ticker"])
        frames.append(t)

    if frames:
        log = pd.concat(frames)
        log.index.name = "date_day3"
        out = "/sessions/upbeat-serene-mccarthy/mnt/Liquidity Trap Backtest/trade_log.csv"
        log.to_csv(out)
        print(f"  Trade log saved → trade_log.csv  ({len(log)} trades)")


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  Scanning {len(CONFIG['tickers'])} tickers...")

    all_metrics = []

    for ticker in CONFIG["tickers"]:
        print(f"  ↳ {ticker}...", end=" ", flush=True)
        try:
            df, mktcap = fetch_data(ticker)

            if len(df) < 35:
                print("not enough data, skipping.")
                all_metrics.append({"ticker": ticker, "trades": 0})
                continue

            # Market cap filter (current cap as proxy)
            if mktcap > CONFIG["filter_mktcap_max"] and mktcap > 0:
                print(f"market cap ${mktcap/1e6:.0f}M > filter, skipping.")
                all_metrics.append({"ticker": ticker, "trades": 0})
                continue

            df     = identify_setups(df)
            trades = calculate_trade_returns(df)
            m      = compute_metrics(trades, ticker)

            setup_count = df["setup_active"].sum()
            trigger_count = len(trades)
            print(f"{setup_count} setups found → {trigger_count} triggered (price reached D2 high)")
            all_metrics.append(m)

        except Exception as e:
            print(f"error: {e}")
            all_metrics.append({"ticker": ticker, "trades": 0})

    print_summary(all_metrics)
    plot_results(all_metrics)
    save_trade_log(all_metrics)
    print("\n  Done! Open backtest_results.png and trade_log.csv in your workspace.\n")
