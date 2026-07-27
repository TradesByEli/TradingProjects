"""
Low Float Gap Backtest
========================
Small-Cap "Low Float Runner" Gap-Day Continuation Study

Objective:
  Answer, split by starting price band (bucketed off PRIOR_CLOSE):
    1. How often does a 100%+ gapper hold its own gap  (close > open)?
    2. How often does it close >=150% above the prior close?
    3. How often does it close >=200% above the prior close?
  Then break those same three hit-rates down by secondary variables (float
  size, catalyst, dilution level, entry type, SSR, reverse split, market
  regime, etc.) drawn from the "Low Float Runner" trading playbook.

Two datasets, NEVER blended:
  Dataset A — journal sample (Backtesting/low_float_seed.csv). Selection-
    biased (only tickers the user already flagged as runners). Used to
    compare hit-rates ACROSS subgroups within itself, not as a base rate.
  Dataset B — broad scan. Every current NASDAQ/NYSE/AMEX common stock
    priced < $30 today, scanned for gap_pct >= 100 days over the trailing
    N years (CONFIG["dataset_b"]["lookback_years"]). This is the closest
    practical approximation to an unconditional base rate available from
    yfinance (see "KNOWN SCOPE LIMITS" below).

Exact formulas (per spec, not approximated):
  prior_close = previous trading day's close
  gap_pct     = (open - prior_close) / prior_close * 100
  close_pct   = (close - prior_close) / prior_close * 100
  qualifies as "100% gapper" if gap_pct >= 100
  held_gap    = close > open
  closed_150  = close_pct >= 150
  closed_200  = close_pct >= 200

Price band (assigned from PRIOR_CLOSE, not the gap-day open):
  Sub-$1   prior_close < 1
  $1-5     1 <= prior_close < 5
  $5-10    5 <= prior_close < 10
  $10+     prior_close >= 10

KNOWN SCOPE LIMITS (flagged here + printed in the run summary, not silently
approximated):
  - Float is CURRENT float (yfinance `.info`), not historical float at the
    time of the event — same limitation liquidity_trap_backtest.py notes
    for market cap. Journal-logged float is used only as a fallback when
    yfinance has none.
  - Premarket volume / float rotation only populate when yfinance's
    intraday history still reaches that date (roughly the trailing ~60
    days) — most historical rows will be blank.
  - Time-of-day of the triggering breakout, %-of-day above VWAP, and
    pre-breakout consolidation tightness are NOT implemented. They need a
    precise, reproducible "this is the breakout moment" rule per entry
    type (PM Breakout vs HOD Breakout vs Opening Range Break, etc.) that
    wasn't specified — rather than guess, these columns are left blank in
    both datasets.
  - Catalyst / dilution-cash-need / entry-type / opp_grade are journal-only
    tags (no news-feed integration here) — blank in Dataset B.
  - Dataset B universe = tickers priced < $30 as of TODAY, scanned over
    the trailing lookback window. Tickers that were sub-$30 historically
    but have since risen above $30, or that have since been delisted, are
    not in the universe. This is a practical scope choice (confirmed with
    the user), not a full-market/full-history claim.
  - SSR status is a derived proxy (Reg SHO Rule 201: prior day's low <=
    90% of the close two days prior), not a sourced exchange SSR flag feed.

Usage:
  py low_float_gap_backtest.py                 # run both datasets
  py low_float_gap_backtest.py --only-a         # journal sample only (fast)
  py low_float_gap_backtest.py --only-b         # broad scan only (slow)
"""

import io
import logging
import re
import sys
import time
import warnings

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────
#  CONFIG — Edit these to customize your backtest
# ─────────────────────────────────────────────────────────
CONFIG = {
    "seed_csv": HERE / "low_float_seed.csv",
    "output_dir": HERE,

    # ── CORE FORMULA THRESHOLDS ───────────────────────────
    "gap_threshold_pct": 100.0,
    "closed_150_pct":    150.0,
    "closed_200_pct":    200.0,

    "price_bands": [
        ("Sub-$1", 0.0,  1.0),
        ("$1-5",   1.0,  5.0),
        ("$5-10",  5.0,  10.0),
        ("$10+",   10.0, float("inf")),
    ],

    "float_buckets": [
        ("<5M",    0.0,        5_000_000.0),
        ("5-15M",  5_000_000.0,  15_000_000.0),
        ("15-50M", 15_000_000.0, 50_000_000.0),
        ("50M+",   50_000_000.0, float("inf")),
    ],

    "day_vol_mult_buckets":    [("<2x", 0, 2), ("2-5x", 2, 5), ("5-10x", 5, 10), ("10x+", 10, float("inf"))],
    "whole_dollar_dist_buckets": [("<5%", 0, 5), ("5-15%", 5, 15), ("15%+", 15, float("inf"))],

    "ssr_decline_pct":                    10.0,  # Reg SHO Rule 201 trigger
    "reverse_split_lookback_days":        90,
    "day2_continuation_lookback_days":    10,    # trading days
    "history_lookback_years":             3,     # per-ticker history pulled for Dataset A / secondary calcs
    "market_regime_proxy":                "IWM",
    "market_regime_flat_band_pct":        0.25,  # |pct chg| below this => "Flat"

    # ── DATASET B (broad scan) ────────────────────────────
    "dataset_b": {
        "enabled": True,
        "lookback_years": 5,
        "price_prefilter_max": 30.0,
        "float_max_shares": 50_000_000,
        "batch_size": 60,
        "batch_pause_sec": 1.5,
        "cache_dir": HERE / "cache",
        "universe_cache_file": HERE / "cache" / "universe_symbols.csv",
        "prefilter_cache_file": HERE / "cache" / "price_prefilter.csv",
        "max_tickers": 25,   # cap for smoke-testing; None = no cap
    },
}
# ─────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────
#  SMALL UTILITIES
# ─────────────────────────────────────────────────────────
def parse_human_number(s):
    """Parse '2M' / '279M' / '1.4B' / '627K' / '—' / '' -> float or None."""
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s in {"-", "—", "N/A", "n/a"} or s.lower() == "nan":
        return None
    s = s.replace(",", "").replace("$", "")
    m = re.match(r"^([\d.]+)\s*([KkMmBb]?)$", s)
    if not m:
        return None
    num = float(m.group(1))
    mult = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9}[m.group(2).upper()]
    return num * mult


def bucket_value(value, bands):
    """Map a numeric value into a labeled band [lo, hi)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    for label, lo, hi in bands:
        if lo <= value < hi:
            return label
    return None


def yesno(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return "Yes" if bool(v) else "No"


# ─────────────────────────────────────────────────────────
#  DATA FETCH HELPERS (yfinance)
# ─────────────────────────────────────────────────────────
def fetch_history(ticker, start, end, retries=2, pause=1.0):
    """Daily OHLCV, split/dividend-adjusted, tz-naive, lower-cased columns."""
    for attempt in range(retries + 1):
        try:
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if df is None or df.empty:
                return None
            df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception:
            if attempt < retries:
                time.sleep(pause)
                continue
            return None


@lru_cache(maxsize=None)
def get_float_shares(ticker):
    """Current float (or shares outstanding fallback). NOT historical."""
    try:
        info = yf.Ticker(ticker).info
        f = info.get("floatShares") or info.get("sharesOutstanding")
        return float(f) if f else None
    except Exception:
        return None


@lru_cache(maxsize=None)
def get_splits(ticker):
    """Cached raw split history as a tuple of (Timestamp, ratio)."""
    try:
        s = yf.Ticker(ticker).splits
        if s is None or len(s) == 0:
            return tuple()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return tuple(zip(s.index.to_list(), s.values.tolist()))
    except Exception:
        return tuple()


def is_recent_reverse_split(ticker, event_date, lookback_days):
    splits = get_splits(ticker)
    if not splits:
        return False
    window_start = pd.Timestamp(event_date) - pd.Timedelta(days=lookback_days)
    window_end = pd.Timestamp(event_date)
    for d, ratio in splits:
        d = pd.Timestamp(d)
        if window_start <= d < window_end and ratio < 1.0:
            return True
    return False


def fetch_premarket_volume(ticker, event_date):
    """Best-effort premarket volume via intraday bars. Only works for
    dates still within yfinance's intraday retention window (~60 days)."""
    try:
        start = pd.Timestamp(event_date)
        end = start + pd.Timedelta(days=1)
        intraday = yf.Ticker(ticker).history(start=start, end=end, interval="5m",
                                              prepost=True, auto_adjust=True)
        if intraday is None or intraday.empty:
            return None
        if intraday.index.tz is not None:
            intraday.index = intraday.index.tz_convert("America/New_York")
        cutoff = pd.Timestamp("09:30").time()
        pm = intraday[intraday.index.time < cutoff]
        if pm.empty:
            return None
        return float(pm["Volume"].sum())
    except Exception:
        return None


def build_iwm_regime(start_date, end_date, cfg):
    """date -> 'Up'/'Down'/'Flat' for the market-regime control variable."""
    proxy = cfg["market_regime_proxy"]
    hist = fetch_history(proxy, pd.Timestamp(start_date) - timedelta(days=10),
                          pd.Timestamp(end_date) + timedelta(days=5))
    if hist is None or hist.empty:
        return {}
    hist = hist.copy()
    hist["pct_chg"] = hist["close"].pct_change() * 100
    band = cfg["market_regime_flat_band_pct"]
    regime = {}
    for ts, row in hist.iterrows():
        chg = row["pct_chg"]
        if pd.isna(chg):
            continue
        if chg > band:
            label = "Up"
        elif chg < -band:
            label = "Down"
        else:
            label = "Flat"
        regime[ts.date()] = label
    return regime


# ─────────────────────────────────────────────────────────
#  CORE METRIC COMPUTATION (shared by Dataset A and B)
# ─────────────────────────────────────────────────────────
def compute_event_metrics(hist, idx, ticker, iwm_regime, cfg):
    """
    hist: full daily OHLCV df (lower-cased cols), ascending, tz-naive index.
    idx : integer row position of the event/gap day.
    Returns a dict of computed fields, or None if idx has no valid prior day.
    """
    if idx <= 0 or idx >= len(hist):
        return None

    row = hist.iloc[idx]
    date = hist.index[idx]
    prior_close = hist["close"].iloc[idx - 1]
    o, h, l, c, v = row["open"], row["high"], row["low"], row["close"], row["volume"]

    if prior_close is None or pd.isna(prior_close) or prior_close <= 0:
        return None

    gap_pct = (o - prior_close) / prior_close * 100
    close_pct = (c - prior_close) / prior_close * 100
    held_gap = bool(c > o)
    closed_150 = bool(close_pct >= cfg["closed_150_pct"])
    closed_200 = bool(close_pct >= cfg["closed_200_pct"])
    qualifies_100 = bool(gap_pct >= cfg["gap_threshold_pct"])

    price_band = bucket_value(prior_close, cfg["price_bands"])

    # 30-trading-day avg volume, STRICTLY before the event day (no look-ahead)
    window = hist["volume"].iloc[max(0, idx - 30):idx]
    avg_vol_30 = window.mean() if len(window) > 0 else np.nan
    day_vol_mult = (v / avg_vol_30) if (avg_vol_30 and avg_vol_30 > 0) else np.nan

    # Did today's high break the high of the previous highest-volume day
    # (within the history actually fetched, strictly before this event)?
    prior_hist = hist.iloc[:idx]
    if len(prior_hist) > 0:
        prior_high_vol_idx = prior_hist["volume"].idxmax()
        prior_high_vol_high = prior_hist.loc[prior_high_vol_idx, "high"]
        broke_prior_high_vol_day_high = bool(h > prior_high_vol_high)
    else:
        broke_prior_high_vol_day_high = None

    # SSR proxy (Reg SHO Rule 201): prior day's low <= 90% of 2-days-ago close
    if idx >= 2:
        low_tm1 = hist["low"].iloc[idx - 1]
        close_tm2 = hist["close"].iloc[idx - 2]
        ssr_active = bool(low_tm1 <= close_tm2 * (1 - cfg["ssr_decline_pct"] / 100))
    else:
        ssr_active = None

    # Day 1 vs Day 2+: any other qualifying gap>=100% day for this ticker
    # within the trailing N trading days (computed from price history so it
    # applies identically to Dataset A and Dataset B).
    lookback_n = cfg["day2_continuation_lookback_days"]
    recent_window = hist.iloc[max(0, idx - lookback_n):idx]
    if len(recent_window) >= 2:
        rw_prior_close = recent_window["close"].shift(1)
        rw_gap = (recent_window["open"] - rw_prior_close) / rw_prior_close * 100
        had_recent_qualifying_gap = bool((rw_gap >= cfg["gap_threshold_pct"]).any())
    else:
        had_recent_qualifying_gap = False
    continuation = "Day 2+" if had_recent_qualifying_gap else "Day 1"

    # Whole-dollar proximity at the open (naturally reduces to the $1 level
    # for sub-$1 names, since round(open) = 1 in that case).
    nearest_dollar = round(o)
    dollar_dist = o - nearest_dollar
    dollar_dist_pct = (dollar_dist / o * 100) if o else np.nan

    regime = iwm_regime.get(date.date())

    return {
        "ticker": ticker,
        "date": date.date(),
        "prior_close": round(float(prior_close), 4),
        "open": round(float(o), 4),
        "high": round(float(h), 4),
        "low": round(float(l), 4),
        "close": round(float(c), 4),
        "volume": int(v) if pd.notna(v) else None,
        "gap_pct": round(float(gap_pct), 2),
        "close_pct": round(float(close_pct), 2),
        "qualifies_100": qualifies_100,
        "held_gap": held_gap,
        "closed_150": closed_150,
        "closed_200": closed_200,
        "price_band": price_band,
        "avg_vol_30": round(float(avg_vol_30), 0) if pd.notna(avg_vol_30) else None,
        "day_vol_avg30_mult": round(float(day_vol_mult), 2) if pd.notna(day_vol_mult) else None,
        "broke_prior_high_vol_day_high": broke_prior_high_vol_day_high,
        "ssr_active": ssr_active,
        "continuation": continuation,
        "open_dollar_dist": round(float(dollar_dist), 4),
        "open_dollar_dist_pct": round(float(dollar_dist_pct), 2) if pd.notna(dollar_dist_pct) else None,
        "market_regime": regime,
    }


def enrich_with_shared_secondary_fields(m, ticker, event_date, cfg):
    """Fields computed identically for both datasets: float bucket, reverse
    split flag, day-vol-mult bucket, whole-dollar bucket."""
    f = get_float_shares(ticker)
    m["float_shares"] = f
    m["float_source"] = "yfinance_current" if f is not None else None
    m["float_bucket"] = bucket_value(f, cfg["float_buckets"]) if f is not None else None

    m["recent_reverse_split"] = is_recent_reverse_split(ticker, event_date, cfg["reverse_split_lookback_days"])

    dvm = m.get("day_vol_avg30_mult")
    m["day_vol_mult_bucket"] = bucket_value(dvm, cfg["day_vol_mult_buckets"]) if dvm is not None else None

    ddp = m.get("open_dollar_dist_pct")
    m["near_whole_dollar_bucket"] = bucket_value(abs(ddp), cfg["whole_dollar_dist_buckets"]) if ddp is not None else None
    return m


# ─────────────────────────────────────────────────────────
#  DATASET A — JOURNAL SAMPLE
# ─────────────────────────────────────────────────────────
def run_dataset_a(cfg):
    print("\n" + "=" * 100)
    print("  DATASET A — journal sample (conditional, selection-biased)")
    print("=" * 100)

    seed = pd.read_csv(cfg["seed_csv"], dtype=str).fillna("")
    seed["date"] = pd.to_datetime(seed["date"])

    regime = build_iwm_regime(seed["date"].min(), seed["date"].max(), cfg)

    records = []
    for _, row in seed.iterrows():
        ticker = row["ticker"].strip()
        event_date = row["date"]
        print(f"  ↳ {ticker} ({event_date.date()})...", end=" ", flush=True)

        start = event_date - timedelta(days=cfg["history_lookback_years"] * 365 + 60)
        end = event_date + timedelta(days=5)
        hist = fetch_history(ticker, start, end)
        if hist is None:
            print("no data, skipping.")
            continue

        target = pd.Timestamp(event_date.date())
        if target not in hist.index:
            print("event date not in history (delisted/renamed/no trade), skipping.")
            continue
        idx = hist.index.get_loc(target)

        m = compute_event_metrics(hist, idx, ticker, regime, cfg)
        if m is None:
            print("insufficient prior-day data, skipping.")
            continue

        m = enrich_with_shared_secondary_fields(m, ticker, event_date, cfg)

        # journal-only fields
        m["journal_label"] = row["label"] or None
        m["opp_grade"] = row["opp_grade"] or None
        m["catalyst"] = row["catalyst"] or None
        m["trade_types"] = row["trade_types"] or None
        m["dilution_cash_need"] = row["dilution_cash_need"] or None
        m["journal_float_raw"] = row["float"] or None
        m["journal_day_vol_raw"] = row["day_vol"] or None
        m["journal_avg_vol_30_raw"] = row["avg_vol_30d"] or None

        # float fallback to journal manual entry if yfinance had none
        if m["float_shares"] is None:
            jf = parse_human_number(row["float"])
            if jf is not None:
                m["float_shares"] = jf
                m["float_source"] = "journal_manual"
                m["float_bucket"] = bucket_value(jf, cfg["float_buckets"])

        # best-effort premarket volume / float rotation
        pm_vol = fetch_premarket_volume(ticker, event_date)
        m["premarket_volume"] = pm_vol
        m["pm_vol_over_5m"] = (pm_vol > 5_000_000) if pm_vol is not None else None
        m["float_rotation"] = (pm_vol / m["float_shares"]) if (pm_vol is not None and m["float_shares"]) else None

        # not implemented anywhere (documented gap) — keep columns present for schema consistency
        m["breakout_time_bucket"] = None
        m["pct_day_above_vwap"] = None
        m["consolidation_tightness_pct"] = None

        records.append(m)
        status = "qualifies (100%+)" if m["qualifies_100"] else "below 100% threshold"
        print(f"gap {m['gap_pct']:+.0f}% | close {m['close_pct']:+.0f}% | {status}")
        time.sleep(0.2)

    df = pd.DataFrame(records)
    print(f"\n  Dataset A: {len(df)}/{len(seed)} journal rows fetched successfully.")
    return df


# ─────────────────────────────────────────────────────────
#  DATASET B — BROAD SCAN
# ─────────────────────────────────────────────────────────
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
EXCLUDE_NAME_KEYWORDS = ["Warrant", "Right", "Unit", "Preferred", "Depositary", "Notes", "Debenture"]


def fetch_universe_symbols(cache_file):
    if cache_file.exists():
        return pd.read_csv(cache_file)["symbol"].tolist()

    def _get(url):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        lines = r.text.strip().splitlines()
        if lines and lines[-1].startswith("File Creation Time"):
            lines = lines[:-1]
        return "\n".join(lines)

    nasdaq_df = pd.read_csv(io.StringIO(_get(NASDAQ_URL)), sep="|", keep_default_na=False)
    other_df = pd.read_csv(io.StringIO(_get(OTHER_URL)), sep="|", keep_default_na=False)

    def clean(df, symbol_col, name_col, etf_col, test_col):
        df = df[df[test_col].astype(str).str.upper() == "N"]
        if etf_col in df.columns:
            df = df[df[etf_col].astype(str).str.upper() == "N"]
        keep_mask = ~df[name_col].astype(str).str.contains(
            "|".join(EXCLUDE_NAME_KEYWORDS), case=False, na=False)
        df = df[keep_mask]
        symbols = df[symbol_col].astype(str).str.strip()
        symbols = symbols[~symbols.str.contains(r"[.\$]", regex=True)]
        return symbols.tolist()

    nasdaq_syms = clean(nasdaq_df, "Symbol", "Security Name", "ETF", "Test Issue")
    other_syms = clean(other_df, "ACT Symbol", "Security Name", "ETF", "Test Issue")

    all_syms = sorted(set(nasdaq_syms) | set(other_syms))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": all_syms}).to_csv(cache_file, index=False)
    return all_syms


def _split_ticker_frame(raw, chunk, ticker):
    """Pull one ticker's sub-frame out of a (possibly multi-ticker) yf.download() result."""
    try:
        if len(chunk) == 1:
            sub = raw
        else:
            if ticker not in raw.columns.get_level_values(0):
                return None
            sub = raw[ticker]
        if sub is None or sub.empty:
            return None
        sub = sub.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        sub.columns = [c.lower() for c in sub.columns]
        if sub.index.tz is not None:
            sub.index = sub.index.tz_localize(None)
        return sub
    except Exception:
        return None


def prefilter_by_price(symbols, b_cfg):
    cache_file = b_cfg["prefilter_cache_file"]
    if cache_file.exists():
        return pd.read_csv(cache_file)["symbol"].tolist()

    batch_size = b_cfg["batch_size"] * 3
    chunks = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    keep = []
    print(f"  Price prefilter: scanning {len(symbols)} tickers in {len(chunks)} batches "
          f"(keep if last close < ${b_cfg['price_prefilter_max']:.0f})...")
    for ci, chunk in enumerate(chunks):
        try:
            raw = yf.download(chunk, period="5d", group_by="ticker",
                               threads=True, progress=False, auto_adjust=True)
        except Exception as e:
            print(f"    batch {ci + 1}/{len(chunks)}: download failed ({e}), skipping")
            continue
        for t in chunk:
            sub = _split_ticker_frame(raw, chunk, t)
            if sub is None or sub.empty:
                continue
            last_close = sub["close"].dropna()
            if not last_close.empty and last_close.iloc[-1] < b_cfg["price_prefilter_max"]:
                keep.append(t)
        print(f"    batch {ci + 1}/{len(chunks)} done — {len(keep)} kept so far")
        time.sleep(b_cfg["batch_pause_sec"])

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": keep}).to_csv(cache_file, index=False)
    return keep


def batch_fetch_history(tickers, start, end, b_cfg):
    cache_dir = b_cfg["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    batch_size = b_cfg["batch_size"]
    chunks = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

    histories = {}
    print(f"  Downloading {b_cfg['lookback_years']}y history for {len(tickers)} tickers "
          f"in {len(chunks)} batches...")
    for ci, chunk in enumerate(chunks):
        cache_file = cache_dir / f"hist_batch_{ci:04d}.pkl"
        if cache_file.exists():
            try:
                histories.update(pd.read_pickle(cache_file))
                print(f"    batch {ci + 1}/{len(chunks)}: loaded from cache")
                continue
            except Exception:
                pass

        try:
            raw = yf.download(chunk, start=start, end=end, group_by="ticker",
                               threads=True, progress=False, auto_adjust=True)
        except Exception as e:
            print(f"    batch {ci + 1}/{len(chunks)}: download failed ({e}), skipping")
            continue

        batch_data = {t: _split_ticker_frame(raw, chunk, t) for t in chunk}
        try:
            pd.to_pickle(batch_data, cache_file)
        except Exception:
            pass

        histories.update(batch_data)
        n_ok = sum(1 for v in batch_data.values() if v is not None)
        print(f"    batch {ci + 1}/{len(chunks)}: {n_ok}/{len(chunk)} tickers OK")
        time.sleep(b_cfg["batch_pause_sec"])

    return histories


def run_dataset_b(cfg):
    b = cfg["dataset_b"]
    if not b["enabled"]:
        return pd.DataFrame()

    print("\n" + "=" * 100)
    print("  DATASET B — broad scan (unconditional base rate)")
    print("=" * 100)

    symbols = fetch_universe_symbols(b["universe_cache_file"])
    print(f"  Universe: {len(symbols)} candidate common-stock tickers "
          f"(NASDAQ+NYSE+AMEX, ETFs/warrants/units/preferreds excluded)")
    if b["max_tickers"]:
        symbols = symbols[:b["max_tickers"]]

    sub_universe = prefilter_by_price(symbols, b)
    print(f"  {len(sub_universe)} tickers currently priced < ${b['price_prefilter_max']:.0f} "
          f"(current-price prefilter — see scope notes)")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=int(b["lookback_years"] * 365.25) + 30)

    regime = build_iwm_regime(start_date, end_date, cfg)
    histories = batch_fetch_history(sub_universe, start_date, end_date, b)

    events = []
    for ticker, hist in histories.items():
        if hist is None or len(hist) < 2:
            continue
        prior_close = hist["close"].shift(1)
        gap_pct = (hist["open"] - prior_close) / prior_close * 100
        qualifying_positions = np.where(gap_pct.values >= cfg["gap_threshold_pct"])[0]
        for pos in qualifying_positions:
            m = compute_event_metrics(hist, int(pos), ticker, regime, cfg)
            if m is not None:
                events.append(m)

    print(f"  {len(events)} qualifying 100%+ gap-day events found across "
          f"{len(sub_universe)} scanned tickers")

    df = pd.DataFrame(events)
    if df.empty:
        return df

    # Enrich only the tickers that actually produced a hit (much smaller set
    # than the full universe) with float / splits / bucket fields.
    unique_tickers = df["ticker"].unique().tolist()
    print(f"  Enriching {len(unique_tickers)} unique hit tickers with float / split data...")
    enriched_rows = []
    for _, r in df.iterrows():
        row = r.to_dict()
        row = enrich_with_shared_secondary_fields(row, row["ticker"], row["date"], cfg)
        enriched_rows.append(row)
    df = pd.DataFrame(enriched_rows)

    before = len(df)
    max_float = b["float_max_shares"]
    df["passes_float_filter"] = df["float_shares"].apply(
        lambda f: (f is None) or pd.isna(f) or (f < max_float))
    df = df[df["passes_float_filter"]].copy()
    print(f"  Float < {max_float / 1e6:.0f}M filter (best-effort, current float only): "
          f"{len(df)}/{before} events kept ({before - len(df)} excluded; events with unknown "
          f"float are KEPT, not excluded, per best-effort policy)")

    # journal-only fields — not available for the broad scan
    for col in ["opp_grade", "catalyst", "trade_types", "dilution_cash_need",
                "premarket_volume", "pm_vol_over_5m", "float_rotation",
                "breakout_time_bucket", "pct_day_above_vwap", "consolidation_tightness_pct"]:
        df[col] = None

    return df


# ─────────────────────────────────────────────────────────
#  SUMMARY TABLES
# ─────────────────────────────────────────────────────────
def summarize_by_band(df, cfg):
    q = df[df["qualifies_100"]]
    rows = []
    for label, _, _ in cfg["price_bands"]:
        sub = q[q["price_band"] == label]
        n = len(sub)
        rows.append({
            "price_band": label,
            "trades": n,
            "held_gap_pct": round(sub["held_gap"].mean() * 100, 1) if n else None,
            "closed_150_pct": round(sub["closed_150"].mean() * 100, 1) if n else None,
            "closed_200_pct": round(sub["closed_200"].mean() * 100, 1) if n else None,
        })
    n_all = len(q)
    rows.append({
        "price_band": "ALL",
        "trades": n_all,
        "held_gap_pct": round(q["held_gap"].mean() * 100, 1) if n_all else None,
        "closed_150_pct": round(q["closed_150"].mean() * 100, 1) if n_all else None,
        "closed_200_pct": round(q["closed_200"].mean() * 100, 1) if n_all else None,
    })
    return pd.DataFrame(rows)


def summarize_by_column(df, col, multi_label=False):
    q = df[df["qualifies_100"]].copy()
    if col not in q.columns or q[col].isna().all():
        return None

    if multi_label:
        q[col] = q[col].fillna("").astype(str).str.split(";")
        q = q.explode(col)
        q[col] = q[col].str.strip()
        q = q[q[col] != ""]
        if q.empty:
            return None

    rows = []
    for val, sub in q.groupby(col, dropna=True):
        n = len(sub)
        rows.append({
            "subgroup": val,
            "trades": n,
            "held_gap_pct": round(sub["held_gap"].mean() * 100, 1),
            "closed_150_pct": round(sub["closed_150"].mean() * 100, 1),
            "closed_200_pct": round(sub["closed_200"].mean() * 100, 1),
        })
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("trades", ascending=False).reset_index(drop=True)


SECONDARY_VARS = [
    ("float_bucket", "Float Size Bucket", False),
    ("pm_vol_over_5m", "Premarket Vol > 5M shares", False),
    ("day_vol_mult_bucket", "Day Vol vs 30d AvgVol Multiple", False),
    ("catalyst", "Catalyst Category (multi-label)", True),
    ("dilution_cash_need", "Dilution / Cash Need Level", False),
    ("continuation", "Day 1 vs Day 2+ Continuation", False),
    ("broke_prior_high_vol_day_high", "Broke Prior High-Volume Day's High", False),
    ("ssr_active", "SSR Status", False),
    ("recent_reverse_split", "Recent Reverse Split (trailing 90d)", False),
    ("trade_types", "Entry / Breakout Type (multi-label)", True),
    ("market_regime", "Market Regime (IWM that day)", False),
    ("near_whole_dollar_bucket", "Whole-Dollar Proximity at Open", False),
    ("opp_grade", "Opportunity Grade (journal)", False),
]


def prep_bool_cols(df):
    df = df.copy()
    for col in ["pm_vol_over_5m", "broke_prior_high_vol_day_high", "ssr_active", "recent_reverse_split"]:
        if col in df.columns:
            df[col] = df[col].apply(yesno)
    return df


# ─────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────
def plot_band_summary(summary_df, dataset_label, out_path):
    bands = summary_df[summary_df["price_band"] != "ALL"].reset_index(drop=True)
    if bands.empty or bands["trades"].sum() == 0:
        return
    x = np.arange(len(bands))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, bands["held_gap_pct"], width, label="Held Gap (close>open)", color="#2196F3")
    ax.bar(x, bands["closed_150_pct"], width, label="Closed >=150%", color="#4CAF50")
    ax.bar(x + width, bands["closed_200_pct"], width, label="Closed >=200%", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.price_band}\n(n={r.trades})" for r in bands.itertuples()])
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title(f"Hit Rates by Price Band — {dataset_label}", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    for i, r in bands.iterrows():
        for off, key in zip([-width, 0, width], ["held_gap_pct", "closed_150_pct", "closed_200_pct"]):
            val = r[key]
            if val is not None and not pd.isna(val):
                ax.text(i + off, val + 1, f"{val:.0f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_secondary_breakdowns(breakdown_dict, dataset_label, out_path):
    available = {k: v for k, v in breakdown_dict.items() if v is not None and len(v) > 0}
    if not available:
        return
    n = len(available)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    width = 0.25
    for ax, (name, tbl) in zip(axes, available.items()):
        x = np.arange(len(tbl))
        ax.bar(x - width, tbl["held_gap_pct"], width, label="Held Gap", color="#2196F3")
        ax.bar(x, tbl["closed_150_pct"], width, label=">=150%", color="#4CAF50")
        ax.bar(x + width, tbl["closed_200_pct"], width, label=">=200%", color="#FF9800")
        labels = [f"{r.subgroup}\n(n={r.trades})" for r in tbl.itertuples()]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_ylabel("Hit Rate (%)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25, axis="y")
    for ax in axes[len(available):]:
        ax.axis("off")
    fig.suptitle(f"Secondary Variable Breakdown — {dataset_label}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────
#  PRINT + SAVE (per dataset)
# ─────────────────────────────────────────────────────────
def print_dataset_summary(df, dataset_label, note, cfg):
    print("\n" + "=" * 100)
    print(f"  {dataset_label}")
    print(f"  ({note})")
    print("=" * 100)
    if df.empty:
        print("  No data.")
        return None, {}

    total = len(df)
    qualifying = df[df["qualifies_100"]]
    print(f"  Rows fetched: {total}  |  Qualifying 100%+ gap events: {len(qualifying)}  "
          f"({total - len(qualifying)} fetched rows did not meet the 100% gap threshold and "
          f"are excluded from all hit-rate stats below)")

    band_summary = summarize_by_band(df, cfg)
    print("\n  --- Hit Rates by Price Band (assigned from prior_close) ---")
    print(band_summary.to_string(index=False))

    print("\n  --- Secondary Variable Breakdown ---")
    breakdowns = {}
    for col, name, multi in SECONDARY_VARS:
        tbl = summarize_by_column(df, col, multi_label=multi)
        breakdowns[name] = tbl
        print(f"\n  · {name}:")
        if tbl is None:
            print("      [not available for this dataset — no data populated, see scope notes]")
        else:
            print(tbl.to_string(index=False))

    return band_summary, breakdowns


def print_scope_notes(cfg):
    b = cfg["dataset_b"]
    print("\n" + "=" * 100)
    print("  Dataset A and Dataset B are NEVER blended. Dataset A's hit-rates are conditional on")
    print("  the ticker already being a recognized runner in the journal (selection bias) — use it")
    print("  only to compare subgroups against each other. Dataset B is the closest practical")
    print("  approximation to a true base rate, subject to the scope notes below.")
    print("=" * 100)
    print("  Scope notes / known data gaps:")
    print("   - Float is CURRENT float from yfinance (journal manual entry as fallback for Dataset A")
    print("     only), not historical float at the time of the event.")
    print("   - Premarket volume / float rotation only populate when yfinance's intraday history")
    print("     still covers that date (roughly the trailing ~60 days) — most rows will be blank.")
    print("   - Breakout time-of-day, %-of-day above VWAP, and consolidation tightness are NOT")
    print("     implemented (no reproducible breakout-detection rule was specified) — blank in both.")
    print("   - Catalyst / dilution-cash-need / entry-type / opp_grade are journal-only tags and are")
    print("     blank in Dataset B (no news-feed integration in this script).")
    print(f"   - Dataset B universe = current NASDAQ/NYSE/AMEX common stocks priced < "
          f"${b['price_prefilter_max']:.0f} TODAY, scanned over the trailing {b['lookback_years']} years.")
    print("     Tickers that were sub-$30 historically but have since risen above $30, or have since")
    print("     been delisted, are excluded — a practical scope limit agreed with the user, not a")
    print("     full-market/full-history claim.")
    print("=" * 100 + "\n")


def process_and_output(df_a, df_b, cfg):
    out_dir = cfg["output_dir"]

    if not df_a.empty:
        df_a_p = prep_bool_cols(df_a)
        band_a, breakdown_a = print_dataset_summary(
            df_a_p, "DATASET A — JOURNAL SAMPLE",
            "conditional, selection-biased — do NOT treat as a base rate", cfg)
        if band_a is not None:
            plot_band_summary(band_a, "Dataset A (journal sample)",
                               out_dir / "low_float_gap_dataset_a_band_chart.png")
            plot_secondary_breakdowns(breakdown_a, "Dataset A (journal sample)",
                                      out_dir / "low_float_gap_dataset_a_secondary_chart.png")
        df_a.to_csv(out_dir / "low_float_gap_dataset_a_trade_log.csv", index=False)
        print("\n  Saved -> low_float_gap_dataset_a_trade_log.csv, "
              "low_float_gap_dataset_a_band_chart.png, low_float_gap_dataset_a_secondary_chart.png")

    if not df_b.empty:
        df_b_p = prep_bool_cols(df_b)
        band_b, breakdown_b = print_dataset_summary(
            df_b_p, "DATASET B — BROAD SCAN",
            "unconditional base rate — true incidence, not selection-biased", cfg)
        if band_b is not None:
            plot_band_summary(band_b, "Dataset B (broad scan)",
                               out_dir / "low_float_gap_dataset_b_band_chart.png")
            plot_secondary_breakdowns(breakdown_b, "Dataset B (broad scan)",
                                      out_dir / "low_float_gap_dataset_b_secondary_chart.png")
        df_b.to_csv(out_dir / "low_float_gap_dataset_b_trade_log.csv", index=False)
        print("\n  Saved -> low_float_gap_dataset_b_trade_log.csv, "
              "low_float_gap_dataset_b_band_chart.png, low_float_gap_dataset_b_secondary_chart.png")
    elif cfg["dataset_b"]["enabled"]:
        print("\n  Dataset B produced no qualifying events (or was skipped this run).")

    print_scope_notes(cfg)


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    only_a = "--only-a" in sys.argv
    only_b = "--only-b" in sys.argv
    run_a = not only_b
    run_b = not only_a

    df_a = pd.DataFrame()
    df_b = pd.DataFrame()

    if run_a:
        df_a = run_dataset_a(CONFIG)

    if run_b and CONFIG["dataset_b"]["enabled"]:
        df_b = run_dataset_b(CONFIG)

    process_and_output(df_a, df_b, CONFIG)
    print("\n  Done.\n")
