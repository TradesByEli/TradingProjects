"""Phase 2 detectors: key levels and ATR-normalized extension.

Design follows the decisions locked with the user:
  * Levels come from BOTH an MA detector and a pivot detector, tagged by type,
    so MA-based and pivot-based mean reversion can be compared directly.
  * Extension is ATR-normalized on multiple clocks (intraday from the open and
    from the prior close; multi-day over 1/3/5 days).
  * Every threshold is a parameter, not a constant, so Phase 3 can sweep them.

This module is pure computation over daily bars -- no I/O, no network -- so it
can be unit-checked against the 21 known examples before any scan runs.
"""
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

@dataclass
class Params:
    # --- level detection ---
    ma_lengths: tuple = (5, 9, 20, 50, 100, 200)
    level_tol_atr: float = 0.35       # how close price must come to "hit" a level
    pivot_lookback: int = 130         # sessions searched for swing pivots
    pivot_fractal_n: int = 3          # bars each side required for a swing point
    pivot_touch_tol_atr: float = 0.35
    pivot_min_touches: int = 2        # "tested multiple times on the daily"
    pivot_min_age_days: int = 2

    # --- EXHAUSTION cluster: daily-ATR extension (any ONE clock qualifies) ---
    # Calibrated on the deck's exhaustion examples, where the move is large
    # relative to daily ATR.
    min_ext_from_open_atr: float = 0.75
    min_ext_from_prior_close_atr: float = 1.0
    min_move_atr_3d: float = 2.0
    min_move_atr_5d: float = 2.5
    min_consecutive_days: int = 3

    # --- OPEN-DRIVE cluster: intraday velocity ---
    # These setups travel only 0.08-0.98 daily-ATR, so daily ATR is the wrong
    # yardstick. What distinguishes them is SPEED: a large fraction of the
    # typical opening range covered in the first few minutes. Measured on
    # minute bars in the confirmation stage, not on daily bars.
    open_drive_cutoff_et: str = "10:30"   # setups after this are exhaustion-type
    min_speed_ratio: float = 4.0          # travel rate vs. average minute pace

    # --- daily PREFILTER (deliberately loose) ---
    # The daily stage only decides which symbol-days are worth pulling minute
    # bars for. The real setup test runs on minute bars. Keeping this loose
    # avoids baking the exhaustion-cluster assumptions into candidate
    # selection, which would silently drop the open-drive cluster.
    prefilter_min_ext_from_open_atr: float = 0.30
    prefilter_min_ext_from_prior_close_atr: float = 0.35
    prefilter_min_move_atr_3d: float = 0.70
    prefilter_min_move_atr_5d: float = 0.80
    prefilter_min_consecutive_days: int = 2

    # --- liquidity / tradability ---
    min_dollar_vol: float = 20_000_000
    min_price: float = 5.0
    min_atr_pct: float = 2.0          # daily ATR as % of price

    # --- session ---
    scan_start_et: str = "09:30"
    scan_end_et: str = "12:00"


DEFAULT = Params()


# --------------------------------------------------------------------------
# Daily feature computation
# --------------------------------------------------------------------------

def add_daily_features(df, p=DEFAULT):
    """Attach ATR, MAs, Bollinger, range/volume averages to a symbol's bars.

    Anything describing 'what was knowable before today' is shifted by one bar.
    """
    df = df.sort_values("date").reset_index(drop=True)
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    pc = c.shift(1)

    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean().shift(1)
    df["prior_close"] = pc

    for n in p.ma_lengths:
        df[f"sma{n}"] = c.rolling(n).mean().shift(1)   # yesterday's MA value

    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    df["bb_pos"] = ((c - (ma20 - 2 * sd20)) / (4 * sd20)).clip(-1, 2)

    df["avg_vol20"] = v.rolling(20).mean().shift(1)
    df["dollar_vol"] = c * v
    df["avg_dollar_vol20"] = (c * v).rolling(20).mean().shift(1)
    df["atr_pct"] = df["atr14"] / pc * 100
    return df


def consecutive_dir_days(closes, direction):
    """Consecutive same-direction closes ending at the last element."""
    n = 0
    for j in range(len(closes) - 1, 0, -1):
        d = closes[j] - closes[j - 1]
        if (direction == "long" and d < 0) or (direction == "short" and d > 0):
            n += 1
        else:
            break
    return n


# --------------------------------------------------------------------------
# Level detector
# --------------------------------------------------------------------------

def swing_pivots(df, i, direction, p=DEFAULT):
    """Fractal swing highs/lows in the lookback window before bar i.

    For a long setup we want SUPPORT below, i.e. prior swing lows (and prior
    highs that have become support). Returns a list of candidate prices.
    """
    lo = max(0, i - p.pivot_lookback)
    w = df.iloc[lo:i]                      # strictly before today
    if len(w) < 2 * p.pivot_fractal_n + 1:
        return []

    n = p.pivot_fractal_n
    highs, lows = w["high"].values, w["low"].values
    out = []
    for k in range(n, len(w) - n):
        if highs[k] == max(highs[k - n:k + n + 1]):
            out.append(("swing_high", float(highs[k]), len(w) - k))
        if lows[k] == min(lows[k - n:k + n + 1]):
            out.append(("swing_low", float(lows[k]), len(w) - k))
    return out


def count_touches(df, i, price, atr, p=DEFAULT):
    """How many prior bars traded into a band around this price."""
    lo = max(0, i - p.pivot_lookback)
    w = df.iloc[lo:i]
    tol = p.pivot_touch_tol_atr * atr
    return int((((w["low"] - tol) <= price) & ((w["high"] + tol) >= price)).sum())


def candidate_levels(df, i, direction, p=DEFAULT):
    """All qualifying levels for bar i, each tagged by type.

    Returns list of dicts: {level_type, level_detail, level_price, age_days,
    touch_count}. For a long setup only levels at or below the prior close are
    support; for a short setup only levels at or above it are resistance.
    """
    atr = df["atr14"].iloc[i]
    ref = df["prior_close"].iloc[i]
    if pd.isna(atr) or pd.isna(ref) or atr <= 0:
        return []

    out = []

    # --- moving averages ---
    for n in p.ma_lengths:
        v = df[f"sma{n}"].iloc[i]
        if pd.isna(v):
            continue
        if (direction == "long" and v > ref) or (direction == "short" and v < ref):
            continue          # wrong side to act as support/resistance
        out.append({"level_type": "MA", "level_detail": f"SMA{n}",
                    "level_price": float(v), "age_days": "", "touch_count": ""})

    # --- swing pivots ---
    seen = []
    for kind, price, age in swing_pivots(df, i, direction, p):
        if (direction == "long" and price > ref) or (direction == "short" and price < ref):
            continue
        if age < p.pivot_min_age_days:
            continue
        # collapse near-identical pivots into one level
        if any(abs(price - s) <= p.pivot_touch_tol_atr * atr for s in seen):
            continue
        touches = count_touches(df, i, price, atr, p)
        if touches < p.pivot_min_touches:
            continue
        seen.append(price)
        out.append({"level_type": "pivot", "level_detail": kind,
                    "level_price": price, "age_days": age,
                    "touch_count": touches})
    return out


def nearest_level_hit(levels, ext_px, atr, p=DEFAULT):
    """The qualifying level closest to the extension extreme.

    Returns a level_type of "none" when the extreme reached no level rather
    than returning None. Extensions into new highs/lows (E10 CRCL, T01 SNDK)
    genuinely have no prior level -- the nearest pivot sits 1.6-1.9 ATR away.
    Emitting them tagged lets the backtest measure whether hitting a level
    actually improves the edge, instead of assuming it does.
    """
    best = None
    for lv in levels:
        d = abs(ext_px - lv["level_price"])
        if d <= p.level_tol_atr * atr and (best is None or d < best[0]):
            best = (d, lv)
    if best is None:
        nearest = min((abs(ext_px - lv["level_price"]) for lv in levels),
                      default=None)
        return {"level_type": "none", "level_detail": "", "level_price": "",
                "age_days": "", "touch_count": "",
                "level_dist_atr": round(nearest / atr, 3) if nearest else ""}
    lv = dict(best[1])
    lv["level_dist_atr"] = round(best[0] / atr, 3)
    return lv


# --------------------------------------------------------------------------
# Extension detector
# --------------------------------------------------------------------------

def extension_metrics(df, i, ext_px, direction):
    """ATR-normalized extension on every clock, plus the context measures."""
    atr = df["atr14"].iloc[i]
    o = df["open"].iloc[i]
    pc = df["prior_close"].iloc[i]
    m = {
        "ext_from_open_atr": abs(ext_px - o) / atr,
        "ext_from_prior_close_atr": abs(ext_px - pc) / atr,
        "gap_pct": (o / pc - 1) * 100,
    }
    for n in (1, 3, 5):
        if i - n >= 0:
            base = df["close"].iloc[i - n]
            m[f"move_atr_{n}d"] = abs(ext_px - base) / atr
            m[f"move_pct_{n}d"] = (ext_px / base - 1) * 100
    m["consecutive_dir_days"] = consecutive_dir_days(
        df["close"].iloc[max(0, i - 12):i].values, direction)
    m["bb_pos_prior"] = df["bb_pos"].iloc[i - 1] if i > 0 else np.nan
    return m


def extension_qualifies(m, p=DEFAULT, prefilter=False):
    """Any single clock clearing its threshold is enough. Returns the list of
    clocks that fired, so the scan records WHY each signal qualified.

    With prefilter=True the loose candidate-selection thresholds are used
    instead of the strict exhaustion-cluster ones.
    """
    t = ((p.prefilter_min_ext_from_open_atr,
          p.prefilter_min_ext_from_prior_close_atr,
          p.prefilter_min_move_atr_3d, p.prefilter_min_move_atr_5d,
          p.prefilter_min_consecutive_days) if prefilter else
         (p.min_ext_from_open_atr, p.min_ext_from_prior_close_atr,
          p.min_move_atr_3d, p.min_move_atr_5d, p.min_consecutive_days))
    fired = []
    if m.get("ext_from_open_atr", 0) >= t[0]:
        fired.append("open")
    if m.get("ext_from_prior_close_atr", 0) >= t[1]:
        fired.append("prior_close")
    if m.get("move_atr_3d", 0) >= t[2]:
        fired.append("3d")
    if m.get("move_atr_5d", 0) >= t[3]:
        fired.append("5d")
    if m.get("consecutive_dir_days", 0) >= t[4]:
        fired.append("consec")
    return fired


def intraday_velocity(minutes, ext_ts, ext_px, direction, atr14, p=DEFAULT):
    """Open-drive extension measure: how FAST price reached the extreme,
    relative to the average pace of a normal session.

        speed_ratio = (travel / minutes_elapsed) / (ATR / 390)

    Normalizing by daily ATR and elapsed time avoids the circularity of using
    the same day's opening range as the unit: a large reversal inflates that
    range and perversely suppresses the score (T10 reversed 0.20 ATR in two
    minutes yet scored only 0.39 under the range-based measure).

    Returns None when the extreme is too late to be an open-drive event.
    `minutes` is the RTH minute frame with columns ts/open/high/low/close.
    """
    if minutes is None or minutes.empty or not atr14 or pd.isna(atr14):
        return None
    day = minutes["ts"].iloc[0].date()
    if ext_ts > pd.Timestamp(f"{day} {p.open_drive_cutoff_et}"):
        return None

    o = float(minutes["open"].iloc[0])
    elapsed = max((ext_ts - minutes["ts"].iloc[0]).total_seconds() / 60, 1.0)
    travel_atr = abs(ext_px - o) / atr14
    baseline = 1.0 / 390.0                      # ATR per minute, average day
    return {
        "speed_ratio": (travel_atr / elapsed) / baseline,
        "minutes_to_extreme": elapsed,
        "travel_atr": travel_atr,
    }


def tradable(df, i, p=DEFAULT):
    """Liquidity and volatility floor -- keeps the scan on names you'd trade."""
    return (df["avg_dollar_vol20"].iloc[i] >= p.min_dollar_vol
            and df["prior_close"].iloc[i] >= p.min_price
            and df["atr_pct"].iloc[i] >= p.min_atr_pct)
