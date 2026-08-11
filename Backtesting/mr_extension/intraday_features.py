"""Intraday features for each example, from Liberator minute bars.

The key job is locating the *setup* extension extreme -- the low (long) or high
(short) that price reverted from at the annotated time of day. This is NOT
generally the daily extreme: T06 (SNDK 2026-07-13) reversed off 1701 in the
morning and only made its 1646 daily low hours later.

Risk is measured from a reference ENTRY, not from the extreme itself. Anchoring
risk at the extreme assumes a fill at the exact low, forces MAE to ~0 by
construction, and inflates R.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
INTRA = ROOT / "data" / "intraday"

# How far around the annotated ToD to accept the reversal extreme.
WINDOW_BEFORE_MIN = 20
WINDOW_AFTER_MIN = 10


def load_minutes(symbol, date):
    f = INTRA / f"{symbol}_{date}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df["ts"] = pd.to_datetime(df["timestamp"]).dt.floor("min")
    df = df.sort_values("ts").reset_index(drop=True)
    return df[["ts", "open", "high", "low", "close", "volume", "spread"]]


def rth(df):
    t = df["ts"].dt.time
    return df[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]


def session_vwap(df):
    """RTH-anchored VWAP series (typical price weighted)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cv = df["volume"].cumsum()
    return (tp * df["volume"]).cumsum() / cv.replace(0, np.nan)


def rnd(x, n=2):
    if x is None or x == "":
        return ""
    if isinstance(x, float) and (pd.isna(x) or np.isinf(x)):
        return ""
    return round(float(x), n)


def to_2m(df):
    """Resample 1-minute bars to 2-minute, aligned to the 09:30 open."""
    g = df.set_index("ts").resample("2min", origin="start_day", offset="30min")
    out = g.agg({"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}).dropna()
    return out.reset_index()


def trigger_2m_break(post, direction):
    """First 2-minute bar whose high (long) / low (short) is taken out by a
    later bar. Returns the fill price (the broken level) and its timestamp."""
    b = to_2m(post)
    if len(b) < 2:
        return None, None
    for k in range(len(b) - 1):
        lvl = b["high"].iloc[k] if direction == "long" else b["low"].iloc[k]
        nxt = b.iloc[k + 1]
        if (direction == "long" and nxt["high"] > lvl) or \
           (direction == "short" and nxt["low"] < lvl):
            return float(lvl), nxt["ts"]
    return None, None


def compute(symbol, date, next_date, direction, tod, atr14):
    df = load_minutes(symbol, date)
    if df is None or df.empty:
        return {"intraday_source": "missing"}

    r = rth(df).reset_index(drop=True)
    if r.empty:
        return {"intraday_source": "no_rth"}
    r["vwap"] = session_vwap(r)

    day = pd.Timestamp(date).date()
    tod_ts = pd.Timestamp(f"{day} {tod}")
    lo_ts = max(tod_ts - pd.Timedelta(minutes=WINDOW_BEFORE_MIN),
                pd.Timestamp(f"{day} 09:30"))
    hi_ts = tod_ts + pd.Timedelta(minutes=WINDOW_AFTER_MIN)

    win = r[(r["ts"] >= lo_ts) & (r["ts"] <= hi_ts)]
    if win.empty:
        return {"intraday_source": "tod_window_empty"}

    if direction == "long":
        k = win["low"].idxmin()
        ext_px = float(win["low"].loc[k])
    else:
        k = win["high"].idxmax()
        ext_px = float(win["high"].loc[k])
    ext_ts = win["ts"].loc[k]
    i = int(r.index[r["ts"] == ext_ts][0])

    sign = 1 if direction == "long" else -1
    stop = ext_px - sign * 0.1 * atr14

    post = r.iloc[i:]                      # from the extreme forward
    opening_print = float(r["open"].iloc[0])
    vwap_at_ext = float(r["vwap"].iloc[i])

    entry_px, entry_ts = trigger_2m_break(post, direction)
    if entry_px is None:
        return {"intraday_source": "no_trigger",
                "ext_time_et": ext_ts.strftime("%H:%M"),
                "ext_price_intraday": rnd(ext_px)}

    risk = abs(entry_px - stop)
    held = post[post["ts"] >= entry_ts]

    fav_from_ext = ((post["high"].max() - ext_px) if direction == "long"
                    else (ext_px - post["low"].min()))
    fav = ((held["high"].max() - entry_px) if direction == "long"
           else (entry_px - held["low"].min()))
    adv = ((entry_px - held["low"].min()) if direction == "long"
           else (held["high"].max() - entry_px))
    mfe_R, mae_R = fav / risk, adv / risk
    stopped = ((held["low"].min() <= stop) if direction == "long"
               else (held["high"].max() >= stop))

    j = (held["high"].idxmax() if direction == "long" else held["low"].idxmin())
    time_to_mfe = (r["ts"].loc[j] - entry_ts).total_seconds() / 60

    def r_at(price):
        return sign * (price - entry_px) / risk

    hit_vwap = held[(held["high"] >= held["vwap"]) if direction == "long"
                    else (held["low"] <= held["vwap"])]
    r_vwap = r_at(float(hit_vwap["vwap"].iloc[0])) if len(hit_vwap) else ""

    first30 = r[r["ts"] < pd.Timestamp(f"{day} 10:00")]

    out = {
        "intraday_source": "liberator_minute",
        "ext_time_et": ext_ts.strftime("%H:%M"),
        "ext_price_intraday": rnd(ext_px),
        "ext_matches_daily_extreme": "",     # filled by caller
        "opening_print": rnd(opening_print),
        "vwap_at_extension": rnd(vwap_at_ext),
        "vwap_dist_at_ext_atr": rnd(abs(vwap_at_ext - ext_px) / atr14, 3),
        "entry_ref_price": rnd(entry_px),
        "entry_ref_time_et": entry_ts.strftime("%H:%M"),
        "entry_slippage_from_ext_atr": rnd(abs(entry_px - ext_px) / atr14, 3),
        "structural_stop": rnd(stop),
        "risk_per_share": rnd(risk),
        "risk_as_pct_atr": rnd(risk / atr14 * 100, 1),
        "stopped_out": bool(stopped),
        "mfe_R": rnd(mfe_R),
        "mae_R": rnd(mae_R),
        # Ceiling: what a fill at the exact extreme would have yielded, same
        # stop. The gap to mfe_R is what the trigger costs you.
        "mfe_R_perfect_entry": rnd(fav_from_ext / abs(ext_px - stop)),
        "time_to_mfe_min": rnd(time_to_mfe, 0),
        "r_at_vwap": rnd(r_vwap),
        "r_at_opening_print": rnd(r_at(opening_print)),
        "r_at_close": rnd(r_at(float(r["close"].iloc[-1]))),
        "vol_first_30m": rnd(first30["volume"].sum(), 0),
        "spread_at_ext": rnd(float(r["spread"].iloc[i]), 3),
    }

    last_hr = r[r["ts"] >= pd.Timestamp(f"{day} 15:00")]
    if len(last_hr):
        px = last_hr["high"].max() if direction == "long" else last_hr["low"].min()
        out["r_at_intraday_swing"] = rnd(r_at(float(px)))

    if next_date:
        nd = load_minutes(symbol, next_date)
        if nd is not None and not nd.empty:
            nr = rth(nd)
            if len(nr):
                px = nr["high"].max() if direction == "long" else nr["low"].min()
                out["r_at_overnight_swing"] = rnd(r_at(float(px)))
    return out
