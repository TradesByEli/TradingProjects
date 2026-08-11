"""Phase 3 execution grid: entry triggers, targets, and trailing methods.

Computed inline during the scan so the minute bars are only fetched once.
Everything is vectorized -- no per-bar Python loops -- so the grid costs little
next to the network fetch.

All results are in R off the structural stop (0.1 ATR beyond the extension
extreme), which is fixed per signal. Changing the entry changes the risk, so
each trigger has its own R denominator: a later entry is a worse price but a
tighter stop, and the grid is what reveals whether that trade-off pays.
"""
import numpy as np
import pandas as pd


def resample(df, rule):
    return (df.set_index("ts")
              .resample(rule, origin="start_day", offset="30min")
              .agg({"open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"})
              .dropna().reset_index())


# ------------------------------------------------------------------ triggers

def trig_bar_break(bars, direction):
    """First bar whose high (long) / low (short) the next bar takes out."""
    if len(bars) < 2:
        return None, None
    lvl = (bars["high"] if direction == "long" else bars["low"]).values[:-1]
    nxt = (bars["high"] if direction == "long" else bars["low"]).values[1:]
    hit = (nxt > lvl) if direction == "long" else (nxt < lvl)
    k = int(np.argmax(hit)) if hit.any() else None
    if k is None:
        return None, None
    return float(lvl[k]), bars["ts"].iloc[k + 1]


def trig_ema_close(bars, direction, span):
    """First bar closing back through its own EMA."""
    if len(bars) < 2:
        return None, None
    ema = bars["close"].ewm(span=span, adjust=False).mean().values
    c = bars["close"].values
    hit = (c > ema) if direction == "long" else (c < ema)
    hit[0] = False
    if not hit.any():
        return None, None
    k = int(np.argmax(hit))
    return float(c[k]), bars["ts"].iloc[k]


TRIGGERS = {
    "immediate": None,                       # fill at the extreme (ceiling)
    "brk2m": ("bar", "2min"),
    "brk5m": ("bar", "5min"),
    "ema9_2m": ("ema", "2min", 9),
    "ema9_5m": ("ema", "5min", 9),
}


def entry_for(name, spec, post, direction, ext_px, ext_ts, cache):
    """`cache` holds FULL-session resampled bars, built once per symbol and
    shared across both directions -- resampling per signal was half the scan's
    compute. Bars are sliced to the post-extreme window here."""
    if spec is None:
        return ext_px, ext_ts
    bars = cache[spec[1]]
    bars = bars[bars["ts"] >= ext_ts]
    if spec[0] == "bar":
        return trig_bar_break(bars, direction)
    return trig_ema_close(bars, direction, spec[2])


# -------------------------------------------------------------------- trails

def trail_bar(bars, direction, start_ts):
    """Exit when a bar takes out the prior bar's low (long) / high (short)."""
    b = bars[bars["ts"] >= start_ts]
    if len(b) < 3:
        return None
    ref = (b["low"] if direction == "long" else b["high"]).values
    test = (b["low"] if direction == "long" else b["high"]).values
    prev = ref[:-1]
    cur = test[1:]
    hit = (cur < prev) if direction == "long" else (cur > prev)
    if not hit.any():
        return None
    k = int(np.argmax(hit))
    return float(prev[k])


def trail_ema(bars, direction, start_ts, span):
    """Exit when a bar closes back through the EMA."""
    b = bars[bars["ts"] >= start_ts]
    if len(b) < 3:
        return None
    ema = b["close"].ewm(span=span, adjust=False).mean().values
    c = b["close"].values
    hit = (c < ema) if direction == "long" else (c > ema)
    hit[0] = False
    if not hit.any():
        return None
    k = int(np.argmax(hit))
    return float(c[k])


TRAILS = {
    "trail2m": ("bar", "2min"),
    "trail5m": ("bar", "5min"),
    "trail15m": ("bar", "15min"),
    "ema9_5m": ("ema", "5min", 9),
    "ema21_5m": ("ema", "5min", 21),
}


# ---------------------------------------------------------------------- main

def evaluate(post, r, direction, ext_px, ext_ts, stop, targets, cache):
    """Full grid for one signal.

    `targets` maps a target name to its price. Returns a flat dict of
    R-multiples keyed <trigger>__<exit>.
    """
    sign = 1 if direction == "long" else -1
    out = {}

    ts_all = post["ts"].values
    high_all = post["high"].values
    low_all = post["low"].values
    close_all = post["close"].values

    for tname, spec in TRIGGERS.items():
        entry, ets = entry_for(tname, spec, post, direction, ext_px, ext_ts, cache)
        if entry is None:
            continue
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        # Hot loop -- operate on numpy arrays. Doing this with pandas filtering
        # per trigger x target was the scan's dominant compute cost.
        s = np.searchsorted(ts_all, np.datetime64(ets), side="left")
        H, L, C = high_all[s:], low_all[s:], close_all[s:]
        if len(H) < 2:
            continue

        out[f"{tname}__entry"] = round(entry, 4)
        out[f"{tname}__risk_pct_atr"] = round(risk / abs(ext_px - stop) * 10, 1)

        lo, hi = L.min(), H.max()
        stopped = (lo <= stop) if direction == "long" else (hi >= stop)
        out[f"{tname}__stopped"] = bool(stopped)
        out[f"{tname}__mfe_R"] = round(
            ((hi - entry) if direction == "long" else (entry - lo)) / risk, 3)
        out[f"{tname}__mae_R"] = round(
            ((entry - lo) if direction == "long" else (hi - entry)) / risk, 3)

        # first bar index where the stop broke (len(H) if never)
        stop_hits = (L <= stop) if direction == "long" else (H >= stop)
        k_stop = int(np.argmax(stop_hits)) if stop_hits.any() else len(H)

        # --- fixed targets. If the stop was hit first, the trade is -1R. ---
        for gname, gpx in targets.items():
            if gpx is None or (isinstance(gpx, float) and np.isnan(gpx)):
                continue
            reach = (H >= gpx) if direction == "long" else (L <= gpx)
            k_tgt = int(np.argmax(reach)) if reach.any() else len(H)
            if k_tgt == len(H):
                px = stop if stopped else float(C[-1])
            else:
                px = stop if k_stop < k_tgt else gpx
            out[f"{tname}__{gname}"] = round(sign * (px - entry) / risk, 3)

        # --- trailing methods ---
        for rname, rspec in TRAILS.items():
            bars = cache[rspec[1]]
            px = (trail_bar(bars, direction, ets) if rspec[0] == "bar"
                  else trail_ema(bars, direction, ets, rspec[2]))
            if px is None:
                px = float(C[-1])
            # a trail can never do better than the stop if the stop broke first
            if direction == "long":
                px = max(px, stop) if not stopped else min(px, max(px, stop))
            out[f"{tname}__{rname}"] = round(sign * (px - entry) / risk, 3)

    return out
