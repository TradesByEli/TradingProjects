"""Phase 2 scan: run the detectors directly on minute bars, day by day.

No daily prefilter. Ranking candidates on daily bars was tested and abandoned:
the 21 known examples sit at a median rank of 155 out of ~1000 per day, and no
scheme (extension size, level tightness, ATR%) kept more than 9 of 21 at an
affordable cut. These setups are not identifiable from daily bars.

So every symbol-day in the universe is evaluated on minute bars. Bars are
streamed -- fetched per day, reduced to signals, and discarded -- so disk use
stays small despite covering ~575,000 symbol-days.

Resumable: completed dates are recorded, so re-running continues where it left
off. Signals append to scan_results.csv.
"""
import subprocess
import sys
from io import StringIO
from pathlib import Path
import pandas as pd
import numpy as np

import detectors as D
import grid as GRID

ROOT = Path(__file__).parent
HIST = ROOT / "data" / "daily_hist"
UNIV = ROOT / "data" / "universe" / "universe500.csv"
OUTDIR = ROOT / "data" / "scan_days"      # one file per day
OUT = ROOT / "scan_results.csv"           # concatenated at the end

# Canonical column order. Grid columns are absent on days where a trigger never
# fires, so appending raw frames to one CSV produced rows of varying width and
# silently misaligned data. Every day's frame is reindexed to this list.
BASE_COLS = [
    "symbol", "date", "direction", "cluster", "why", "ext_time_et", "ext_price",
    "minutes_to_extreme", "speed_ratio", "level_type", "level_detail",
    "level_price", "level_dist_atr", "level_touch_count", "ext_from_open_atr",
    "ext_from_prior_close_atr", "move_atr_3d", "move_atr_5d",
    "consecutive_dir_days", "gap_pct", "atr14", "atr_pct", "avg_dollar_vol20",
    "entry_ref_price", "entry_ref_time_et", "structural_stop", "risk_per_share",
    "risk_as_pct_atr", "stopped_out", "mfe_R", "mae_R", "r_at_vwap",
    "r_at_opening_print", "r_at_prior_close", "r_at_intraday_swing",
    "r_at_close", "r_at_overnight_swing",
]
_TARGETS = ["tgt_vwap", "tgt_open_print", "tgt_prior_close", "tgt_close",
            "tgt_intraday_swing", "tgt_overnight"]
GRID_COLS = [
    f"{t}__{s}"
    for t in GRID.TRIGGERS
    for s in (["entry", "risk_pct_atr", "stopped", "mfe_R", "mae_R"]
              + _TARGETS + list(GRID.TRAILS))
]
COLUMNS = BASE_COLS + GRID_COLS

SCAN_START, SCAN_END = "2022-01-03", "2026-08-03"
SCAN_WINDOW_END = "12:00"        # setups must form by here
p = D.DEFAULT


# ---------------------------------------------------------------- daily prep

def build_daily_context(symbols):
    """Per-symbol daily features + fractal pivots, indexed by date."""
    frames = [pd.read_csv(f) for f in sorted(HIST.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["symbol"].isin(symbols)]
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    df = (df[["symbol", "date", "open", "high", "low", "close", "volume"]]
          .drop_duplicates(["symbol", "date"])
          .sort_values(["symbol", "date"]))

    ctx = {}
    for sym, g in df.groupby("symbol", sort=False):
        if len(g) < 220:
            continue
        g = D.add_daily_features(g.reset_index(drop=True), p)
        h, l = g["high"].values, g["low"].values
        n = p.pivot_fractal_n
        N = len(g)
        hi = np.zeros(N, bool)
        lo = np.zeros(N, bool)
        for k in range(n, N - n):
            if h[k] == h[k - n:k + n + 1].max():
                hi[k] = True
            if l[k] == l[k - n:k + n + 1].min():
                lo[k] = True
        g["_pos"] = np.arange(N)
        ctx[sym] = {"df": g, "hi": hi, "lo": lo,
                    "idx": {d: i for i, d in enumerate(g["date"])}}
    return ctx


def levels_for(c, i, direction):
    """MA + pivot levels available on bar i, on the correct side."""
    g, atr = c["df"], c["df"]["atr14"].iloc[i]
    ref = g["prior_close"].iloc[i]
    out = []
    for n in p.ma_lengths:
        v = g[f"sma{n}"].iloc[i]
        if pd.isna(v):
            continue
        if (direction == "long" and v > ref) or (direction == "short" and v < ref):
            continue
        out.append({"level_type": "MA", "level_detail": f"SMA{n}",
                    "level_price": float(v), "age_days": "", "touch_count": ""})

    lo_w = max(0, i - p.pivot_lookback)
    h, l = g["high"].values, g["low"].values
    piv = np.concatenate([h[lo_w:i][c["hi"][lo_w:i]], l[lo_w:i][c["lo"][lo_w:i]]])
    if len(piv):
        side = piv[piv <= ref] if direction == "long" else piv[piv >= ref]
        seen = []
        tol = p.pivot_touch_tol_atr * atr
        for price in sorted(side, reverse=(direction == "short")):
            if any(abs(price - s) <= tol for s in seen):
                continue
            touches = int((((l[lo_w:i] - tol) <= price)
                           & ((h[lo_w:i] + tol) >= price)).sum())
            if touches < p.pivot_min_touches:
                continue
            seen.append(price)
            out.append({"level_type": "pivot", "level_detail": "swing",
                        "level_price": float(price), "age_days": "",
                        "touch_count": touches})
    return out


# --------------------------------------------------------------- minute prep

def fetch_day(symbols, date):
    r = subprocess.run(
        ["kti", "liberator", "query", "--dataset", "minute_bars",
         "--date", date, "--symbols", ",".join(symbols),
         "--fields", "symbol,timestamp,open,high,low,close,volume",
         "--format", "csv"],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or r.stdout.count("\n") < 10:
        return None
    df = pd.read_csv(StringIO(r.stdout))
    df["ts"] = pd.to_datetime(df["timestamp"]).dt.floor("min")
    return df


def analyse(sym, date, c, m, results):
    """Evaluate one symbol-day for both directions."""
    d = pd.Timestamp(date)
    i = c["idx"].get(d)
    if i is None or i < 1:
        return
    g = c["df"]
    atr = g["atr14"].iloc[i]
    if pd.isna(atr) or atr <= 0 or not D.tradable(g, i, p):
        return

    t = m["ts"].dt.time
    r = m[(t >= pd.Timestamp("09:30").time())
          & (t < pd.Timestamp("16:00").time())].sort_values("ts")
    if len(r) < 60:
        return
    r = r.reset_index(drop=True)
    tp = (r["high"] + r["low"] + r["close"]) / 3
    r["vwap"] = (tp * r["volume"]).cumsum() / r["volume"].cumsum().replace(0, np.nan)

    win_end = pd.Timestamp(f"{d.date()} {SCAN_WINDOW_END}")
    win = r[r["ts"] <= win_end]
    if win.empty:
        return

    # Resample once per symbol and share across both directions -- doing this
    # per signal was roughly half the scan's compute time.
    bar_cache = {rule: GRID.resample(r, rule) for rule in ("2min", "5min", "15min")}

    for direction in ("long", "short"):
        k = win["low"].idxmin() if direction == "long" else win["high"].idxmax()
        ext_px = float(win["low"].loc[k] if direction == "long"
                       else win["high"].loc[k])
        ext_ts = win["ts"].loc[k]
        j = int(r.index[r["ts"] == ext_ts][0])

        # ---- classify ----
        mm = D.extension_metrics(g, i, ext_px, direction)
        strict = D.extension_qualifies(mm, p)
        vel = D.intraday_velocity(r, ext_ts, ext_px, direction, atr, p)
        if strict:
            cluster, why = "exhaustion", ",".join(strict)
        elif vel and vel["speed_ratio"] >= p.min_speed_ratio:
            cluster, why = "open_drive", f"speed={vel['speed_ratio']:.1f}"
        else:
            continue

        lv = D.nearest_level_hit(levels_for(c, i, direction), ext_px, atr, p)

        # ---- reference entry + outcomes ----
        post = r.iloc[j:].reset_index(drop=True)
        entry_px, entry_ts = D_trigger(post, direction)
        if entry_px is None:
            continue
        sign = 1 if direction == "long" else -1
        stop = ext_px - sign * 0.1 * atr
        risk = abs(entry_px - stop)
        if risk <= 0:
            continue
        held = post[post["ts"] >= entry_ts]
        if held.empty:
            continue

        fav = ((held["high"].max() - entry_px) if direction == "long"
               else (entry_px - held["low"].min()))
        adv = ((entry_px - held["low"].min()) if direction == "long"
               else (held["high"].max() - entry_px))
        stopped = ((held["low"].min() <= stop) if direction == "long"
                   else (held["high"].max() >= stop))

        def r_at(px):
            return round(sign * (px - entry_px) / risk, 3)

        hit_v = held[(held["high"] >= held["vwap"]) if direction == "long"
                     else (held["low"] <= held["vwap"])]
        last_hr = r[r["ts"] >= pd.Timestamp(f"{d.date()} 15:00")]
        nxt = g.iloc[i + 1] if i + 1 < len(g) else None

        # ---- full execution grid (triggers x targets x trails) ----
        targets = {
            "tgt_vwap": float(r["vwap"].iloc[j]) if not pd.isna(r["vwap"].iloc[j]) else None,
            "tgt_open_print": float(r["open"].iloc[0]),
            "tgt_prior_close": float(g["prior_close"].iloc[i]),
            "tgt_close": float(r["close"].iloc[-1]),
        }
        if len(last_hr):
            targets["tgt_intraday_swing"] = float(
                last_hr["high"].max() if direction == "long" else last_hr["low"].min())
        if nxt is not None:
            targets["tgt_overnight"] = float(
                nxt["high"] if direction == "long" else nxt["low"])
        grid_cols = GRID.evaluate(post, r, direction, ext_px, ext_ts, stop,
                                  targets, bar_cache)

        results.append({
            "symbol": sym, "date": date, "direction": direction,
            "cluster": cluster, "why": why,
            "ext_time_et": ext_ts.strftime("%H:%M"),
            "ext_price": round(ext_px, 4),
            "minutes_to_extreme": round(vel["minutes_to_extreme"], 0) if vel else "",
            "speed_ratio": round(vel["speed_ratio"], 2) if vel else "",
            "level_type": lv["level_type"], "level_detail": lv["level_detail"],
            "level_price": lv["level_price"], "level_dist_atr": lv["level_dist_atr"],
            "level_touch_count": lv["touch_count"],
            "ext_from_open_atr": round(mm.get("ext_from_open_atr", np.nan), 3),
            "ext_from_prior_close_atr": round(mm.get("ext_from_prior_close_atr", np.nan), 3),
            "move_atr_3d": round(mm.get("move_atr_3d", np.nan), 3),
            "move_atr_5d": round(mm.get("move_atr_5d", np.nan), 3),
            "consecutive_dir_days": mm.get("consecutive_dir_days", ""),
            "gap_pct": round(mm.get("gap_pct", np.nan), 3),
            "atr14": round(atr, 4), "atr_pct": round(g["atr_pct"].iloc[i], 3),
            "avg_dollar_vol20": int(g["avg_dollar_vol20"].iloc[i]),
            "entry_ref_price": round(entry_px, 4),
            "entry_ref_time_et": entry_ts.strftime("%H:%M"),
            "structural_stop": round(stop, 4),
            "risk_per_share": round(risk, 4),
            "risk_as_pct_atr": round(risk / atr * 100, 1),
            "stopped_out": bool(stopped),
            "mfe_R": round(fav / risk, 3), "mae_R": round(adv / risk, 3),
            "r_at_vwap": r_at(float(hit_v["vwap"].iloc[0])) if len(hit_v) else "",
            "r_at_opening_print": r_at(float(r["open"].iloc[0])),
            "r_at_prior_close": r_at(float(g["prior_close"].iloc[i])),
            "r_at_intraday_swing": (
                r_at(float(last_hr["high"].max() if direction == "long"
                           else last_hr["low"].min())) if len(last_hr) else ""),
            "r_at_close": r_at(float(r["close"].iloc[-1])),
            "r_at_overnight_swing": (
                r_at(float(nxt["high"] if direction == "long" else nxt["low"]))
                if nxt is not None else ""),
            **grid_cols,
        })


def D_trigger(post, direction):
    """First 2-minute bar break after the extreme (the reference entry)."""
    b = (post.set_index("ts")
             .resample("2min", origin="start_day", offset="30min")
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "volume": "sum"}).dropna().reset_index())
    if len(b) < 2:
        return None, None
    for k in range(len(b) - 1):
        lvl = b["high"].iloc[k] if direction == "long" else b["low"].iloc[k]
        nxt = b.iloc[k + 1]
        if (direction == "long" and nxt["high"] > lvl) or \
           (direction == "short" and nxt["low"] < lvl):
            return float(lvl), nxt["ts"]
    return None, None


def main():
    symbols = pd.read_csv(UNIV)["symbol"].tolist()
    print(f"universe: {len(symbols)} symbols", flush=True)
    ctx = build_daily_context(set(symbols))
    print(f"daily context built for {len(ctx)} symbols", flush=True)

    dates = sorted({d for c in ctx.values() for d in c["idx"]
                    if pd.Timestamp(SCAN_START) <= d <= pd.Timestamp(SCAN_END)})
    dates = [d.date().isoformat() for d in dates]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    done = {f.stem for f in OUTDIR.glob("*.csv")}
    todo = [d for d in dates if d not in done]
    print(f"{len(dates)} scan dates, {len(todo)} remaining", flush=True)

    for n, date in enumerate(todo, 1):
        m = fetch_day(symbols, date)
        if m is None:
            (OUTDIR / f"{date}.csv").write_text(",".join(COLUMNS) + "\n",
                                                encoding="utf-8")
            continue
        results, errs = [], 0
        for sym, g in m.groupby("symbol", sort=False):
            c = ctx.get(sym)
            if c is not None:
                try:
                    analyse(sym, date, c, g, results)
                except Exception as e:
                    errs += 1
                    if errs <= 3:
                        print(f"    ERR {sym} {date}: {type(e).__name__}: {e}",
                              flush=True)
        if errs:
            print(f"    {date}: {errs} symbol errors", flush=True)
        # One file per day, written atomically with a fixed schema: avoids both
        # the ragged-row problem and any clobbering if two scans overlap.
        df = (pd.DataFrame(results) if results else pd.DataFrame(columns=COLUMNS))
        tmp = OUTDIR / f".{date}.tmp"
        df.reindex(columns=COLUMNS).to_csv(tmp, index=False)
        tmp.replace(OUTDIR / f"{date}.csv")
        if n % 20 == 0:
            print(f"  {n}/{len(todo)} days, last {date}, "
                  f"{len(results)} signals", flush=True)

    consolidate()


def consolidate():
    """Concatenate the per-day files into scan_results.csv."""
    files = sorted(OUTDIR.glob("*.csv"))
    if not files:
        return
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True).reindex(columns=COLUMNS)
    df.to_csv(OUT, index=False)
    print(f"\nconsolidated {len(files)} days -> {len(df):,} signals in {OUT.name}")


if __name__ == "__main__":
    main()
