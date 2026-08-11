"""Phase 1: build the coded dataset from the PPTX examples + daily bars.

Emits examples.csv -- one row per annotated example, with level, extension,
volume, market-context, and R-based outcome features.

Outcomes are anchored to the INTRADAY setup extreme (see intraday_features),
not the daily extreme: they are often different events.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from intraday_features import compute as compute_intraday

ROOT = Path(__file__).parent
DATA = ROOT / "data"

# Sector proxy per symbol. SPY/QQQ are always included as broad context.
SECTOR_PROXY = {
    "SOXL": "SMH", "NVDA": "SMH", "MU": "SMH", "INTC": "SMH", "SNDK": "SMH",
    "AAPL": "QQQ", "MSFT": "QQQ", "PLTR": "QQQ",
    "IBIT": "IBIT",   # crypto; itself -- no clean proxy, flagged in notes
    "BE": "SPY", "CRCL": "SPY", "HOOD": "SPY", "FIG": "SPY", "SPCX": "SPY",
}


def load(symbol):
    df = pd.read_csv(DATA / f"{symbol}.csv")
    df["date"] = pd.to_datetime(df["Timestamp"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "Open", "High", "Low", "Close", "Volume"]].rename(
        columns=str.lower
    )


def add_features(df):
    """Daily-bar derived features. All shifted where 'prior' is meant."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    pc = c.shift(1)

    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    # ATR as of the PRIOR close: what you'd have known at the open.
    df["atr14"] = tr.rolling(14).mean().shift(1)

    for n in (5, 9, 20, 50, 100, 200):
        df[f"sma{n}"] = c.rolling(n).mean()
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()

    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_upper"], df["bb_lower"] = ma20 + 2 * sd20, ma20 - 2 * sd20
    df["bb_pos"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["range"] = h - l
    df["avg_range20"] = df["range"].rolling(20).mean().shift(1)
    df["range_expansion"] = df["range"] / df["avg_range20"]

    df["avg_vol20"] = v.rolling(20).mean().shift(1)
    df["rvol_daily"] = v / df["avg_vol20"]
    df["vol_vs_prior_day"] = v / v.shift(1)

    df["day_pct"] = (c / pc - 1) * 100
    df["gap_pct"] = (df["open"] / pc - 1) * 100
    return df


def consec_days(df, i, direction):
    """Count consecutive same-direction closes ending at index i."""
    n = 0
    for j in range(i, 0, -1):
        d = df["close"].iloc[j] - df["close"].iloc[j - 1]
        if (direction == "long" and d < 0) or (direction == "short" and d > 0):
            n += 1
        else:
            break
    return n


def bb_traversal_days(df, i, direction):
    """Bars since price was at the opposite Bollinger extreme (<=25 lookback)."""
    target_hi = direction == "long"   # long MR: came from upper band down
    for k in range(1, min(26, i + 1)):
        p = df["bb_pos"].iloc[i - k]
        if pd.isna(p):
            return ""
        if (target_hi and p >= 0.95) or (not target_hi and p <= 0.05):
            return k
    return ""


def pct_off_extreme(df, i, direction, lookback=60):
    """% off the running high (long) or low (short) of the lookback window."""
    w = df.iloc[max(0, i - lookback):i + 1]
    if direction == "long":
        return (df["low"].iloc[i] / w["high"].max() - 1) * 100
    return (df["high"].iloc[i] / w["low"].min() - 1) * 100


def find_level(df, i, direction, cited, atr):
    """Resolve the cited level against MAs and daily swing pivots.

    Returns (level_type, level_detail, level_price, age_days, touch_count).
    Tolerance is 0.35 ATR -- tight enough to be meaningful, loose enough for
    hand-annotated round numbers.
    """
    if cited is None or pd.isna(cited):
        px = df["low"].iloc[i] if direction == "long" else df["high"].iloc[i]
        best = None
        for n in (5, 9, 20, 50, 100, 200):
            mv = df[f"sma{n}"].iloc[i]
            if pd.isna(mv):
                continue
            d = abs(px - mv)
            if best is None or d < best[0]:
                best = (d, n, mv)
        if best is None:
            return ("unresolved", "", "", "", "")
        return ("MA", f"SMA{best[1]}", round(best[2], 2), "", "")

    tol = 0.35 * atr if atr and not pd.isna(atr) else abs(cited) * 0.01

    for n in (5, 9, 20, 50, 100, 200):
        mv = df[f"sma{n}"].iloc[i]
        if not pd.isna(mv) and abs(mv - cited) <= tol:
            return ("MA", f"SMA{n}", round(mv, 2), "", "")

    age, touches = "", 0
    for k in range(1, min(130, i + 1)):
        bh, bl = df["high"].iloc[i - k], df["low"].iloc[i - k]
        if abs(bh - cited) <= tol or abs(bl - cited) <= tol:
            touches += 1
            if age == "":
                age = k
    ltype = "round" if float(cited) % 25 == 0 else "pivot"
    if touches == 0:
        ltype = "HTF"   # cited but not tested in the last 130 sessions
    return (ltype, "prior daily high/low", cited, age, touches)


def regime(df, i):
    """Index regime from MA stack + 20-day range width."""
    c = df["close"].iloc[i]
    s20, s50 = df["sma20"].iloc[i], df["sma50"].iloc[i]
    if pd.isna(s50):
        return ""
    w = df.iloc[max(0, i - 19):i + 1]
    range_pct = (w["high"].max() / w["low"].min() - 1) * 100
    if c > s20 > s50 and range_pct > 4:
        return "uptrend"
    if c < s20 < s50 and range_pct > 4:
        return "downtrend"
    return "range"


def rnd(x, n=2):
    return "" if x is None or pd.isna(x) else round(float(x), n)


def ctx(df, i, prefix):
    """Market-context block for an index/proxy at index i."""
    c = df["close"].iloc[i]
    s20, s50 = df["sma20"].iloc[i], df["sma50"].iloc[i]
    w = df.iloc[max(0, i - 19):i + 1]
    return {
        f"{prefix}_close_vs_20sma_pct": rnd((c / s20 - 1) * 100),
        f"{prefix}_close_vs_50sma_pct": rnd((c / s50 - 1) * 100),
        f"{prefix}_above_20sma": bool(c > s20) if not pd.isna(s20) else "",
        f"{prefix}_above_50sma": bool(c > s50) if not pd.isna(s50) else "",
        f"{prefix}_20d_range_pct": rnd((w["high"].max() / w["low"].min() - 1) * 100),
        f"{prefix}_regime": regime(df, i),
        f"{prefix}_day_pct": rnd(df["day_pct"].iloc[i]),
        f"{prefix}_gap_pct": rnd(df["gap_pct"].iloc[i]),
    }


def vix_bucket(v):
    if pd.isna(v):
        return ""
    return "<15" if v < 15 else "15-20" if v < 20 else "20-25" if v < 25 else ">25"


def main():
    man = pd.read_csv(ROOT / "manifest.csv")
    man["date"] = pd.to_datetime(man["date"])

    cache = {}

    def get(sym):
        if sym not in cache:
            cache[sym] = add_features(load(sym))
        return cache[sym]

    spy, qqq, vix = get("SPY"), get("QQQ"), get("VIX")
    rows = []

    for _, m in man.iterrows():
        sym, d, direction = m["symbol"], m["date"], m["direction"]
        df = get(sym)
        idx = df.index[df["date"] == d]
        if len(idx) == 0:
            rows.append({**m.to_dict(),
                         "data_error": f"no daily bar for {sym} on {d.date()}"})
            continue
        i = int(idx[0])
        atr = df["atr14"].iloc[i]
        o, h, l, c = (df[k].iloc[i] for k in ("open", "high", "low", "close"))
        pc = df["close"].iloc[i - 1]
        ext_px = l if direction == "long" else h        # daily extension extreme
        rev_px = h if direction == "long" else l

        ltype, ldetail, lprice, lage, ltouch = find_level(
            df, i, direction, m["level_price_cited"], atr
        )

        r = {
            "id": m["id"], "slide": m["slide"], "author": m["author"],
            "date": d.date().isoformat(), "symbol": sym, "direction": direction,
            "setup_label": m["setup_label"], "tod_et": m["tod_et"],
            "session_phase": (
                "open_drive" if m["tod_et"] <= "09:45"
                else "morning" if m["tod_et"] <= "10:30" else "late_morning"
            ),
            # --- level ---
            "level_price_cited": m["level_price_cited"],
            "level_price": lprice, "level_type": ltype, "level_detail": ldetail,
            "level_age_days": lage, "level_touch_count": ltouch,
            "level_dist_from_extreme_atr": "", "level_pierced": "",
            # --- daily OHLC ---
            "day_open": rnd(o), "day_high": rnd(h), "day_low": rnd(l),
            "day_close": rnd(c), "prior_close": rnd(pc),
            # --- extension ---
            "atr14_daily": rnd(atr),
            "atr_pct_of_price": rnd(atr / pc * 100),
            "ext_from_open_atr": rnd(abs(ext_px - o) / atr, 3),
            "ext_from_prior_close_atr": rnd(abs(ext_px - pc) / atr, 3),
            "ext_from_hod_lod_atr": rnd(abs(ext_px - rev_px) / atr, 3),
            "gap_pct": rnd(df["gap_pct"].iloc[i]),
            "day_pct": rnd(df["day_pct"].iloc[i]),
            "consecutive_dir_days": consec_days(df, i - 1, direction),
            "bb_pos_prior": rnd(df["bb_pos"].iloc[i - 1], 3),
            "bb_traversal_days": bb_traversal_days(df, i, direction),
            "pct_off_swing_extreme_60d": rnd(pct_off_extreme(df, i, direction)),
            "daily_range_expansion": rnd(df["range_expansion"].iloc[i], 2),
            # --- volume ---
            "rvol_daily": rnd(df["rvol_daily"].iloc[i], 2),
            "vol_vs_prior_day": rnd(df["vol_vs_prior_day"].iloc[i], 2),
            "capitulation_flag": bool(df["vol_vs_prior_day"].iloc[i] >= 2.0),
            # --- symbol position vs its own MAs ---
            "close_vs_50sma_pct": rnd((c / df["sma50"].iloc[i] - 1) * 100),
            "close_vs_200sma_pct": rnd((c / df["sma200"].iloc[i] - 1) * 100),
        }

        for n in (1, 3, 5):
            if i - n >= 0:
                base = df["close"].iloc[i - n]
                r[f"move_pct_{n}d"] = rnd((ext_px / base - 1) * 100)
                r[f"move_atr_{n}d"] = rnd(abs(ext_px - base) / atr, 2)

        for name, idf in (("spy", spy), ("qqq", qqq)):
            j = idf.index[idf["date"] == d]
            r.update(ctx(idf, int(j[0]), name) if len(j) else {})

        proxy = SECTOR_PROXY.get(sym, "SPY")
        r["sector_proxy"] = proxy
        pdf = get(proxy)
        j = pdf.index[pdf["date"] == d]
        if len(j):
            r.update(ctx(pdf, int(j[0]), "proxy"))

        j = vix.index[vix["date"] == d]
        if len(j):
            k = int(j[0])
            vc = vix["close"].iloc[k]
            r.update({
                "vix_close": rnd(vc),
                "vix_change_pct": rnd(vix["day_pct"].iloc[k]),
                "vix_vs_20sma_pct": rnd((vc / vix["sma20"].iloc[k] - 1) * 100),
                "vix_bucket": vix_bucket(vc),
            })

        r["daily_extreme"] = rnd(ext_px)

        # --- outcome, anchored to the INTRADAY setup extreme ---
        next_date = (df["date"].iloc[i + 1].date().isoformat()
                     if i + 1 < len(df) else None)
        intr = compute_intraday(sym, d.date().isoformat(), next_date,
                                direction, m["tod_et"], atr)
        r.update(intr)

        if intr.get("ext_price_intraday") not in ("", None):
            ip = float(intr["ext_price_intraday"])
            r["ext_matches_daily_extreme"] = bool(abs(ip - ext_px) / atr < 0.05)
            # Re-express extension measures against the real setup extreme.
            r["ext_from_open_atr"] = rnd(abs(ip - o) / atr, 3)
            r["ext_from_prior_close_atr"] = rnd(abs(ip - pc) / atr, 3)
            for n in (1, 3, 5):
                if i - n >= 0:
                    base = df["close"].iloc[i - n]
                    r[f"move_pct_{n}d"] = rnd((ip / base - 1) * 100)
                    r[f"move_atr_{n}d"] = rnd(abs(ip - base) / atr, 2)
            if r["level_price"] != "":
                r["level_dist_from_extreme_atr"] = rnd(
                    abs(ip - float(r["level_price"])) / atr, 3)
                r["level_pierced"] = (bool(ip < float(r["level_price"]))
                                      if direction == "long"
                                      else bool(ip > float(r["level_price"])))

        if r.get("vol_first_30m") not in ("", None) and not pd.isna(df["avg_vol20"].iloc[i]):
            # first-30m volume vs 1/13th of a normal day (RTH is ~6.5h)
            r["rvol_first_30m"] = rnd(
                float(r["vol_first_30m"]) / (df["avg_vol20"].iloc[i] / 13), 2)

        r["year_confidence"] = m["year_confidence"]
        r["notes_cited"] = m["notes_cited"]
        rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "examples.csv", index=False)
    print(f"wrote examples.csv: {len(out)} rows x {len(out.columns)} cols")
    if "data_error" in out.columns:
        errs = out[out["data_error"].notna()]
        if len(errs):
            print("\nDATA ERRORS:")
            print(errs[["id", "symbol", "date", "data_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
