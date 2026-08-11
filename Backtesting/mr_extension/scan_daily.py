"""Scan stage 1: daily prefilter over the universe.

Selects which symbol-days are worth pulling minute bars for. The real setup
test runs on minute bars in stage 2.

Why the prefilter requires a LEVEL, not just extension: ranking candidates by
extension magnitude alone puts the 21 known examples at a median rank of 155
out of ~1000 per day -- keeping the top 200 would still capture only 11 of 21.
Extension size does not distinguish these setups. What distinguishes them is
extension INTO a level, so level proximity is part of candidate selection.

Level-less signals are still emitted, but only when the extension is large
(the E10 CRCL / T01 SNDK case: extension into new highs, where no prior level
exists). That keeps the "does the level matter?" question testable.

Uses the DAILY extreme as a stand-in for the intraday setup extreme -- strictly
over-inclusive, the correct bias for a prefilter.
"""
from pathlib import Path
import pandas as pd
import numpy as np

import detectors as D

ROOT = Path(__file__).parent
HIST = ROOT / "data" / "daily_hist"

SCAN_START, SCAN_END = "2022-01-03", "2026-08-03"
LEVELLESS_MIN_SCORE = 1.2      # extension needed to emit without a level
p = D.DEFAULT


def load_all():
    frames = [pd.read_csv(f) for f in sorted(HIST.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    df = df[["symbol", "date", "open", "high", "low", "close", "volume"]]
    return df.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])


def fractal_points(h, l, n):
    """Indices of fractal swing highs/lows over the whole series, once."""
    N = len(h)
    hi = np.zeros(N, bool)
    lo = np.zeros(N, bool)
    for k in range(n, N - n):
        if h[k] == h[k - n:k + n + 1].max():
            hi[k] = True
        if l[k] == l[k - n:k + n + 1].min():
            lo[k] = True
    return hi, lo


def main():
    all_df = load_all()
    print(f"loaded {len(all_df):,} symbol-days across "
          f"{all_df['symbol'].nunique()} symbols", flush=True)

    start, end = pd.Timestamp(SCAN_START), pd.Timestamp(SCAN_END)
    out = []
    done = 0

    for sym, g in all_df.groupby("symbol", sort=False):
        done += 1
        if done % 200 == 0:
            print(f"  {done} symbols, {len(out):,} candidates", flush=True)
        if len(g) < 220:
            continue
        g = D.add_daily_features(g.reset_index(drop=True), p)
        h, l = g["high"].values, g["low"].values
        c = g["close"].values
        hi_f, lo_f = fractal_points(h, l, p.pivot_fractal_n)
        ma_cols = {n: g[f"sma{n}"].values for n in p.ma_lengths}

        idxs = g.index[(g["date"] >= start) & (g["date"] <= end)]
        for i in idxs:
            atr = g["atr14"].iloc[i]
            if pd.isna(atr) or atr <= 0 or not D.tradable(g, i, p):
                continue
            ref = g["prior_close"].iloc[i]
            lo_w = max(0, i - p.pivot_lookback)

            # pivot levels available as of today
            piv = np.concatenate([h[lo_w:i][hi_f[lo_w:i]], l[lo_w:i][lo_f[lo_w:i]]])

            for direction in ("long", "short"):
                ext_px = l[i] if direction == "long" else h[i]
                m = D.extension_metrics(g, i, ext_px, direction)
                fired = D.extension_qualifies(m, p, prefilter=True)
                if not fired:
                    continue

                score = max(m.get("ext_from_open_atr", 0),
                            m.get("ext_from_prior_close_atr", 0))

                # nearest level on the correct side, MA or pivot
                best_d, best_t, best_p_ = None, "", None
                for n, arr in ma_cols.items():
                    v = arr[i]
                    if np.isnan(v):
                        continue
                    if (direction == "long" and v > ref) or \
                       (direction == "short" and v < ref):
                        continue
                    d = abs(ext_px - v)
                    if best_d is None or d < best_d:
                        best_d, best_t, best_p_ = d, f"MA:SMA{n}", v
                if len(piv):
                    side = piv[piv <= ref] if direction == "long" else piv[piv >= ref]
                    if len(side):
                        k = np.argmin(np.abs(ext_px - side))
                        d = abs(ext_px - side[k])
                        if best_d is None or d < best_d:
                            best_d, best_t, best_p_ = d, "pivot", float(side[k])

                dist_atr = (best_d / atr) if best_d is not None else np.inf
                at_level = dist_atr <= p.level_tol_atr
                if not at_level and score < LEVELLESS_MIN_SCORE:
                    continue

                out.append({
                    "symbol": sym, "date": g["date"].iloc[i].date().isoformat(),
                    "direction": direction,
                    "prefilter_clocks": ",".join(fired),
                    "ext_score": round(score, 3),
                    "level_type_daily": best_t if at_level else "none",
                    "level_price_daily": round(best_p_, 4) if at_level else "",
                    "level_dist_atr_daily": (round(dist_atr, 3)
                                             if np.isfinite(dist_atr) else ""),
                    "atr14": round(atr, 4),
                    "prior_close": round(ref, 4),
                    "day_open": round(g["open"].iloc[i], 4),
                    "daily_extreme": round(ext_px, 4),
                    "avg_dollar_vol20": int(g["avg_dollar_vol20"].iloc[i]),
                    "atr_pct": round(g["atr_pct"].iloc[i], 3),
                })

    cand = pd.DataFrame(out)
    cand.to_csv(ROOT / "candidates.csv", index=False)
    print(f"\ncandidates: {len(cand):,}")
    print(cand["direction"].value_counts().to_string())
    print(cand["level_type_daily"].value_counts().to_string())
    print(f"unique symbol-days: {cand.groupby(['symbol','date']).ngroups:,}")
    print(f"per day/dir: mean "
          f"{len(cand)/cand['date'].nunique()/2:.1f}")


if __name__ == "__main__":
    main()
