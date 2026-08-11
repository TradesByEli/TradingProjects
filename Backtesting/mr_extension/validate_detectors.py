"""Critical checkpoint: do the Phase 2 detectors fire on the 21 known examples?

Two-stage, exactly as the scan will run:
  1. DAILY PREFILTER (loose) -- would this symbol-day have been selected for a
     minute-bar pull at all? A miss here is fatal: the scan would never see it.
  2. MINUTE CONFIRMATION (strict) -- does it classify as one of the two setups?
     Exhaustion via daily-ATR extension, or open-drive via intraday velocity.

Level presence is recorded but NOT required (level_type="none" is allowed), so
the backtest can test whether the level matters rather than assuming it.
"""
from pathlib import Path
import pandas as pd

import detectors as D
from intraday_features import load_minutes, rth

ROOT = Path(__file__).parent


def load_symbol(sym):
    df = pd.read_csv(ROOT / "data" / f"{sym}.csv")
    df["date"] = pd.to_datetime(df["Timestamp"]).dt.normalize()
    df = df.rename(columns=str.lower)[
        ["date", "open", "high", "low", "close", "volume"]]
    return D.add_daily_features(df)


def classify(df, i, minutes, ext_ts, ext_px, direction, p=D.DEFAULT):
    """Which setup is this, and does it qualify? Returns (cluster, why)."""
    m = D.extension_metrics(df, i, ext_px, direction)
    strict = D.extension_qualifies(m, p)
    if strict:
        return "exhaustion", ",".join(strict)
    vel = D.intraday_velocity(minutes, ext_ts, ext_px, direction, df["atr14"].iloc[i], p)
    if vel and vel["speed_ratio"] >= p.min_speed_ratio:
        return "open_drive", f"spd={vel['speed_ratio']:.1f}"
    return None, (f"spd={vel['speed_ratio']:.1f}" if vel else "no-vel")


def check(p=D.DEFAULT, verbose=True):
    ex = pd.read_csv(ROOT / "examples.csv")
    ex["date"] = pd.to_datetime(ex["date"])
    rows = []

    for _, e in ex.iterrows():
        df = load_symbol(e["symbol"])
        idx = df.index[df["date"] == e["date"]]
        if len(idx) == 0:
            rows.append({"id": e["id"], "prefilter": "NO BAR", "cluster": "-"})
            continue
        i = int(idx[0])
        atr = df["atr14"].iloc[i]
        ext_px = float(e["ext_price_intraday"])
        direction = e["direction"]

        m = D.extension_metrics(df, i, ext_px, direction)
        pre = D.extension_qualifies(m, p, prefilter=True)
        liq = D.tradable(df, i, p)

        mins = load_minutes(e["symbol"], e["date"].date().isoformat())
        mins = rth(mins).reset_index(drop=True) if mins is not None else None
        ext_ts = pd.Timestamp(f"{e['date'].date()} {e['ext_time_et']}")

        cluster, why = classify(df, i, mins, ext_ts, ext_px, direction, p)
        levels = D.candidate_levels(df, i, direction, p)
        hit = D.nearest_level_hit(levels, ext_px, atr, p)

        rows.append({
            "id": e["id"], "sym": e["symbol"], "dir": direction,
            "author": e["author"][:6],
            "liq": liq,
            "prefilter": "PASS" if (pre and liq) else "DROP",
            "cluster": cluster or "-",
            "why": why,
            "level": (f"{hit['level_type']}:{hit['level_detail']}"
                      if hit["level_type"] != "none" else "none"),
            "lvl_dATR": hit["level_dist_atr"],
            "result": "FIRE" if (pre and liq and cluster) else "MISS",
        })

    out = pd.DataFrame(rows)
    if verbose:
        print(out.to_string(index=False))
        print(f"\nprefilter passed : {(out['prefilter']=='PASS').sum()} / {len(out)}")
        print(f"fully classified : {(out['result']=='FIRE').sum()} / {len(out)}")
        print("\nby cluster:")
        print(out[out["result"] == "FIRE"]["cluster"].value_counts().to_string())
        miss = out[out["result"] == "MISS"]
        if len(miss):
            print("\nMISSES -- each needs an explanation before scanning:")
            print(miss.to_string(index=False))
    return out


if __name__ == "__main__":
    check()
