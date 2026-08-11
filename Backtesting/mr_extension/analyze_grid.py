"""Phase 3 analysis: which entry trigger, target, and trail actually pay?

Everything is reported separately for long-MR and short-MR (the user's
decision), and separately for the two clusters (exhaustion vs open_drive),
because Phase 1 suggested they behave differently.

Every table carries its sample size. Where n is too small to conclude
anything, the output says so rather than implying a finding.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent

TRIGGERS = ["immediate", "brk2m", "brk5m", "ema9_2m", "ema9_5m"]
TARGETS = ["tgt_vwap", "tgt_open_print", "tgt_prior_close",
           "tgt_intraday_swing", "tgt_close", "tgt_overnight"]
TRAILS = ["trail2m", "trail5m", "trail15m", "ema9_5m", "ema21_5m"]

MIN_N = 30          # below this, report but flag as untrustworthy


def load():
    d = pd.read_csv(ROOT / "scan_results.csv", low_memory=False)
    d["date"] = pd.to_datetime(d["date"])
    return d


def expectancy(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {}
    return {
        "n": len(s), "mean_R": s.mean(), "median_R": s.median(),
        "win_rate": (s > 0).mean(), "p10": s.quantile(0.10),
        "p90": s.quantile(0.90),
        # robustness: mean of the worst decile
        "worst_decile": s[s <= s.quantile(0.10)].mean(),
    }


def table(d, cols, label):
    rows = []
    for c in cols:
        col = f"{c[0]}__{c[1]}" if isinstance(c, tuple) else c
        if col not in d.columns:
            continue
        e = expectancy(d[col])
        if e:
            rows.append({label: col, **e})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows).sort_values("mean_R", ascending=False)
    for c in ("mean_R", "median_R", "p10", "p90", "worst_decile"):
        t[c] = t[c].round(3)
    t["win_rate"] = (t["win_rate"] * 100).round(1)
    return t


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def analyse_slice(d, name):
    section(f"{name}   (n={len(d):,} signals)")
    if len(d) < MIN_N:
        print(f"  n={len(d)} is too small to conclude anything. Reported for "
              f"completeness only.")

    print("\n-- ENTRY TRIGGER (exit at session close, common yardstick) --")
    t = table(d, [(x, "tgt_close") for x in TRIGGERS], "trigger")
    print(t.to_string(index=False) if len(t) else "  no data")

    print("\n-- TARGET (entry = 2-minute bar break) --")
    t = table(d, [("brk2m", x) for x in TARGETS], "target")
    print(t.to_string(index=False) if len(t) else "  no data")

    print("\n-- TRAILING METHOD (entry = 2-minute bar break) --")
    t = table(d, [("brk2m", x) for x in TRAILS], "trail")
    print(t.to_string(index=False) if len(t) else "  no data")

    print("\n-- BEST COMBINATION (top 12 by mean R) --")
    combos = []
    for tr in TRIGGERS:
        for ex in TARGETS + TRAILS:
            col = f"{tr}__{ex}"
            if col in d.columns:
                e = expectancy(d[col])
                if e and e["n"] >= MIN_N:
                    combos.append({"trigger": tr, "exit": ex, **e})
    if combos:
        c = pd.DataFrame(combos).sort_values("mean_R", ascending=False)
        for k in ("mean_R", "median_R", "p10", "p90", "worst_decile"):
            c[k] = c[k].round(3)
        c["win_rate"] = (c["win_rate"] * 100).round(1)
        print(c.head(12).to_string(index=False))
        print("\n  most ROBUST (top 5 by worst-decile mean):")
        print(c.sort_values("worst_decile", ascending=False)
               .head(5).to_string(index=False))


def conditional(d, col, ref="brk2m__tgt_close", bins=None):
    """Does this context variable separate winners from losers?"""
    x = d.copy()
    if bins is not None:
        x["_b"] = pd.cut(pd.to_numeric(x[col], errors="coerce"), bins)
    else:
        x["_b"] = x[col]
    g = x.groupby("_b", observed=True)[ref].agg(
        n="count", mean_R="mean", median_R="median",
        win_rate=lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean())
    g = g[g["n"] >= MIN_N]
    if g.empty:
        return None
    g["mean_R"] = g["mean_R"].round(3)
    g["median_R"] = g["median_R"].round(3)
    g["win_rate"] = (g["win_rate"] * 100).round(1)
    return g


def main():
    d = load()
    print(f"loaded {len(d):,} signals, {d['date'].min().date()} .. "
          f"{d['date'].max().date()}")
    print(f"symbols: {d['symbol'].nunique()}   "
          f"days: {d['date'].nunique()}")
    print("\nsignal mix:")
    print(d.groupby(["direction", "cluster"]).size().to_string())

    for direction in ("long", "short"):
        for cluster in ("exhaustion", "open_drive"):
            s = d[(d["direction"] == direction) & (d["cluster"] == cluster)]
            if len(s):
                analyse_slice(s, f"{direction.upper()} / {cluster}")

    section("CONTEXT: what separates good signals from bad?")
    print("\nBase reference = brk2m entry, exit at session close.\n")
    for direction in ("long", "short"):
        s = d[d["direction"] == direction]
        print(f"\n### {direction.upper()} (n={len(s):,})")
        for col, bins in [
            ("level_type", None),
            ("level_dist_atr", [0, .1, .2, .35, 1, 99]),
            ("level_touch_count", [0, 2, 4, 8, 999]),
            ("ext_from_open_atr", [0, .5, 1, 1.5, 2, 99]),
            ("speed_ratio", [0, 4, 10, 25, 9999]),
            ("consecutive_dir_days", [-1, 0, 1, 2, 3, 99]),
            ("atr_pct", [0, 3, 5, 8, 99]),
        ]:
            if col not in s.columns:
                continue
            g = conditional(s, col, bins=bins)
            if g is not None:
                print(f"\n  by {col}:")
                print("   " + g.to_string().replace("\n", "\n   "))

    section("HOW DO THE DECK'S 21 EXAMPLES COMPARE?")
    ex = pd.read_csv(ROOT / "examples.csv")
    ex["date"] = pd.to_datetime(ex["date"])
    key = ex[["id", "symbol", "date", "direction"]]
    m = key.merge(d, on=["symbol", "date", "direction"], how="left")
    found = m["cluster"].notna().sum()
    print(f"\n{found} of {len(key)} deck examples were re-found by the scan.")
    if found:
        print(f"\ndeck examples   median brk2m__tgt_close: "
              f"{pd.to_numeric(m['brk2m__tgt_close'], errors='coerce').median():.3f} R")
        print(f"whole population median brk2m__tgt_close: "
              f"{pd.to_numeric(d['brk2m__tgt_close'], errors='coerce').median():.3f} R")
        print("\n  -> if these are close, the deck's examples are not special "
              "and the edge is in the execution/context, not the selection.")


if __name__ == "__main__":
    main()
