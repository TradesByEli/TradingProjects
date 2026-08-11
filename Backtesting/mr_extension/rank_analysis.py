"""Where do the 21 known examples rank among candidates on their own date?

Stage 2 can afford minute bars for roughly 60 unique symbols per day, so the
ranking must put the known setups near the top. Tests several ranking schemes
to find one that does.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent

cand = pd.read_csv(ROOT / "candidates.csv")
ex = pd.read_csv(ROOT / "examples.csv")
cand["level_dist_atr_daily"] = pd.to_numeric(
    cand["level_dist_atr_daily"], errors="coerce")

# candidate schemes ------------------------------------------------------
cand["s_ext"] = cand["ext_score"]
cand["s_tight"] = -cand["level_dist_atr_daily"].fillna(9.9)
cand["s_combo"] = cand["ext_score"] - 2.0 * cand["level_dist_atr_daily"].fillna(9.9)
# relative volatility: ATR% is how "in play" a name is
cand["s_atrpct"] = cand["atr_pct"]
cand["s_combo2"] = (cand["ext_score"] + 0.15 * cand["atr_pct"]
                    - 2.0 * cand["level_dist_atr_daily"].fillna(9.9))

SCHEMES = ["s_ext", "s_tight", "s_combo", "s_atrpct", "s_combo2"]
for s in SCHEMES:
    cand[f"r_{s}"] = cand.groupby(["date", "direction"])[s].rank(
        ascending=False, method="min")

key = ex[["id", "symbol", "date", "direction"]].copy()
merged = key.merge(cand, on=["symbol", "date", "direction"], how="left")

print(f"pool size per day/dir: mean "
      f"{len(cand)/cand['date'].nunique()/2:.0f}\n")
print(f"{'id':5} {'sym':6} " + " ".join(f"{s.replace('s_',''):>9}" for s in SCHEMES))
print("-" * 62)
for _, r in merged.iterrows():
    if pd.isna(r.get("r_s_ext")):
        print(f"{r['id']:5} {r['symbol']:6} " + " ".join(f"{'ABSENT':>9}" for _ in SCHEMES))
        continue
    print(f"{r['id']:5} {r['symbol']:6} "
          + " ".join(f"{int(r[f'r_{s}']):9d}" for s in SCHEMES))

print("\nexamples kept at each per-day/dir cut:")
print(f"{'cut':>6} " + " ".join(f"{s.replace('s_',''):>9}" for s in SCHEMES))
for n in (20, 40, 60, 100, 200):
    row = []
    for s in SCHEMES:
        rr = merged[f"r_{s}"].dropna()
        row.append(f"{int((rr <= n).sum()):9d}")
    print(f"{n:6d} " + " ".join(row))
print(f"\n(of {len(merged)} examples; ABSENT ones can never be recovered)")
