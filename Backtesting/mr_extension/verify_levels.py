"""Verify each cited level actually sits at the day's extension extreme.

If the cited price is nowhere near that day's high/low, either the date is
wrong (Thomas's slides omit the year) or the level was mis-transcribed.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
man = pd.read_csv(ROOT / "manifest.csv")
man["date"] = pd.to_datetime(man["date"])

print(f"{'id':5} {'sym':5} {'date':11} {'dir':6} {'cited':>9} {'low':>9} {'high':>9} "
      f"{'dist%':>7}  verdict")
print("-" * 88)

for _, m in man.iterrows():
    df = pd.read_csv(ROOT / "data" / f"{m['symbol']}.csv")
    df["date"] = pd.to_datetime(df["Timestamp"]).dt.normalize()
    row = df[df["date"] == m["date"]]
    if row.empty:
        print(f"{m['id']:5} {m['symbol']:5} {str(m['date'].date()):11} "
              f"{m['direction']:6} {'':>9} {'':>9} {'':>9} {'':>7}  NO BAR")
        continue
    lo, hi = float(row["Low"].iloc[0]), float(row["High"].iloc[0])
    cited = m["level_price_cited"]
    if pd.isna(cited):
        print(f"{m['id']:5} {m['symbol']:5} {str(m['date'].date()):11} "
              f"{m['direction']:6} {'(MA)':>9} {lo:9.2f} {hi:9.2f} {'':>7}  no cited price")
        continue
    ext = lo if m["direction"] == "long" else hi
    dist = (ext - cited) / cited * 100
    verdict = ("MATCH" if abs(dist) <= 1.5 else
               "CLOSE" if abs(dist) <= 4 else "MISMATCH -- investigate")
    print(f"{m['id']:5} {m['symbol']:5} {str(m['date'].date()):11} {m['direction']:6} "
          f"{cited:9.2f} {lo:9.2f} {hi:9.2f} {dist:+7.2f}%  {verdict}")
