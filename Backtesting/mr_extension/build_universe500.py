"""Select the ~500-symbol scan universe: most liquid names, plus every deck
symbol and the leveraged/high-beta ETFs, which must never be screened out."""
from pathlib import Path
import pandas as pd

from build_universe import DECK_SYMBOLS, ETFS

ROOT = Path(__file__).parent
UNIV = ROOT / "data" / "universe"
HIST = ROOT / "data" / "daily_hist"

TARGET = 500

frames = [pd.read_csv(f) for f in sorted(HIST.glob("202[4-6]*.csv"))]
df = pd.concat(frames, ignore_index=True)
df["dollar_vol"] = df["close"] * df["volume"]

med = df.groupby("symbol")["dollar_vol"].median().sort_values(ascending=False)
seeds = [s for s in set(DECK_SYMBOLS) | set(ETFS) if s in med.index]
top = [s for s in med.index if s not in seeds][:TARGET - len(seeds)]

universe = sorted(set(seeds) | set(top))
pd.Series(universe, name="symbol").to_csv(UNIV / "universe500.csv", index=False)
print(f"universe500: {len(universe)} symbols "
      f"({len(seeds)} deck/ETF seeds + {len(top)} by liquidity)")
missing = sorted(set(DECK_SYMBOLS) - set(universe))
print(f"deck symbols missing: {missing if missing else 'none'}")
