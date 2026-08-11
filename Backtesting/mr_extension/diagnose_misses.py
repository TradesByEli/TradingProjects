"""Why did the detectors miss? Print the raw numbers behind each decision."""
from pathlib import Path
import pandas as pd

import detectors as D
from validate_detectors import load_symbol

ROOT = Path(__file__).parent
p = D.DEFAULT

ex = pd.read_csv(ROOT / "examples.csv")
ex["date"] = pd.to_datetime(ex["date"])

print("=== EXTENSION METRICS (threshold in header) ===")
print(f"{'id':5} {'auth':7} {'open':>6} {'pclose':>7} {'3d':>6} {'5d':>6} {'consec':>7}")
print(f"{'':5} {'':7} {p.min_ext_from_open_atr:>6} "
      f"{p.min_ext_from_prior_close_atr:>7} {p.min_move_atr_3d:>6} "
      f"{p.min_move_atr_5d:>6} {p.min_consecutive_days:>7}")
print("-" * 50)
for _, e in ex.iterrows():
    df = load_symbol(e["symbol"])
    i = int(df.index[df["date"] == e["date"]][0])
    m = D.extension_metrics(df, i, e["ext_price_intraday"], e["direction"])
    print(f"{e['id']:5} {e['author'][:6]:7} {m.get('ext_from_open_atr',0):6.2f} "
          f"{m.get('ext_from_prior_close_atr',0):7.2f} {m.get('move_atr_3d',0):6.2f} "
          f"{m.get('move_atr_5d',0):6.2f} {m.get('consecutive_dir_days',0):7d}")

print("\n\n=== NEAREST LEVEL, by relaxed search ===")
print(f"{'id':5} {'cited_lvl':>10} {'ext_px':>10} {'bestMA':>9} {'dATR':>6} "
      f"{'bestPivot':>10} {'dATR':>6} {'n_all':>6}")
print("-" * 70)
for _, e in ex.iterrows():
    df = load_symbol(e["symbol"])
    i = int(df.index[df["date"] == e["date"]][0])
    atr = df["atr14"].iloc[i]
    ext_px = e["ext_price_intraday"]

    # relaxed: ignore the side filter and the touch/age minimums
    best_ma = min(
        ((abs(ext_px - df[f"sma{n}"].iloc[i]), n) for n in p.ma_lengths
         if not pd.isna(df[f"sma{n}"].iloc[i])), default=(None, None))
    piv = D.swing_pivots(df, i, e["direction"], p)
    best_pv = min(((abs(ext_px - pr), k) for k, pr, _ in piv), default=(None, None))

    print(f"{e['id']:5} {str(e['level_price']):>10} {ext_px:10.2f} "
          f"{('SMA'+str(best_ma[1])) if best_ma[1] else '-':>9} "
          f"{(best_ma[0]/atr) if best_ma[0] is not None else float('nan'):6.2f} "
          f"{best_pv[1] or '-':>10} "
          f"{(best_pv[0]/atr) if best_pv[0] is not None else float('nan'):6.2f} "
          f"{len(piv):6d}")
