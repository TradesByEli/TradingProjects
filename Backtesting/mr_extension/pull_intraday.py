"""Fetch Liberator minute bars for each example date (and the next session,
for overnight-swing targets). Caches one CSV per symbol/date under data/intraday/.
"""
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "intraday"
OUT.mkdir(parents=True, exist_ok=True)


def next_session(symbol, date):
    """Next trading day for this symbol, from its daily bar file."""
    df = pd.read_csv(ROOT / "data" / f"{symbol}.csv")
    d = pd.to_datetime(df["Timestamp"]).dt.normalize()
    later = d[d > pd.Timestamp(date)]
    return later.iloc[0].date().isoformat() if len(later) else None


def fetch(symbol, date):
    dest = OUT / f"{symbol}_{date}.csv"
    if dest.exists() and dest.stat().st_size > 500:
        return "cached"
    r = subprocess.run(
        ["kti", "liberator", "query", "--dataset", "minute_bars",
         "--symbols", symbol, "--date", date, "--format", "csv"],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or r.stdout.count("\n") < 2:
        return f"FAILED ({r.returncode})"
    dest.write_text(r.stdout, encoding="utf-8")
    return f"{r.stdout.count(chr(10))} rows"


if __name__ == "__main__":
    man = pd.read_csv(ROOT / "manifest.csv")
    seen = set()
    for _, m in man.iterrows():
        for d in (m["date"], next_session(m["symbol"], m["date"])):
            if d is None or (m["symbol"], d) in seen:
                continue
            seen.add((m["symbol"], d))
            print(f"{m['symbol']:6} {d}  {fetch(m['symbol'], d)}", flush=True)
