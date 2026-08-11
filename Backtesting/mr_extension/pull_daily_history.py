"""Pull daily bars for the whole scan window, month by month.

Liberator's client times out at 120s on any query using --symbols over a date
range, but an UNFILTERED month-range query returns ~254k rows in ~68s. So we
pull full-market months and filter to the universe locally, keeping only ~13%
of the rows.

Starts a year before SCAN_START so 200-SMA and ATR are defined on day one.
"""
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
UNIV = ROOT / "data" / "universe"
OUT = ROOT / "data" / "daily_hist"
OUT.mkdir(parents=True, exist_ok=True)

HIST_START = "2021-01-01"      # lookback buffer for 200-SMA
HIST_END = "2026-08-04"

FIELDS = "symbol,timestamp,open,high,low,close,volume"


def months(start, end):
    d = pd.Timestamp(start).normalize().replace(day=1)
    last = pd.Timestamp(end)
    while d < last:
        nxt = (d + pd.offsets.MonthBegin(1))
        yield d.date().isoformat(), min(nxt, last).date().isoformat()
        d = nxt


def main():
    universe = set(pd.read_csv(UNIV / "universe.csv")["symbol"])
    print(f"universe: {len(universe)} symbols")

    for s, e in months(HIST_START, HIST_END):
        dest = OUT / f"{s[:7]}.csv"
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"{s[:7]}  cached", flush=True)
            continue
        r = subprocess.run(
            ["kti", "liberator", "query", "--dataset", "daily_bars",
             "--start-date", s, "--end-date", e, "--fields", FIELDS,
             "--format", "csv", "--limit", "400000"],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or r.stdout.count("\n") < 100:
            print(f"{s[:7]}  FAILED rc={r.returncode}", flush=True)
            continue
        from io import StringIO
        df = pd.read_csv(StringIO(r.stdout))
        df = df[df["symbol"].isin(universe)]
        keep = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        df[keep].to_csv(dest, index=False)
        print(f"{s[:7]}  {len(df)} rows kept", flush=True)


if __name__ == "__main__":
    main()
