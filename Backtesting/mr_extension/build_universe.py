"""Build the scan universe: liquid momentum names + leveraged ETFs.

Two stages, to keep the data pull bounded:
  1. Sample full-market daily bars on a handful of dates spread across the scan
     window; keep symbols whose median dollar volume clears a floor.
  2. Pull full OHLCV history for those survivors only, and apply the ATR% filter
     per-day at scan time (universe membership is date-dependent).

The deck's 14 symbols are always seeded in, so the known examples can never be
screened out of their own study.
"""
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
UNIV = DATA / "universe"
UNIV.mkdir(parents=True, exist_ok=True)

SCAN_START, SCAN_END = "2022-01-03", "2026-08-03"

# Dates spread across the window for the liquidity screen.
SAMPLE_DATES = ["2022-03-15", "2022-10-13", "2023-06-15", "2024-02-15",
                "2024-09-17", "2025-04-15", "2025-11-18", "2026-06-16"]

DECK_SYMBOLS = ["SNDK", "SOXL", "IBIT", "AAPL", "BE", "PLTR", "NVDA", "CRCL",
                "MSFT", "MU", "INTC", "HOOD", "FIG", "SPCX"]

# Leveraged / high-beta ETFs of the kind that appear in the deck.
ETFS = ["SOXL", "SOXS", "TQQQ", "SQQQ", "SPXL", "SPXS", "TNA", "TZA", "LABU",
        "LABD", "FAS", "FAZ", "NUGT", "DUST", "YINN", "YANG", "IBIT", "ETHA",
        "MSTU", "MSTZ", "NVDL", "TSLL", "TSLQ", "UVXY", "VXX", "ARKK", "SMH",
        "XBI", "GDX", "SPY", "QQQ", "IWM"]

MIN_MEDIAN_DOLLAR_VOL = 50_000_000    # $50M/day median across sample dates


def fetch_day(date):
    dest = UNIV / f"market_{date}.csv"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    r = subprocess.run(
        ["kti", "liberator", "query", "--dataset", "daily_bars", "--date", date,
         "--fields", "symbol,close,volume", "--format", "csv", "--limit", "20000"],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or r.stdout.count("\n") < 100:
        print(f"  {date}: FAILED")
        return None
    dest.write_text(r.stdout, encoding="utf-8")
    return dest


def main():
    frames = []
    for d in SAMPLE_DATES:
        f = fetch_day(d)
        if f is None:
            continue
        df = pd.read_csv(f)
        df = df[["symbol", "close", "volume"]].dropna()
        df["dollar_vol"] = df["close"] * df["volume"]
        df["date"] = d
        frames.append(df)
        print(f"  {d}: {len(df)} symbols", flush=True)

    all_df = pd.concat(frames)
    med = all_df.groupby("symbol")["dollar_vol"].median()
    px = all_df.groupby("symbol")["close"].median()

    keep = med[(med >= MIN_MEDIAN_DOLLAR_VOL) & (px >= 5)].index
    universe = sorted(set(keep) | set(DECK_SYMBOLS) | set(ETFS))
    # Drop anything with a non-alphabetic ticker (warrants, units, preferreds).
    universe = [s for s in universe if s.isalpha() and len(s) <= 5]

    pd.Series(universe, name="symbol").to_csv(UNIV / "universe.csv", index=False)
    print(f"\nuniverse: {len(universe)} symbols "
          f"(liquidity screen kept {len(keep)}, plus deck + ETF seeds)")
    print(f"scan window: {SCAN_START} .. {SCAN_END}")


if __name__ == "__main__":
    main()
