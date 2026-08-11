# Phase 1 notes — coded dataset of the 21 deck examples

`examples.csv` — 21 rows × 103 columns. One row per annotated example (20 dates;
INTC 2026-05-26 split into T04a long / T04b short; slide 23 dropped as a
duplicate of slide 18). **Reviewed and approved by the user.**

## Corrections made to the deck

| id | Deck says | Corrected to | Evidence |
|---|---|---|---|
| E10 | 07/20/25 | **2025-07-18** | 07/20/25 is a Sunday. CRCL high on 07/18 was 262.97 vs the cited 263 resistance — exact match. |
| T10 | level 1225 | **1124.38** | Slide chart shows level lines at 1124.38 / 1115.68 and a day low of 1121.27. "1225" is a digit typo for ~1125; 1225 is 8.5% away from the extreme and matches nothing. |

Thomas's slides omit the year. All were assumed 2026 and then **verified against
price data** (`verify_levels.py`): every cited level lands on that date's
extension extreme. 18 of 21 matched within 1.5%, so the year inference is
confirmed rather than assumed.

Two levels sit further out:
- **T05** SOXL 2026-06-03 — cited 262 (a 260–262 zone), low 257.26, −1.81%. Zone pierced slightly.
- **T06** SNDK 2026-07-13 — cited 1700, daily low 1646, −3.17%. **Resolved by intraday data**: the setup reversed off 1701.00 at 09:45, matching the citation almost exactly. The 1646 low came hours later.

## The methodology problem T06 exposed — and the fix

**1. The daily extreme is not the setup extreme.** SNDK 2026-07-13 reverted off
1701 in the morning, then broke down to 1646 in the afternoon. Anchoring to the
daily low would have produced the wrong level, wrong stop, and a fabricated
result. All extension and outcome fields are anchored to the **intraday**
extreme located in a window around the annotated ToD (−20/+10 min). The located
extreme falls within a few minutes of the annotation in all 21 cases;
`ext_matches_daily_extreme` flags the 3 where it differs from the daily extreme
(T03, T06, T07).

**2. Risk cannot be measured from the extreme itself.** Doing so assumes a fill
at the exact low, forces MAE to ~0 by construction, and inflated MFE to 5–18R.
The dataset uses a **reference entry**: the first 2-minute bar high (long) /
low (short) break after the extreme — the first trigger on the user's list. Stop
sits 0.1 ATR beyond the extreme. Resulting risk is 12–47% of daily ATR.
`mfe_R_perfect_entry` retains the extreme-anchored ceiling, so the gap between
it and `mfe_R` measures **what the trigger costs you**.

Under this construction **T06 is a loser** — stopped out (`stopped_out=True`,
MAE 1.82R). T03 and T07 close negative. The deck presents all 21 as successes.

## Data sources

- **Daily bars, ATR/MA/BB, RVOL, index & VIX context** — `kti price daily`, 2024-06-01→2026-08-03. VIX requires the ticker `$VIX` (`^VIX`, `VIX`, `.VIX` all return empty).
- **Intraday** — `kti liberator query --dataset minute_bars`, extended hours 04:00–20:00 ET. **Reaches back to ~2012**, with full extended-hours coverage from ~2022. The plan assumed KITE was the only multi-year intraday source; **Liberator removes the KITE dependency entirely**, for Phase 2 as well.
- ATR14 and the 20-day volume/range averages are shifted one bar, so they reflect only what was knowable at the open.

## Judgment calls

- **Sector proxy**: SMH for SOXL/NVDA/MU/INTC/SNDK; QQQ for AAPL/MSFT/PLTR; SPY for BE/CRCL/HOOD/FIG/SPCX. **IBIT has no clean proxy and is mapped to itself** — its `proxy_*` columns are self-referential and must not be read as independent context.
- **Level resolution**: a cited price is `MA` if within 0.35 ATR of a 5/9/20/50/100/200 SMA; else `pivot` (with age and touch count over a 130-session lookback), `round` if a multiple of 25, or `HTF` if untested in 130 sessions. Where no price was cited, the nearest MA to the extreme was used.
- **T08 (FIG)** — the extreme is the 09:30 bar itself, at the search-window edge. The "extension" was an overnight gap, not an intraday drive. Structurally different; consider excluding from intraday-extension statistics.
- **T10 (SNDK 2026-08-03)** is the day the deck was last edited. Its session is incomplete (575 of ~961 minute bars), so `r_at_intraday_swing` / `r_at_overnight_swing` are blank and close-based figures are provisional.
- Empty `level_age_days` / `level_touch_count` (11 rows) are MA-type levels. Empty `close_vs_50sma_pct` / `close_vs_200sma_pct` (2 rows) are SPCX and FIG, too recently listed.

## What the sample suggests — n is far too small to trust

Hypotheses for Phase 2 to test, not findings.

- **The two clusters behave very differently.** Elijah's (later ToD, multi-day exhaustion) median MFE 4.72R and 4.29R at the close; Thomas's (open-drive into a prior-day level) median MFE 2.19R and 1.19R at the close. Supports treating them as **two setups**.
- **VWAP looks like a weak target.** Holding to the close beat exiting at VWAP in 16 of 21 cases (median 2.75R vs 1.41R for longs). VWAP may be leaving most of the move behind.
- **Regime is strikingly uniform**: 19 of 21 in the 15–20 VIX bucket or below, and SPY never in a downtrend. As sampled this is a **benign-tape phenomenon**. Whether it works at VIX >25 or with SPY below its 50-SMA is completely untested — the deck contains no such example. **The single biggest blind spot.**
- Level types split 11 MA / 8 pivot / 2 round, so both detectors in Phase 2 are justified.

## Incident note

The entire `research/mr_extension/` tree was wiped once mid-session by something
outside this work (the `%TEMP%` extraction survived; writes persist normally
before and after). Everything was rebuilt from scripts and re-fetched data, and
the rebuild reproduced the original numbers exactly.

## Files

- `manifest.csv` — hand-encoded deck transcription with corrections
- `build_features.py` — daily features, level resolution, market context
- `intraday_features.py` — extreme location, reference entry, R outcomes
- `pull_intraday.py` — Liberator minute-bar fetch and cache
- `verify_levels.py` — cited-level vs actual-bar verification
- `examples.csv` — the coded dataset
