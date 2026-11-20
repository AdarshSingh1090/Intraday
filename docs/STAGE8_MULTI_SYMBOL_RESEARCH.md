# Stage 8 — Multi-symbol historical research data

## First milestone
Build a reproducible historical-data collection layer before strategy evaluation. It downloads each requested Yahoo ticker independently, normalizes candles to the canonical OHLCV schema, performs a strict audit, saves accepted files, and writes a manifest. A failed ticker does not abort the batch.

## Usage
From `ai_intraday_agent_stage1`:

```powershell
python main.py fetch-research-data --symbols RELIANCE.NS,TCS.NS,HDFCBANK.NS,ICICIBANK.NS,INFY.NS --interval 5 --start-date 2026-01-01 --end-date 2026-02-01 --output-dir data/research
```

The end date follows provider semantics (end-exclusive). Start with a smaller liquid basket and inspect `manifest.json` before backtesting. The CLI will report saved, rejected, and errored tickers.

## Limitations and controls
- Yahoo Finance intraday history has provider-specific retention/availability limits. A requested date range may fail or return no data; failures are recorded rather than hidden.
- This is not a broker-grade data source. Validate timestamp timezone, session completeness, missing bars, corporate actions, splits, and volume before using data for research.
- Accepted here means basic structural/ohlcv checks passed, not that the dataset is complete or statistically suitable.
- No live execution, broker API, order placement, or automated capital deployment is added.
- Keep final test data untouched until strategy selection is complete. The Stage 7 chronological experiment/walk-forward tools should be used next.

## Acceptance checklist
- [ ] Fetch at least 10 liquid NSE symbols for a consistent date window.
- [ ] Review all per-symbol audit reports and missing-session behavior.
- [ ] Save a data-source snapshot and provider metadata.
- [ ] Run baseline strategies and cost stress tests on chronological splits.


## Stage 8.1 — completeness audit

After downloading per-symbol CSV files, run:

```powershell
python main.py audit-research-data --directory data/research --interval 5
```

The audit checks each observed session for missing 5-minute bars, duplicate bars,
timestamps outside 09:15–15:30 IST, and timestamp alignment. It also compares
symbols against a reference dataset (the largest file by default) to identify
cross-symbol timestamp mismatches. It writes `completeness_audit.json`.

**Important:** this is a conservative completeness check, not an NSE exchange
calendar. It audits dates present in each dataset and cannot identify a missing
entire trading session or distinguish an exchange holiday from a whole-day data
gap. A dataset that passes structural and session checks is not automatically
suitable for backtesting; inspect the report and source quality.
