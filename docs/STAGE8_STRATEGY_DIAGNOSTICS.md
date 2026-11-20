# Stage 8.2 — Strategy Filter Diagnostics

This diagnostic evaluates the current rule-based strategy filters across historical OHLCV CSV files. It is research-only and does not simulate trades, calculate returns, relax rules, or automatically select a configuration.

## Run

From the `ai_intraday_agent_stage1` directory:

```bash
python main.py strategy-diagnostics --directory data/research --report-json reports/strategy_diagnostics.json
```

The directory should contain per-symbol OHLCV CSVs (timestamp, open, high, low, close, volume).

## Report contents

- Valid indicator observations and directional trend-aligned observations.
- Pass/fail counts and pass rate for trend strength, EMA alignment, RSI range, momentum, relative volume, EMA extension, and candle confirmation.
- Bottleneck ordering: lowest pass rate first, descriptive only.
- Filter combinations: all filters, each individual filter removed, every pair removed, RSI+relative-volume hard filters removed, trend gates removed, and no filters.
- The strategy's actual baseline rule: both hard filters (RSI and relative volume) plus configured minimum score.
- Per-observation diagnostic rows and symbol-level plus aggregate counts.

## Interpretation cautions

- Counts do not imply profitability or statistical significance.
- Filter removal is a what-if diagnostic, not a recommendation.
- No winner or preferred parameter set is produced.
- Indicators are calculated chronologically using current/prior data; no forward outcome is used.
- The diagnostic universe is observations oriented by EMA20 slope sign (long when positive, short when negative), so EMA alignment is tested independently. It separately reports valid-indicator rows.
- Data completeness and session coverage should be assessed separately before drawing conclusions.
- Paper-only; no broker execution.
