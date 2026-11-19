# Stage 7 — Strategy Validation Protocol & Initial Audit

## Scope
Research-only. Paper simulation only. No broker API or live execution is introduced.

## Confirmed implementation observations (branch stage2-data-quality-investigation)
- Signal evaluation uses bars through index t and forwards bars beginning at t+1.
- Forward simulation is filtered to the signal candle's calendar date, reducing overnight exposure.
- The simulator applies adverse slippage on both entry and exit, estimated percentage fees, and optional flat fees.
- If stop and target are touched within the same candle, the simulator assumes stop first.
- Stop gaps are filled at the opening price (worse than stop); target fills are currently at target price.
- Backtester skips overlapping positions by resuming after the simulated exit candle.
- Quality gate is called on the historical window with freshness effectively disabled and minimum recent quality set to zero; other anomaly/gap blockers still apply.

## Audit findings / risks to resolve
1. **Drawdown definition:** current metric is absolute peak-to-trough of cumulative trade P&L starting at zero. It is not portfolio-equity drawdown and has no percentage denominator.
2. **No equity-based return metrics:** starting/ending equity, percentage return, daily returns, Sharpe/Sortino, or risk-adjusted return are absent.
3. **No statistical uncertainty:** no expectancy confidence interval or sample-size caution.
4. **Cost/risk integration:** risk sizing uses signal close/reference; entry gap can alter exposure and stop risk. Fees are not included in stop-risk budget. Capital/exposure limits need to be rechecked at simulated fill.
5. **Input cleaning:** malformed prices are removed; duplicate timestamps are silently deduplicated; sorting and dropping can conceal dataset defects. A strict validation report should precede backtesting.
6. **Intrabar assumptions:** OHLC candles cannot reveal stop/target ordering when both are hit. Stop-first is conservative, but must be explicitly reported as an assumption. Target fills may be optimistic under gaps/slippage/queue effects.
7. **Trading-session assumptions:** date-only filtering can include out-of-session bars if the source includes them. Session boundaries, holidays, timezone, and candle completeness need explicit validation.
8. **No benchmark or trade attribution:** no baseline comparison or breakdown by symbol, side, regime, month, volatility, time-of-day, and setup score.
9. **No out-of-sample workflow:** parameter selection and final reporting must be separated by chronological periods.
10. **Potential strategy/data issue:** relative volume is computed from the full historical window's screening logic; confirm its rolling calculation does not include future bars. Indicators should be strictly causal and signal only on completed bars.

## Required experiment design
- Chronological split: development / validation / final test; never tune on final test.
- Walk-forward evaluation: roll training/validation windows forward, report each fold and aggregate.
- Purged boundaries: ensure trades do not cross evaluation period boundaries; avoid overlapping samples.
- Minimum data: multiple symbols, several months including trend, range, high-volatility and low-volatility regimes.
- Costs: base, conservative, and stress cases for slippage and fees.
- Benchmarks: no-trade (zero), buy-and-hold over same sessions, and simple directional baseline where applicable.
- Robustness: parameter sensitivity grid; avoid selecting a single best point. Report neighborhood stability.
- Reporting: trades, win rate, profit factor, net expectancy, average/median R, total return, equity drawdown (absolute and percent), Sharpe/Sortino (with caveats), trade distribution, confidence intervals, and segment breakdowns.

## Acceptance gates (research gates, not promises)
- No known look-ahead/data leakage.
- Data validation passes and all assumptions are explicit.
- Positive net expectancy after costs on unseen test periods.
- Profit factor > 1.0 on test (preferably meaningfully above, not marginal).
- Drawdown is within predeclared risk tolerance.
- Performance is not driven by one symbol, one day, or a small number of outliers.
- Results remain directionally robust under higher costs and modest parameter perturbations.
- Sample size and uncertainty are disclosed; insufficient evidence means continue paper testing or reject.

## Next implementation steps
1. Add performance evaluation module and tests.
2. Attach risk/R metrics to simulated trades.
3. Harden historical input validation and session controls.
4. Add benchmark, period, symbol, side, and regime breakdowns.
5. Add chronological experiment runner and CLI report/export.

## Stage 7 implementation status (current draft)
- `performance.py`: equity-based return/drawdown, expectancy, median trade, win/loss ratio, daily P&L, daily Sharpe/Sortino (caveated), approximate Calmar, R multiples, deterministic trade bootstrap interval.
- `experiment.py`: chronological split by trading date, expanding/rolling walk-forward fold generator, and parameter sensitivity runner.
- Backtester now rejects invalid/missing OHLCV, duplicate timestamps, inconsistent OHLC, and invalid configuration rather than silently removing or deduplicating rows.
- Paper simulator now rejects malformed forward bars and duplicate timestamps, checks stop-loss risk with estimated slippage and fees, and enforces a supplied actual-entry capital ceiling.
- CLI backtest command added with optional JSON report and trade CSV export.
- Initial CI run failed on two pre-existing test fixtures that contained invalid OHLC (close above high; short candle high below open). The fixtures were corrected to valid OHLC; subsequent run for that commit passed. Newer Stage 7 commits are still being checked.

## Additional causal indicator audit
- EMA, RSI, ATR, price returns, EMA slope and rolling volume calculations use current/prior rows only; no explicit future-row lookups found in these calculations.
- Relative volume denominator currently uses a 20-bar rolling mean **including the current candle**, which is causal but can dilute the current volume spike. A prior-20-bar baseline should be tested as a separate variant.
- EMA/RSI/ATR and volume averages are computed continuously across calendar/session boundaries. This may be intentional, but opening-session behavior and intraday volume seasonality need separate research.
- Regime/strategy thresholds are hand-set and have not been statistically validated.
- Confidence is a rule score percentage, not a calibrated probability.

## Further framework additions
- Backtest supports inclusive signal date bounds while retaining earlier candles for indicator warm-up; forward exits are bounded by the requested end date.
- Timestamps are normalized to a configurable market timezone (default `Asia/Kolkata`); regular session is configurable (default 09:15 to 15:30, end-exclusive). Out-of-session rows are excluded and counted in the report.
- Simulator also normalizes forward timestamps to the configured timezone.
- Benchmark comparisons now include no-trade, buy-and-hold, and daily open-to-close baselines with the same simple estimated cost assumptions.
- Trades retain trend/regime, score, relative volume, RSI, time bucket, month, and weekday metadata for attribution.
- Reports include segment metrics by symbol, direction, exit reason, trend, trend strength, volatility, time-of-day, month, and weekday.

## CLI
```powershell
python main.py backtest --file historical.csv --symbol RELIANCE --interval 5 --budget 8900 --start-date 2026-01-01 --end-date 2026-03-31 --report-json reports/reliance_test.json --trades-csv reports/reliance_trades.csv
```
Earlier candles are retained for indicator warm-up; only signals in the requested date range are counted. This is essential for chronological out-of-sample evaluation.

- Entry-gap validation now rechecks actual reward-risk after slippage; candidates that fall below the minimum are rejected before simulation.

- Chronological split utilities return validation/test datasets plus `*_with_warmup` views that include only prior history through the evaluation period's end. Run the backtester on these views with matching `start_date`/`end_date` so indicators have history without counting earlier trades.

- Added orchestration helpers for walk-forward validation and separate multi-symbol backtests. Combined multi-symbol trade statistics are explicitly descriptive and **not** a portfolio simulation (no cross-symbol capital/exposure coordination).

## Automated chronological experiment CLI
```powershell
python main.py research --file data/processed/RELIANCE_5_candles.csv --symbol RELIANCE --interval 5 --train-fraction 0.60 --validation-fraction 0.20 --budget 8900 --report-json reports/reliance_experiment.json
```
This runs separate train, validation and final test periods. Validation and test keep earlier candles for indicator warm-up but restrict signals/exits to their respective date ranges. **Do not tune on the final test period.**
