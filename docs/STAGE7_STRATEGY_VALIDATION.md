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
