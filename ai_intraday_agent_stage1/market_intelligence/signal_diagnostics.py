"""Research-only signal diagnostics. Counts setup availability; does not tune on returns."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import pandas as pd

TIMEZONE = "Asia/Kolkata"


def _market_time(values):
    timestamps = pd.to_datetime(values, errors="coerce")
    if timestamps.dt.tz is None:
        return timestamps.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    return timestamps.dt.tz_convert(TIMEZONE)


def _as_of_timestamp(as_of=None):
    if as_of is None:
        return pd.Timestamp.now(tz=TIMEZONE)
    timestamp = pd.Timestamp(as_of)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(TIMEZONE)
    return timestamp.tz_convert(TIMEZONE)


from market_intelligence.evaluation import evaluate_market_data
from market_intelligence.quality_gate import QualityConfig
from market_intelligence.strategy import generate_setup, StrategyConfig


def diagnose_file(path, symbol, interval=5, warmup=60, stride=1, as_of=None, session_start="09:15", session_end="15:30", post_session_end="16:00"):
    df = pd.read_csv(path)
    raw_rows = int(len(df))
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing OHLCV columns: {sorted(required - set(df.columns))}")
    df["timestamp"] = _market_time(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    cutoff = _as_of_timestamp(as_of)
    completed_mask = df["timestamp"] + pd.Timedelta(minutes=interval) <= cutoff
    uncompleted_rows_excluded = int((~completed_mask).sum())
    completed = df.loc[completed_mask].reset_index(drop=True)
    start_time = pd.to_datetime(session_start).time()
    end_time = pd.to_datetime(session_end).time()
    post_end_time = pd.to_datetime(post_session_end).time()
    if not (start_time < end_time <= post_end_time):
        raise ValueError("session_start must be before session_end, and session_end must not exceed post_session_end")
    in_session = (completed["timestamp"].dt.time >= start_time) & (completed["timestamp"].dt.time < end_time)
    post_session = (completed["timestamp"].dt.time >= end_time) & (completed["timestamp"].dt.time < post_end_time)
    outside_market = (completed["timestamp"].dt.time < start_time) | (completed["timestamp"].dt.time >= post_end_time)
    post_session_rows_excluded = int(post_session.sum())
    outside_market_rows_excluded = int(outside_market.sum())
    df = completed.loc[in_session].reset_index(drop=True)
    status_counts = Counter()
    blocker_counts = Counter()
    no_setup_reasons = Counter()
    insufficient_data_reasons = Counter()
    other_status_reasons = Counter()
    strategy_filter_pass_counts = Counter()
    strategy_filter_fail_counts = Counter()
    candidate_timestamps = []
    samples = 0
    eligible_setup_samples = 0
    last_sampled_timestamp = None

    # Evaluate the same strict data gate as the production research backtester.
    for i in range(max(warmup, 1), len(df), stride):
        window = df.iloc[:i + 1].copy()
        bar_timestamp = window.iloc[-1]["timestamp"]
        now = bar_timestamp + pd.Timedelta(minutes=interval)
        last_sampled_timestamp = str(bar_timestamp)
        qcfg = QualityConfig(
            freshness_minutes=10**9,
            recent_window=min(20, len(window)),
            minimum_recent_quality=0.0,
        )
        evaluation = evaluate_market_data(
            window, interval_minutes=interval, now=now,
            quality_config=qcfg, symbol=symbol,
        )
        setup = generate_setup(evaluation, StrategyConfig())
        status_counts[setup.get("status", "UNKNOWN")] += 1
        samples += 1
        if setup.get("status") == "BLOCKED":
            for reason in setup.get("reasons", []):
                blocker_counts[reason] += 1
        elif setup.get("status") == "INSUFFICIENT_DATA":
            for reason in setup.get("reasons", []):
                insufficient_data_reasons[reason] += 1
        elif setup.get("status") == "NO_SETUP":
            for reason in setup.get("reasons", []):
                no_setup_reasons[reason] += 1
            diag = setup.get("diagnostics") or {}
            passed_checks = diag.get("passed_checks", [])
            failed_checks = diag.get("failed_checks", [])
            eligible_setup_samples += 1
            for check in passed_checks:
                strategy_filter_pass_counts[check] += 1
            for check in failed_checks:
                strategy_filter_fail_counts[check] += 1
                no_setup_reasons[f"failed check: {check}"] += 1
            for reason in diag.get("hard_filter_failures", []):
                no_setup_reasons[f"hard filter: {reason}"] += 1
        elif setup.get("status") == "CANDIDATE":
            candidate_timestamps.append(str(bar_timestamp))
            diag = setup.get("diagnostics") or {}
            passed_checks = setup.get("reasons", [])
            failed_checks = diag.get("failed_checks", [])
            eligible_setup_samples += 1
            for check in passed_checks:
                strategy_filter_pass_counts[check] += 1
            for check in failed_checks:
                strategy_filter_fail_counts[check] += 1
        else:
            for reason in setup.get("reasons", []):
                other_status_reasons[f"{setup.get('status')}: {reason}"] += 1

    return {
        "symbol": symbol,
        "file": str(path),
        "rows": raw_rows,
        "completed_rows": int(len(completed)),
        "session_rows": int(len(df)),
        "uncompleted_rows_excluded": uncompleted_rows_excluded,
        "post_session_rows_excluded": post_session_rows_excluded,
        "outside_market_window_rows_excluded": outside_market_rows_excluded,
        "session": {"start": session_start, "end_exclusive": session_end, "post_session_window_end": post_session_end},
        "as_of_timestamp": str(cutoff),
        "last_sampled_timestamp": last_sampled_timestamp,
        "sampled_bars": samples,
        "stride_bars": stride,
        "status_counts": dict(status_counts),
        "blocker_counts": dict(blocker_counts.most_common(20)),
        "no_setup_reasons": dict(no_setup_reasons.most_common(30)),
        "insufficient_data_reasons": dict(insufficient_data_reasons.most_common(30)),
        "other_status_reasons": dict(other_status_reasons.most_common(30)),
        "eligible_setup_samples": eligible_setup_samples,
        "strategy_filter_pass_counts": dict(strategy_filter_pass_counts.most_common()),
        "strategy_filter_fail_counts": dict(strategy_filter_fail_counts.most_common()),
        "strategy_filter_pass_rates": {
            name: round(strategy_filter_pass_counts[name] / (strategy_filter_pass_counts[name] + strategy_filter_fail_counts[name]), 4)
            for name in sorted(set(strategy_filter_pass_counts) | set(strategy_filter_fail_counts))
            if strategy_filter_pass_counts[name] + strategy_filter_fail_counts[name] > 0
        },
        "candidate_timestamps": candidate_timestamps,
        "paper_only": True,
        "execution": "none",
        "interpretation": "Signal-frequency diagnostic only; not a profitability result.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--warmup-bars", type=int, default=60)
    parser.add_argument("--stride-bars", type=int, default=1)
    parser.add_argument("--as-of", help="Timezone-aware or IST-local cutoff for excluding incomplete candles")
    parser.add_argument("--session-start", default="09:15")
    parser.add_argument("--session-end", default="15:30", help="Exclusive continuous-session end.")
    parser.add_argument("--post-session-end", default="16:00")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = diagnose_file(args.file, args.symbol, args.interval,
                           args.warmup_bars, args.stride_bars, args.as_of,
                           args.session_start, args.session_end, args.post_session_end)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "symbol", "rows", "completed_rows", "session_rows", "post_session_rows_excluded", "sampled_bars", "status_counts",
        "blocker_counts", "no_setup_reasons", "insufficient_data_reasons",
        "other_status_reasons", "strategy_filter_pass_counts",
        "strategy_filter_fail_counts", "strategy_filter_pass_rates", "candidate_timestamps"
    )}, indent=2))
    print("PAPER ONLY | EXECUTION NONE")


if __name__ == "__main__":
    main()
