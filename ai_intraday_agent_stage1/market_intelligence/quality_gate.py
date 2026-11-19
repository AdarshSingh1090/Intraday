"""Market-data quality scoring and trade-eligibility gate.

This module flags suspicious provider data without deleting or rewriting candles.
A quality score is a heuristic, not a prediction of profitability.
"""
from __future__ import annotations

from typing import Any
import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class QualityConfig:
    def __init__(self, freshness_minutes=10, recent_window=20,
                 minimum_recent_quality=0.90, flat_high_volume_multiple=2.5,
                 flat_low_volume_fraction=0.10, flat_low_volume_floor=500.0):
        self.freshness_minutes = freshness_minutes
        self.recent_window = recent_window
        self.minimum_recent_quality = minimum_recent_quality
        self.flat_high_volume_multiple = flat_high_volume_multiple
        self.flat_low_volume_fraction = flat_low_volume_fraction
        self.flat_low_volume_floor = flat_low_volume_floor


def _empty_result(message):
    return {
        "status": "no_data", "message": message, "quality_score": 0.0,
        "trade_eligible": False, "blocking_reasons": [message],
        "summary": {}, "latest": None, "annotated": pd.DataFrame(),
    }


def assess_candles(df, interval_minutes=5, now=None, config=None):
    """Assess candle quality and latest-data eligibility. Never mutates input."""
    config = config or QualityConfig()
    if df is None or df.empty:
        return _empty_result("Dataset is empty.")
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        return _empty_result("Missing required columns: " + ", ".join(missing_columns))

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    for col in REQUIRED_COLUMNS[1:]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.sort_values("timestamp", na_position="last").reset_index(drop=True)

    flag_names = [
        "invalid_timestamp", "null_values", "invalid_ohlc", "negative_volume",
        "zero_volume", "flat_low_volume", "flat_high_volume", "duplicate_timestamp"
    ]
    flags = {name: pd.Series(False, index=work.index) for name in flag_names}
    flags["invalid_timestamp"] = work["timestamp"].isna()
    flags["null_values"] = work[REQUIRED_COLUMNS].isna().any(axis=1)
    valid_numeric = work[REQUIRED_COLUMNS[1:]].notna().all(axis=1)
    o, h, l, c, v = (work[col] for col in ["open", "high", "low", "close", "volume"])
    flags["invalid_ohlc"] = valid_numeric & (
        (h < l) | (h < o) | (h < c) | (l > o) | (l > c) |
        (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
    )
    flags["negative_volume"] = v < 0
    flags["zero_volume"] = v == 0
    flags["duplicate_timestamp"] = work["timestamp"].duplicated(keep=False) & work["timestamp"].notna()

    flat = valid_numeric & ((h - l) == 0)
    positive_volumes = work.loc[valid_numeric & (v > 0), "volume"]
    volume_median = float(positive_volumes.median()) if not positive_volumes.empty else 0.0
    low_threshold = max(config.flat_low_volume_floor, volume_median * config.flat_low_volume_fraction)
    high_threshold = volume_median * config.flat_high_volume_multiple if volume_median else float("inf")
    flags["flat_low_volume"] = flat & (v > 0) & (v < low_threshold)
    flags["flat_high_volume"] = flat & (v >= high_threshold) & (v > 0)

    for name, series in flags.items():
        work[name] = series.fillna(False).astype(bool)
    work["quality_flags"] = work.apply(
        lambda row: [name.upper() for name in flag_names if bool(row[name])], axis=1
    )
    work["candle_quality_score"] = 100.0
    penalties = {
        "invalid_timestamp": 40, "null_values": 35, "invalid_ohlc": 40,
        "negative_volume": 40, "zero_volume": 10, "flat_low_volume": 10,
        "flat_high_volume": 20, "duplicate_timestamp": 15,
    }
    for name, penalty in penalties.items():
        work["candle_quality_score"] -= work[name].astype(float) * penalty
    work["candle_quality_score"] = work["candle_quality_score"].clip(0, 100)

    gap_records = []
    total_missing = 0
    if interval_minutes > 0:
        valid_time_rows = work.loc[work["timestamp"].notna()]
        for day, day_df in valid_time_rows.groupby(valid_time_rows["timestamp"].dt.date):
            ts = day_df["timestamp"].drop_duplicates().sort_values()
            if len(ts) < 2:
                continue
            expected = pd.date_range(ts.iloc[0], ts.iloc[-1], freq=f"{interval_minutes}min")
            missing = expected.difference(pd.DatetimeIndex(ts))
            if len(missing):
                gap_records.append({
                    "date": str(day), "missing_count": int(len(missing)),
                    "missing_timestamps": [str(x) for x in missing],
                })
                total_missing += len(missing)

    score = float(work["candle_quality_score"].mean())
    score = max(0.0, score - min(20.0, total_missing * 2.0))
    valid_timestamps = work["timestamp"].dropna()
    if valid_timestamps.empty:
        return _empty_result("No valid timestamps.")

    latest_idx = work["timestamp"].last_valid_index()
    latest_row = work.loc[latest_idx]
    latest_ts = latest_row["timestamp"]
    if now is None:
        current = pd.Timestamp.now(tz=latest_ts.tz) if latest_ts.tz is not None else pd.Timestamp.now()
    else:
        current = pd.Timestamp(now)
    if latest_ts.tz is not None and current.tz is None:
        current = current.tz_localize(latest_ts.tz)
    elif latest_ts.tz is None and current.tz is not None:
        current = current.tz_localize(None)
    age_minutes = max(0.0, (current - latest_ts).total_seconds() / 60.0)

    recent = work.tail(config.recent_window)
    recent_quality = float((recent["candle_quality_score"] >= 80).mean()) if len(recent) else 0.0
    latest_flags = latest_row["quality_flags"]
    blocking = []
    if age_minutes > config.freshness_minutes:
        blocking.append(f"Latest candle is stale ({age_minutes:.1f} minutes old).")
    if latest_flags:
        blocking.append("Latest candle has quality flags: " + ", ".join(latest_flags))
    if recent_quality < config.minimum_recent_quality:
        blocking.append(f"Recent candle quality is below threshold ({recent_quality:.1%}).")
    latest_date = latest_ts.date()
    latest_day_gaps = [g for g in gap_records if g["date"] == str(latest_date)]
    latest_day_missing = sum(g["missing_count"] for g in latest_day_gaps)
    recent_flags = work.tail(config.recent_window)
    recent_anomalies = int(recent_flags[["zero_volume", "flat_low_volume", "flat_high_volume", "duplicate_timestamp"]].any(axis=1).sum())
    if latest_day_missing:
        blocking.append(f"Missing candles in latest trading session: {latest_day_missing}.")
    if recent_anomalies:
        blocking.append(f"Suspicious candles/duplicates in recent window: {recent_anomalies}.")
    if bool(latest_row["invalid_timestamp"] or latest_row["null_values"] or latest_row["invalid_ohlc"] or latest_row["negative_volume"]):
        blocking.append("Latest candle has a structural data error.")

    trade_eligible = not blocking
    summary = {
        "rows": len(work),
        "invalid_timestamps": int(work["invalid_timestamp"].sum()),
        "null_rows": int(work["null_values"].sum()),
        "invalid_ohlc_rows": int(work["invalid_ohlc"].sum()),
        "negative_volume_candles": int(work["negative_volume"].sum()),
        "zero_volume_candles": int(work["zero_volume"].sum()),
        "flat_low_volume_candles": int(work["flat_low_volume"].sum()),
        "flat_high_volume_candles": int(work["flat_high_volume"].sum()),
        "duplicate_timestamps": int(work["duplicate_timestamp"].sum()),
        "missing_candles_within_observed_ranges": int(total_missing),
        "missing_candle_details": gap_records,
        "latest_session_missing_candles": int(latest_day_missing),
        "recent_suspicious_candles": recent_anomalies,
        "recent_quality_fraction": round(recent_quality, 4),
        "median_positive_volume": volume_median,
        "flat_low_volume_threshold": low_threshold,
        "flat_high_volume_threshold": high_threshold,
    }
    latest = {
        "timestamp": str(latest_ts), "age_minutes": round(age_minutes, 2),
        "quality_score": float(latest_row["candle_quality_score"]),
        "flags": latest_flags,
        "close": float(latest_row["close"]) if pd.notna(latest_row["close"]) else None,
        "volume": float(latest_row["volume"]) if pd.notna(latest_row["volume"]) else None,
    }
    return {
        "status": "PASSED" if trade_eligible else "WARNING",
        "quality_score": round(score, 2), "trade_eligible": trade_eligible,
        "blocking_reasons": blocking, "summary": summary, "latest": latest,
        "annotated": work,
    }


def print_quality_report(report):
    print("\n" + "=" * 58)
    print("          MARKET DATA QUALITY / TRADE GATE")
    print("=" * 58)
    if report.get("message"):
        print(report["message"])
        return
    print(f"Status: {report['status']}")
    print(f"Quality score: {report['quality_score']}/100 (heuristic)")
    print(f"Trade eligible: {'YES' if report['trade_eligible'] else 'NO'}")
    print("\n--- Latest candle ---")
    latest = report["latest"]
    print(f"Timestamp: {latest['timestamp']}")
    print(f"Age: {latest['age_minutes']} minutes")
    print(f"Candle score: {latest['quality_score']}/100")
    print(f"Flags: {', '.join(latest['flags']) if latest['flags'] else 'None'}")
    print("\n--- Dataset summary ---")
    for key, value in report["summary"].items():
        if key != "missing_candle_details":
            print(f"{key}: {value}")
    if report["blocking_reasons"]:
        print("\n--- BLOCKING REASONS ---")
        for reason in report["blocking_reasons"]:
            print(f"- {reason}")
    print("=" * 58)
