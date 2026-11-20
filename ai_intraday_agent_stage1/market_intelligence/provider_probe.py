"""Compare two Yahoo Finance intraday retrieval paths for research-data gaps.

Research-only diagnostic. Does not repair data or generate trades.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

TIMEZONE = "Asia/Kolkata"


def expected_session_grid(day, interval_minutes=5, timezone=TIMEZONE):
    start = pd.Timestamp(datetime.combine(day, time(9, 15)), tz=timezone)
    return pd.date_range(start=start, periods=75, freq=f"{interval_minutes}min")


def normalize_provider_frame(frame):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if "timestamp" not in data.columns:
        data = data.reset_index()
    aliases = {str(c).lower(): c for c in data.columns}
    time_col = next((aliases[k] for k in ("timestamp", "datetime", "date") if k in aliases), None)
    if time_col is None:
        raise ValueError("Provider frame lacks timestamp column")
    rename = {time_col: "timestamp"}
    for c in ("open", "high", "low", "close", "volume"):
        if c not in aliases:
            raise ValueError(f"Provider frame lacks {c}")
        rename[aliases[c]] = c
    data = data.rename(columns=rename)[["timestamp", "open", "high", "low", "close", "volume"]]
    ts = pd.to_datetime(data.timestamp, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(TIMEZONE)
    else:
        ts = ts.dt.tz_convert(TIMEZONE)
    data["timestamp"] = ts
    return data.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def summarize_provider_frame(frame, as_of=None, interval_minutes=5):
    data = normalize_provider_frame(frame)
    now = pd.Timestamp(as_of or datetime.now(ZoneInfo(TIMEZONE)))
    if now.tzinfo is None:
        now = now.tz_localize(TIMEZONE)
    else:
        now = now.tz_convert(TIMEZONE)
    observed_dates = sorted(set(data.timestamp.dt.date))
    missing_by_date = {}
    for day in observed_dates:
        expected = expected_session_grid(day, interval_minutes)
        completed = expected[expected + pd.Timedelta(minutes=interval_minutes) <= now]
        observed = set(data.loc[data.timestamp.dt.date == day, "timestamp"])
        missing = [t for t in completed if t not in observed]
        missing_by_date[str(day)] = [t.strftime("%H:%M") for t in missing]
    times = [t for values in missing_by_date.values() for t in values]
    return {
        "rows": int(len(data)),
        "duplicate_timestamps": int(data.timestamp.duplicated().sum()),
        "first_timestamp": str(data.timestamp.min()) if len(data) else None,
        "last_timestamp": str(data.timestamp.max()) if len(data) else None,
        "observed_dates": len(observed_dates),
        "missing_completed_bars": len(times),
        "missing_time_frequency": {t: times.count(t) for t in sorted(set(times))},
        "missing_by_date": missing_by_date,
        "excluded_uncompleted_expected_bars": int(sum(
            len(expected_session_grid(day, interval_minutes)) -
            len(expected_session_grid(day, interval_minutes)[
                expected_session_grid(day, interval_minutes) + pd.Timedelta(minutes=interval_minutes) <= now
            ]) for day in observed_dates
        )),
    }


def compare_frames(primary, alternate, as_of=None, interval_minutes=5):
    a, b = normalize_provider_frame(primary), normalize_provider_frame(alternate)
    ta, tb = set(a.timestamp), set(b.timestamp)
    return {
        "primary": summarize_provider_frame(a, as_of, interval_minutes),
        "alternate": summarize_provider_frame(b, as_of, interval_minutes),
        "timestamp_comparison": {
            "primary_only_timestamps": len(ta - tb),
            "alternate_only_timestamps": len(tb - ta),
            "matched_timestamps": len(ta & tb),
        },
    }


def run_probe(symbols, start, end, output):
    import yfinance as yf
    reports = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(start=start, end=end, interval="5m", auto_adjust=False, prepost=False)
            download = yf.download(symbol, start=start, end=end, interval="5m", auto_adjust=False,
                                  prepost=False, progress=False, group_by="column")
            reports[symbol] = compare_frames(history, download)
        except Exception as exc:
            reports[symbol] = {"error": f"{type(exc).__name__}: {exc}"}
    result = {
        "purpose": "compare Yahoo retrieval paths; no data repair or strategy tuning",
        "provider": "Yahoo Finance via yfinance",
        "start": start,
        "end_exclusive": end,
        "interval_minutes": 5,
        "timezone": TIMEZONE,
        "reports": reports,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="RELIANCE.NS,TCS.NS,HDFCBANK.NS,ICICIBANK.NS,INFY.NS")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", default="reports/yahoo_provider_probe.json")
    args = parser.parse_args()
    result = run_probe([s.strip() for s in args.symbols.split(",") if s.strip()], args.start, args.end, args.output)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
