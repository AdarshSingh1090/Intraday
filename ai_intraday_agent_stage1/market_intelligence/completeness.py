"""Intraday dataset completeness and cross-symbol synchronization audit.

The audit is conservative: it detects gaps within observed sessions and
cross-symbol timestamp mismatches, but does not infer exchange holidays.
Current-session candles that have not completed are excluded from the expected
bar count so a mid-session download is not incorrectly reported as missing the
rest of the trading day.
"""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

OHLCV = ["timestamp", "open", "high", "low", "close", "volume"]
TIMEZONE = "Asia/Kolkata"


def _to_market_time(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    if timestamps.dt.tz is None:
        return timestamps.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    return timestamps.dt.tz_convert(TIMEZONE)


def _as_of_timestamp(as_of=None) -> pd.Timestamp:
    if as_of is None:
        return pd.Timestamp.now(tz=TIMEZONE)
    timestamp = pd.Timestamp(as_of)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(TIMEZONE)
    return timestamp.tz_convert(TIMEZONE)


def _load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path.name}: missing timestamp column")
    frame["timestamp"] = _to_market_time(frame["timestamp"])
    return frame


def audit_intraday_completeness(
    frame: pd.DataFrame,
    interval_minutes: int = 5,
    session_start: str = "09:15",
    session_end: str = "15:30",
    post_session_end: str = "16:00",
    as_of=None,
) -> dict:
    """Audit completed continuous-session candles without counting auction/post-session prints.

    The continuous session is end-exclusive. Rows at or after session_end and
    before post_session_end are reported separately as post-session observations
    and excluded from expected bars. They are not silently repaired or used to
    determine continuous-session completeness.
    """
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    start_time = pd.to_datetime(session_start).time()
    end_time = pd.to_datetime(session_end).time()
    post_end_time = pd.to_datetime(post_session_end).time()
    if not (start_time < end_time <= post_end_time):
        raise ValueError("session_start must be before session_end, and session_end must not exceed post_session_end")
    if frame is None or frame.empty:
        return {"eligible": False, "rows": 0, "observed_dates": 0, "expected_bars_on_observed_dates": 0, "missing_bars": 0, "post_session_observations_excluded": 0, "issues": ["empty dataset"], "per_date": {}}
    if "timestamp" not in frame.columns:
        return {"eligible": False, "rows": len(frame), "observed_dates": 0, "expected_bars_on_observed_dates": 0, "missing_bars": 0, "post_session_observations_excluded": 0, "issues": ["missing timestamp column"], "per_date": {}}
    as_of_ts = _as_of_timestamp(as_of)
    ts = _to_market_time(frame["timestamp"]).dropna()
    if ts.empty:
        return {"eligible": False, "rows": len(frame), "observed_dates": 0, "expected_bars_on_observed_dates": 0, "missing_bars": 0, "post_session_observations_excluded": 0, "issues": ["no valid timestamps"], "per_date": {}}
    per_date, issues = {}, []
    expected_total = missing_total = misaligned_total = outside_total = 0
    uncompleted_total = post_session_total = 0
    full_session_bars = len(pd.date_range(pd.Timestamp(f"2000-01-03 {session_start}", tz=TIMEZONE), pd.Timestamp(f"2000-01-03 {session_end}", tz=TIMEZONE), freq=f"{interval_minutes}min", inclusive="left"))
    for day, values in ts.groupby(ts.dt.date):
        values = values.sort_values()
        day_start = pd.Timestamp(f"{day} {session_start}", tz=TIMEZONE)
        day_end = pd.Timestamp(f"{day} {session_end}", tz=TIMEZONE)
        post_end = pd.Timestamp(f"{day} {post_session_end}", tz=TIMEZONE)
        current_day = day == as_of_ts.date()
        not_started = current_day and as_of_ts <= day_start
        partial = current_day and day_start < as_of_ts < day_end
        if not_started:
            cutoff, expected = day_start, pd.DatetimeIndex([])
        elif partial:
            cutoff = min(day_end, as_of_ts.floor(f"{interval_minutes}min"))
            expected = pd.date_range(day_start, cutoff, freq=f"{interval_minutes}min", inclusive="left")
        else:
            cutoff = day_end
            expected = pd.date_range(day_start, day_end, freq=f"{interval_minutes}min", inclusive="left")
        observed = pd.DatetimeIndex(values)
        in_session = observed[(observed >= day_start) & (observed < day_end)]
        post_session = observed[(observed >= day_end) & (observed < post_end)]
        pre_session = observed[observed < day_start]
        outside_after_close = observed[observed >= post_end]
        outside = len(pre_session) + len(outside_after_close)
        if partial or not_started:
            completed = in_session[in_session < cutoff]
            uncompleted = in_session[in_session >= cutoff]
        else:
            completed, uncompleted = in_session, pd.DatetimeIndex([])
        missing = len(expected.difference(completed))
        misaligned = int(((in_session-day_start) % pd.Timedelta(minutes=interval_minutes) != pd.Timedelta(0)).sum())
        duplicates = int(in_session.duplicated().sum())
        expected_total += len(expected); missing_total += missing; misaligned_total += misaligned
        outside_total += outside; uncompleted_total += len(uncompleted); post_session_total += len(post_session)
        date_issues=[]
        if missing: date_issues.append(f"missing completed bars: {missing} (continuous session)")
        if outside: date_issues.append(f"outside market window: {outside}")
        if misaligned: date_issues.append(f"misaligned continuous-session timestamps: {misaligned}")
        if duplicates: date_issues.append(f"duplicate continuous-session timestamps: {duplicates}")
        per_date[str(day)]={"expected_bars":len(expected),"full_session_expected_bars":full_session_bars,
            "partial_session":bool(partial or not_started),"session_not_started":bool(not_started),"audit_cutoff":str(cutoff),
            "observed_bars_in_session":len(in_session),"observed_completed_bars":len(completed),
            "uncompleted_bars_excluded":len(uncompleted),"post_session_bars_excluded":len(post_session),
            "pre_session_bars":len(pre_session),"outside_after_post_close_bars":len(outside_after_close),
            "missing_bars":missing,"outside_session_bars":outside,"misaligned_bars":misaligned,"duplicate_bars":duplicates,
            "first_observed":str(in_session.min()) if len(in_session) else None,
            "last_observed":str(in_session.max()) if len(in_session) else None,
            "last_completed_observed":str(completed.max()) if len(completed) else None,"issues":date_issues}
        if date_issues: issues.append(f"{day}: "+"; ".join(date_issues))
    return {"eligible":not issues,"rows":len(frame),"as_of_timestamp":str(as_of_ts),
        "session":{"start":session_start,"end_exclusive":session_end,"post_session_window_end":post_session_end},
        "observed_dates":len(per_date),"expected_bars_on_observed_dates":expected_total,
        "full_session_expected_bars_per_observed_date":full_session_bars,
        "observed_bars_in_session":sum(v["observed_bars_in_session"] for v in per_date.values()),
        "observed_completed_bars":sum(v["observed_completed_bars"] for v in per_date.values()),
        "uncompleted_bars_excluded":uncompleted_total,"post_session_observations_excluded":post_session_total,
        "missing_bars":missing_total,"misaligned_bars":misaligned_total,"outside_session_bars":outside_total,
        "issues":issues,"per_date":per_date,
        "note":"Audits observed dates only; excludes post-session/auction observations from continuous-session coverage; does not infer holidays or missing whole trading sessions and never fills gaps."}

def audit_research_directory(
    directory: str | Path,
    interval_minutes: int = 5,
    reference_symbol: str | None = None,
    session_start: str = "09:15",
    session_end: str = "15:30",
    post_session_end: str = "16:00",
    as_of=None,
) -> dict:
    """Audit every CSV and timestamp synchronization across completed bars."""
    root = Path(directory)
    files = sorted(root.glob("*.csv"))
    as_of_ts = _as_of_timestamp(as_of)
    symbols = {}
    for path in files:
        frame = _load_csv(path)
        timestamps = pd.DatetimeIndex(frame["timestamp"].dropna())
        symbols[path.stem] = {
            "rows": len(frame),
            "completeness": audit_intraday_completeness(
                frame, interval_minutes, session_start=session_start,
                session_end=session_end, post_session_end=post_session_end, as_of=as_of_ts,
            ),
            "timestamps": set(timestamps),
        }

    if not symbols:
        report = {
            "eligible": False, "symbols": {}, "issues": ["no CSV datasets found"],
            "as_of_timestamp": str(as_of_ts),
        }
    else:
        chosen = reference_symbol
        if chosen is not None and chosen not in symbols:
            raise ValueError(f"Reference symbol/file not found: {chosen}")
        if chosen is None:
            chosen = max(symbols, key=lambda s: symbols[s]["rows"])
        reference = symbols[chosen]["timestamps"]
        # For the current partial session, synchronize completed bars only.
        current_day_start = pd.Timestamp(f"{as_of_ts.date()} {session_start}", tz=TIMEZONE)
        current_day_end = pd.Timestamp(f"{as_of_ts.date()} {session_end}", tz=TIMEZONE)
        partial_current_session = current_day_start < as_of_ts < current_day_end
        cutoff = as_of_ts.floor(f"{interval_minutes}min") if partial_current_session else current_day_end
        start_time = pd.to_datetime(session_start).time()
        end_time = pd.to_datetime(session_end).time()

        def completed_timestamps(timestamp_set):
            continuous = {t for t in timestamp_set if start_time <= t.time() < end_time}
            if not partial_current_session:
                return continuous
            return {t for t in continuous if t.date() != as_of_ts.date() or t < cutoff}

        reference_completed = completed_timestamps(reference)
        synchronization = {}
        for symbol, entry in symbols.items():
            ts_set = completed_timestamps(entry["timestamps"])
            synchronization[symbol] = {
                "reference": chosen,
                "missing_reference_timestamps": len(reference_completed - ts_set),
                "extra_timestamps_vs_reference": len(ts_set - reference_completed),
                "matched_timestamps": len(reference_completed & ts_set),
            }
        aggregate_issues = []
        for symbol, entry in symbols.items():
            audit = entry["completeness"]
            if not audit["eligible"]:
                issue_types = []
                if audit["missing_bars"]:
                    issue_types.append("missing bars")
                if audit["misaligned_bars"]:
                    issue_types.append("misaligned timestamps")
                if audit["outside_session_bars"]:
                    issue_types.append("outside-session bars")
                if any(date["duplicate_bars"] for date in audit["per_date"].values()):
                    issue_types.append("duplicate timestamps")
                aggregate_issues.append(
                    f"{symbol}: completeness audit failed ({', '.join(issue_types) or 'see per-date issues'})"
                )
            sync = synchronization[symbol]
            if sync["missing_reference_timestamps"] or sync["extra_timestamps_vs_reference"]:
                aggregate_issues.append(
                    f"{symbol}: timestamp synchronization mismatch vs {chosen} "
                    f"(missing={sync['missing_reference_timestamps']}, "
                    f"extra={sync['extra_timestamps_vs_reference']})"
                )
        report = {
            "eligible": not aggregate_issues,
            "symbols": {s: {k: v for k, v in e.items() if k != "timestamps"}
                        for s, e in symbols.items()},
            "reference_symbol": chosen,
            "synchronization": synchronization,
            "issues": aggregate_issues,
            "as_of_timestamp": str(as_of_ts),
            "synchronization_scope": "completed bars only; unfinished current-session bars excluded",
            "session": {"start": session_start, "end_exclusive": session_end, "post_session_window_end": post_session_end},
            "note": "Cross-symbol comparison is limited to completed continuous-session timestamps; post-session/auction observations are excluded. This is not an exchange calendar.",
        }
    out_path = root / "completeness_audit.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_file"] = str(out_path)
    return report
