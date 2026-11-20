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
    as_of=None,
) -> dict:
    """Audit missing bars, session boundaries, duplicates, and alignment.

    Session end is exclusive. Historical observed dates are checked against a
    full session. If the latest/current date is being audited before session
    close, only completed candle start-times up to the current interval boundary
    are expected. The forming candle (and future session slots) are reported as
    uncompleted, not missing. Holidays and absent whole sessions cannot be
    inferred from data alone.
    """
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    if frame is None or frame.empty:
        return {
            "eligible": False, "rows": 0, "observed_dates": 0,
            "expected_bars_on_observed_dates": 0, "missing_bars": 0,
            "issues": ["empty dataset"], "per_date": {},
        }
    if "timestamp" not in frame.columns:
        return {
            "eligible": False, "rows": len(frame), "observed_dates": 0,
            "expected_bars_on_observed_dates": 0, "missing_bars": 0,
            "issues": ["missing timestamp column"], "per_date": {},
        }

    as_of_ts = _as_of_timestamp(as_of)
    ts = _to_market_time(frame["timestamp"]).dropna()
    if ts.empty:
        return {
            "eligible": False, "rows": len(frame), "observed_dates": 0,
            "expected_bars_on_observed_dates": 0, "missing_bars": 0,
            "issues": ["no valid timestamps"], "per_date": {},
        }

    per_date = {}
    issues = []
    expected_total = 0
    missing_total = 0
    misaligned_total = 0
    outside_total = 0
    uncompleted_total = 0
    full_session_bars = len(pd.date_range(
        pd.Timestamp(f"2000-01-03 {session_start}", tz=TIMEZONE),
        pd.Timestamp(f"2000-01-03 {session_end}", tz=TIMEZONE),
        freq=f"{interval_minutes}min", inclusive="left",
    ))

    for day, values in ts.groupby(ts.dt.date):
        values = values.sort_values()
        day_start = pd.Timestamp(f"{day} {session_start}", tz=TIMEZONE)
        day_end = pd.Timestamp(f"{day} {session_end}", tz=TIMEZONE)
        partial_current_session = (
            day == as_of_ts.date()
            and day_start < as_of_ts < day_end
        )
        if partial_current_session:
            cutoff = min(day_end, as_of_ts.floor(f"{interval_minutes}min"))
            expected = pd.date_range(day_start, cutoff, freq=f"{interval_minutes}min", inclusive="left")
        else:
            cutoff = day_end
            expected = pd.date_range(day_start, day_end, freq=f"{interval_minutes}min", inclusive="left")

        observed = pd.DatetimeIndex(values)
        in_session = observed[(observed >= day_start) & (observed < day_end)]
        outside = len(observed) - len(in_session)
        if partial_current_session:
            completed_observed = in_session[in_session < cutoff]
            uncompleted = in_session[in_session >= cutoff]
        else:
            completed_observed = in_session
            uncompleted = pd.DatetimeIndex([])
        missing = len(expected.difference(completed_observed))
        misaligned = int(((in_session - day_start) % pd.Timedelta(minutes=interval_minutes) != pd.Timedelta(0)).sum())
        duplicate_count = int(in_session.duplicated().sum())
        expected_total += len(expected)
        missing_total += missing
        misaligned_total += misaligned
        outside_total += outside
        uncompleted_total += len(uncompleted)

        date_issues = []
        if missing:
            date_issues.append(f"missing completed bars: {missing}")
        if outside:
            date_issues.append(f"outside session: {outside}")
        if misaligned:
            date_issues.append(f"misaligned timestamps: {misaligned}")
        if duplicate_count:
            date_issues.append(f"duplicate session timestamps: {duplicate_count}")
        per_date[str(day)] = {
            "expected_bars": len(expected),
            "full_session_expected_bars": full_session_bars,
            "partial_session": bool(partial_current_session),
            "audit_cutoff": str(cutoff),
            "observed_bars_in_session": len(in_session),
            "observed_completed_bars": len(completed_observed),
            "uncompleted_bars_excluded": len(uncompleted),
            "missing_bars": missing,
            "outside_session_bars": outside,
            "misaligned_bars": misaligned,
            "duplicate_bars": duplicate_count,
            "first_observed": str(in_session.min()) if len(in_session) else None,
            "last_observed": str(in_session.max()) if len(in_session) else None,
            "last_completed_observed": str(completed_observed.max()) if len(completed_observed) else None,
            "issues": date_issues,
        }
        if date_issues:
            issues.append(f"{day}: " + "; ".join(date_issues))

    return {
        "eligible": not issues,
        "rows": len(frame),
        "as_of_timestamp": str(as_of_ts),
        "observed_dates": len(per_date),
        "expected_bars_on_observed_dates": expected_total,
        "full_session_expected_bars_per_observed_date": full_session_bars,
        "observed_bars_in_session": sum(v["observed_bars_in_session"] for v in per_date.values()),
        "observed_completed_bars": sum(v["observed_completed_bars"] for v in per_date.values()),
        "uncompleted_bars_excluded": uncompleted_total,
        "missing_bars": missing_total,
        "misaligned_bars": misaligned_total,
        "outside_session_bars": outside_total,
        "issues": issues,
        "per_date": per_date,
        "note": "Audits observed dates only; current forming bars and future slots are excluded; does not infer holidays or missing whole trading sessions.",
    }


def audit_research_directory(
    directory: str | Path,
    interval_minutes: int = 5,
    reference_symbol: str | None = None,
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
                frame, interval_minutes, as_of=as_of_ts,
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
        current_day_start = pd.Timestamp(f"{as_of_ts.date()} 09:15", tz=TIMEZONE)
        current_day_end = pd.Timestamp(f"{as_of_ts.date()} 15:30", tz=TIMEZONE)
        partial_current_session = current_day_start < as_of_ts < current_day_end
        cutoff = as_of_ts.floor(f"{interval_minutes}min") if partial_current_session else current_day_end

        def completed_timestamps(timestamp_set):
            if not partial_current_session:
                return timestamp_set
            return {t for t in timestamp_set if t.date() != as_of_ts.date() or t < cutoff}

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
            "note": "Cross-symbol comparison uses a reference dataset; it is not an exchange calendar.",
        }
    out_path = root / "completeness_audit.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_file"] = str(out_path)

    # Keep the acquisition manifest's simple OHLCV-validity audit separate,
    # but make research eligibility reflect the full completeness/sync audit.
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and symbols:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_results = manifest.get("results", {})
        eligible_count = 0
        for file_symbol, entry in symbols.items():
            symbol = file_symbol
            for suffix in (f"_{interval_minutes}m", "_5m", "_1m"):
                if symbol.endswith(suffix):
                    symbol = symbol[:-len(suffix)]
                    break
            result_entry = manifest_results.get(symbol)
            if result_entry is None:
                continue
            completeness = entry["completeness"]
            sync = synchronization[file_symbol]
            research_eligible = (
                completeness["eligible"]
                and sync["missing_reference_timestamps"] == 0
                and sync["extra_timestamps_vs_reference"] == 0
            )
            if research_eligible:
                eligible_count += 1
            result_entry["research_eligibility"] = {
                "eligible": research_eligible,
                "completeness": {
                    "missing_bars": completeness["missing_bars"],
                    "misaligned_bars": completeness["misaligned_bars"],
                    "outside_session_bars": completeness["outside_session_bars"],
                    "observed_dates": completeness["observed_dates"],
                    "expected_bars": completeness["expected_bars_on_observed_dates"],
                    "observed_completed_bars": completeness["observed_completed_bars"],
                },
                "synchronization": sync,
                "reference_symbol": chosen,
            }
        manifest["completeness_audit"] = {
            "eligible": report["eligible"],
            "eligible_symbols": eligible_count,
            "audited_symbols": len(symbols),
            "report_file": str(out_path),
            "issues": report["issues"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        report["manifest_updated"] = str(manifest_path)
    return report
