"""Intraday dataset completeness and cross-symbol synchronization audit.

This is a conservative audit: it detects gaps within observed sessions and
cross-symbol timestamp mismatches, but does not infer exchange holidays.
"""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

OHLCV = ["timestamp", "open", "high", "low", "close", "volume"]


def _load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path.name}: missing timestamp column")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame


def audit_intraday_completeness(frame: pd.DataFrame, interval_minutes: int = 5,
                                session_start: str = "09:15",
                                session_end: str = "15:30") -> dict:
    """Audit gaps on observed dates, session boundaries, and bar alignment.

    Session end is exclusive. NSE regular session is 09:15–15:30, so a
    5-minute full session contains 75 bars. Only dates present in the data are
    audited; holidays and absent whole sessions cannot be inferred.
    """
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    if frame is None or frame.empty:
        return {"eligible": False, "rows": 0, "observed_dates": 0,
                "expected_bars_on_observed_dates": 0, "missing_bars": 0,
                "issues": ["empty dataset"], "per_date": {}}
    if "timestamp" not in frame.columns:
        return {"eligible": False, "rows": len(frame), "observed_dates": 0,
                "expected_bars_on_observed_dates": 0, "missing_bars": 0,
                "issues": ["missing timestamp column"], "per_date": {}}

    ts = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return {"eligible": False, "rows": len(frame), "observed_dates": 0,
                "expected_bars_on_observed_dates": 0, "missing_bars": 0,
                "issues": ["no valid timestamps"], "per_date": {}}
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Kolkata")
    else:
        # Localize naive timestamps to exchange timezone.
        ts = ts.dt.tz_localize("Asia/Kolkata")

    start_t = pd.Timestamp(session_start).time()
    end_t = pd.Timestamp(session_end).time()
    per_date = {}
    issues = []
    expected_total = 0
    missing_total = 0
    misaligned_total = 0
    outside_total = 0

    for day, values in ts.groupby(ts.dt.date):
        values = values.sort_values()
        day_start = pd.Timestamp(f"{day} {session_start}", tz="Asia/Kolkata")
        day_end = pd.Timestamp(f"{day} {session_end}", tz="Asia/Kolkata")
        expected = pd.date_range(day_start, day_end, freq=f"{interval_minutes}min", inclusive="left")
        observed = pd.DatetimeIndex(values)
        in_session = observed[(observed >= day_start) & (observed < day_end)]
        outside = len(observed) - len(in_session)
        missing = len(expected.difference(in_session))
        misaligned = int(((in_session - day_start) % pd.Timedelta(minutes=interval_minutes) != pd.Timedelta(0)).sum())
        duplicate_count = int(in_session.duplicated().sum())
        expected_total += len(expected)
        missing_total += missing
        misaligned_total += misaligned
        outside_total += outside
        date_issues = []
        if missing:
            date_issues.append(f"missing bars: {missing}")
        if outside:
            date_issues.append(f"outside session: {outside}")
        if misaligned:
            date_issues.append(f"misaligned timestamps: {misaligned}")
        if duplicate_count:
            date_issues.append(f"duplicate session timestamps: {duplicate_count}")
        per_date[str(day)] = {
            "expected_bars": len(expected),
            "observed_bars_in_session": len(in_session),
            "missing_bars": missing,
            "outside_session_bars": outside,
            "misaligned_bars": misaligned,
            "duplicate_bars": duplicate_count,
            "first_observed": str(in_session.min()) if len(in_session) else None,
            "last_observed": str(in_session.max()) if len(in_session) else None,
            "issues": date_issues,
        }
        if date_issues:
            issues.append(f"{day}: " + "; ".join(date_issues))

    return {
        "eligible": not issues,
        "rows": len(frame),
        "observed_dates": len(per_date),
        "expected_bars_on_observed_dates": expected_total,
        "observed_bars_in_session": sum(v["observed_bars_in_session"] for v in per_date.values()),
        "missing_bars": missing_total,
        "misaligned_bars": misaligned_total,
        "outside_session_bars": outside_total,
        "issues": issues,
        "per_date": per_date,
        "note": "Audits observed dates only; does not infer holidays or missing whole trading sessions.",
    }


def audit_research_directory(directory: str | Path, interval_minutes: int = 5,
                             reference_symbol: str | None = None) -> dict:
    """Audit each CSV and timestamp synchronization across symbols."""
    root = Path(directory)
    files = sorted(root.glob("*.csv"))
    symbols = {}
    for path in files:
        frame = _load_csv(path)
        symbols[path.stem] = {
            "rows": len(frame),
            "completeness": audit_intraday_completeness(frame, interval_minutes),
            "timestamps": set(frame["timestamp"].dropna()),
        }

    if not symbols:
        report = {"eligible": False, "symbols": {}, "issues": ["no CSV datasets found"]}
    else:
        chosen = reference_symbol
        if chosen is not None and chosen not in symbols:
            raise ValueError(f"Reference symbol/file not found: {chosen}")
        if chosen is None:
            chosen = max(symbols, key=lambda s: symbols[s]["rows"])
        reference = symbols[chosen]["timestamps"]
        synchronization = {}
        for symbol, entry in symbols.items():
            ts = entry["timestamps"]
            synchronization[symbol] = {
                "reference": chosen,
                "missing_reference_timestamps": len(reference - ts),
                "extra_timestamps_vs_reference": len(ts - reference),
                "matched_timestamps": len(reference & ts),
            }
        report = {
            "eligible": all(e["completeness"]["eligible"] for e in symbols.values())
                        and all(v["missing_reference_timestamps"] == 0
                                and v["extra_timestamps_vs_reference"] == 0
                                for v in synchronization.values()),
            "symbols": {s: {k: v for k, v in e.items() if k != "timestamps"}
                        for s, e in symbols.items()},
            "reference_symbol": chosen,
            "synchronization": synchronization,
            "issues": [],
            "note": "Cross-symbol comparison uses a reference dataset; it is not an exchange calendar.",
        }
    out_path = root / "completeness_audit.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_file"] = str(out_path)
    return report
