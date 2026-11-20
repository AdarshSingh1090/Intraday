"""Intraday dataset completeness and cross-symbol synchronization audit."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

OHLCV = ["timestamp", "open", "high", "low", "close", "volume"]
MARKET_TZ = "Asia/Kolkata"


def _load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path.name}: missing timestamp column")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame


def _normalize_as_of(as_of=None) -> pd.Timestamp:
    value = pd.Timestamp.now(tz=MARKET_TZ) if as_of is None else pd.Timestamp(as_of)
    return value.tz_localize(MARKET_TZ) if value.tzinfo is None else value.tz_convert(MARKET_TZ)


def audit_intraday_completeness(frame: pd.DataFrame, interval_minutes: int = 5,
                                session_start: str = "09:15",
                                session_end: str = "15:30",
                                as_of=None) -> dict:
    """Audit gaps; current date expects only bars completed by audit time.

    Bar timestamps denote the bar start. On the current exchange-local date,
    expected timestamps stop at the current interval floor (exclusive), so an
    unfinished bar and the remainder of the session are not reported missing.
    Historical observed dates still expect a full regular session.
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

    now = _normalize_as_of(as_of)
    ts = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return {"eligible": False, "rows": len(frame), "observed_dates": 0,
                "expected_bars_on_observed_dates": 0, "missing_bars": 0,
                "issues": ["no valid timestamps"], "per_date": {}}
    ts = ts.dt.tz_convert(MARKET_TZ) if ts.dt.tz is not None else ts.dt.tz_localize(MARKET_TZ)

    end_time = pd.Timestamp(session_end).time()
    per_date, issues = {}, []
    expected_total = missing_total = misaligned_total = outside_total = 0

    for day, values in ts.groupby(ts.dt.date):
        values = values.sort_values()
        day_start = pd.Timestamp(f"{day} {session_start}", tz=MARKET_TZ)
        day_end = pd.Timestamp(f"{day} {session_end}", tz=MARKET_TZ)
        partial = day == now.date() and now.time() < end_time
        expected_end = min(day_end, now.floor(f"{interval_minutes}min")) if partial else day_end
        expected = pd.date_range(day_start, expected_end, freq=f"{interval_minutes}min", inclusive="left")
        observed = pd.DatetimeIndex(values)
        in_session = observed[(observed >= day_start) & (observed < day_end)]
        audit_observed = in_session[in_session < expected_end] if partial else in_session
        outside = len(observed) - len(in_session)
        missing = len(expected.difference(audit_observed))
        misaligned = int(((in_session - day_start) % pd.Timedelta(minutes=interval_minutes) != pd.Timedelta(0)).sum())
        duplicates = int(audit_observed.duplicated().sum())
        expected_total += len(expected); missing_total += missing
        misaligned_total += misaligned; outside_total += outside
        date_issues = []
        if missing: date_issues.append(f"missing completed bars: {missing}")
        if outside: date_issues.append(f"outside session: {outside}")
        if misaligned: date_issues.append(f"misaligned timestamps: {misaligned}")
        if duplicates: date_issues.append(f"duplicate session timestamps: {duplicates}")
        per_date[str(day)] = {
            "expected_bars": len(expected), "observed_bars_in_session": len(in_session),
            "observed_bars_in_audit_window": len(audit_observed), "missing_bars": missing,
            "outside_session_bars": outside, "misaligned_bars": misaligned,
            "duplicate_bars": duplicates, "partial_session": partial,
            "audit_cutoff_exclusive": str(expected_end),
            "first_observed": str(in_session.min()) if len(in_session) else None,
            "last_observed": str(in_session.max()) if len(in_session) else None,
            "issues": date_issues,
        }
        if date_issues: issues.append(f"{day}: " + "; ".join(date_issues))

    return {
        "eligible": not issues, "rows": len(frame), "observed_dates": len(per_date),
        "expected_bars_on_observed_dates": expected_total,
        "observed_bars_in_session": sum(v["observed_bars_in_session"] for v in per_date.values()),
        "missing_bars": missing_total, "misaligned_bars": misaligned_total,
        "outside_session_bars": outside_total, "as_of": str(now), "issues": issues,
        "per_date": per_date,
        "note": "Audits observed dates only; current session is limited to completed bars as of audit time. Holidays and missing whole sessions are not inferred.",
    }


def audit_research_directory(directory: str | Path, interval_minutes: int = 5,
                             reference_symbol: str | None = None, as_of=None) -> dict:
    """Audit each CSV and timestamp synchronization across symbols."""
    root = Path(directory)
    files = sorted(root.glob("*.csv"))
    symbols = {}
    for path in files:
        frame = _load_csv(path)
        symbols[path.stem] = {
            "rows": len(frame),
            "completeness": audit_intraday_completeness(frame, interval_minutes, as_of=as_of),
            "timestamps": set(frame["timestamp"].dropna()),
        }
    if not symbols:
        report = {"eligible": False, "symbols": {}, "issues": ["no CSV datasets found"]}
    else:
        chosen = reference_symbol
        if chosen is not None and chosen not in symbols:
            raise ValueError(f"Reference symbol/file not found: {chosen}")
        if chosen is None: chosen = max(symbols, key=lambda s: symbols[s]["rows"])
        reference = symbols[chosen]["timestamps"]
        sync = {}
        for symbol, entry in symbols.items():
            current = entry["timestamps"]
            sync[symbol] = {"reference": chosen,
                "missing_reference_timestamps": len(reference-current),
                "extra_timestamps_vs_reference": len(current-reference),
                "matched_timestamps": len(reference & current)}
        report = {
            "eligible": all(e["completeness"]["eligible"] for e in symbols.values())
                and all(v["missing_reference_timestamps"] == 0 and v["extra_timestamps_vs_reference"] == 0 for v in sync.values()),
            "symbols": {s: {k:v for k,v in e.items() if k != "timestamps"} for s,e in symbols.items()},
            "reference_symbol": chosen, "synchronization": sync, "issues": [],
            "as_of": str(_normalize_as_of(as_of)),
            "note": "Cross-symbol comparison uses a reference dataset; it is not an exchange calendar.",
        }
    out_path = root / "completeness_audit.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_file"] = str(out_path)
    return report
