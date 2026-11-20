"""Diagnostic probe for alternate Yahoo intraday intervals.

Only the 1-minute interval can be aggregated into exact 5-minute candidates,
and only when all five minute candles are present. This is same-provider
investigation; it never repairs or modifies source CSV files.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

TIMEZONE = "Asia/Kolkata"
ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
INTERVALS = ("1m", "2m", "15m", "30m")
OHLCV = ["timestamp", "open", "high", "low", "close", "volume"]


def expected_grid(day, interval_minutes=5, session_start="09:15", session_end="15:15"):
    tz = ZoneInfo(TIMEZONE)
    start = datetime.combine(day, time.fromisoformat(session_start), tzinfo=tz)
    end = datetime.combine(day, time.fromisoformat(session_end), tzinfo=tz)
    return pd.date_range(start=start, end=end, freq=f"{interval_minutes}min", inclusive="left")


def fetch_chart_day(session, symbol, day, interval, endpoint=ENDPOINT, timeout=30):
    """Fetch a single local day at the requested Yahoo Chart API interval."""
    tz = ZoneInfo(TIMEZONE)
    start = datetime.combine(day, time(0, 0), tzinfo=tz)
    end = start + timedelta(days=1)
    params = {
        "period1": int(start.timestamp()),
        "period2": int(end.timestamp()),
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits",
    }
    response = session.get(
        endpoint.format(symbol=symbol),
        params=params,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        return pd.DataFrame(columns=OHLCV)
    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    rows = []
    for i, stamp in enumerate(timestamps):
        values = []
        for col in OHLCV[1:]:
            array = quotes.get(col) or []
            values.append(array[i] if i < len(array) else None)
        if any(v is None for v in values[:4]):
            continue
        rows.append([pd.Timestamp(stamp, unit="s", tz="UTC").tz_convert(TIMEZONE), *values])
    return pd.DataFrame(rows, columns=OHLCV)


def _read_dataset(path):
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].dt.tz is None:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(TIMEZONE)
    else:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert(TIMEZONE)
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def aggregate_exact_5m_from_1m(frame, stamp):
    """Aggregate exactly the five expected 1m bars for a 5m start timestamp."""
    if frame is None or frame.empty:
        return None
    minute_index = frame.set_index("timestamp")
    expected = pd.date_range(stamp, periods=5, freq="1min")
    if not expected.isin(minute_index.index).all():
        return None
    rows = minute_index.loc[expected]
    if rows.index.duplicated().any() or rows[OHLCV[1:]].isna().any().any():
        return None
    return {
        "open": float(rows["open"].iloc[0]),
        "high": float(rows["high"].max()),
        "low": float(rows["low"].min()),
        "close": float(rows["close"].iloc[-1]),
        "volume": float(rows["volume"].sum()),
    }


def compare_aggregated_1m_with_existing_5m(minute_frame, original_frame, session_start="09:15", session_end="15:15"):
    """Report exact OHLCV agreement on bars that can be built from five 1m rows."""
    if minute_frame is None or minute_frame.empty or original_frame is None or original_frame.empty:
        return {"status": "NOT_COMPARABLE", "compared_bars": 0, "matches": 0, "mismatches": 0}
    start_t = pd.to_datetime(session_start).time()
    end_t = pd.to_datetime(session_end).time()
    existing = original_frame.set_index("timestamp")
    minute_times = minute_frame["timestamp"]
    dates = sorted(set(minute_times.dt.date))
    compared = matches = mismatches = 0
    first_mismatch = None
    for day in dates:
        for stamp in expected_grid(day, session_start=session_start, session_end=session_end):
            if stamp not in existing.index:
                continue
            candidate = aggregate_exact_5m_from_1m(minute_frame, stamp)
            if candidate is None:
                continue
            compared += 1
            row = existing.loc[stamp]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            equal = all(
                abs(float(candidate[k]) - float(row[k])) <= (1e-9 if k == "volume" else 1e-6)
                for k in OHLCV[1:]
            )
            if equal:
                matches += 1
            else:
                mismatches += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "timestamp": str(stamp),
                        "aggregated_1m": candidate,
                        "original_5m": {k: float(row[k]) for k in OHLCV[1:]},
                    }
    return {
        "status": "COMPARED" if compared else "NOT_COMPARABLE",
        "compared_bars": compared,
        "matches": matches,
        "mismatches": mismatches,
        "first_mismatch": first_mismatch,
        "comparison_scope": "same Yahoo provider; exact OHLCV equality with tolerances; diagnostic only",
    }


def probe_alternate_yahoo_intervals(directory, symbols, output, session_start="09:15", session_end="15:15", session=None):
    """Probe missing 5m bars using other Yahoo intervals without modifying CSVs."""
    import requests

    session = session or requests.Session()
    directory = Path(directory)
    reports = {}
    for symbol in symbols:
        path = directory / f"{symbol}_5m.csv"
        if not path.exists():
            reports[symbol] = {"status": "MISSING_DATASET", "error": str(path)}
            continue
        try:
            original = _read_dataset(path)
            missing_by_day = {}
            for day in sorted(set(original.timestamp.dt.date)):
                grid = expected_grid(day, session_start=session_start, session_end=session_end)
                observed = set(original.loc[original.timestamp.dt.date == day, "timestamp"])
                missing = [stamp for stamp in grid if stamp not in observed]
                if missing:
                    missing_by_day[str(day)] = missing
            if not missing_by_day:
                reports[symbol] = {"status": "NO_GAPS", "missing_count_original": 0, "days": {}}
                continue

            day_reports = {}
            for day_text, missing in missing_by_day.items():
                day = pd.Timestamp(day_text).date()
                interval_reports = {}
                for interval in INTERVALS:
                    try:
                        data = fetch_chart_day(session, symbol, day, interval)
                        status = "OK" if not data.empty else "EMPTY"
                        near = {}
                        for stamp in missing:
                            left = stamp - pd.Timedelta(minutes=10)
                            right = stamp + pd.Timedelta(minutes=10)
                            nearby = data.loc[(data.timestamp >= left) & (data.timestamp < right), "timestamp"]
                            near[str(stamp)] = [str(t) for t in nearby]
                        interval_reports[interval] = {
                            "status": status,
                            "rows_returned": int(len(data)),
                            "timestamps_near_missing_bars": near,
                        }
                        if interval == "1m":
                            interval_reports[interval]["reconstructed_5m_candidates"] = {
                                str(stamp): aggregate_exact_5m_from_1m(data, stamp)
                                for stamp in missing
                            }
                            interval_reports[interval]["cross_interval_validation"] = compare_aggregated_1m_with_existing_5m(
                                data, original, session_start, session_end
                            )
                    except Exception as exc:
                        interval_reports[interval] = {
                            "status": "ERROR",
                            "error": f"{type(exc).__name__}: {exc}",
                            "timestamps_near_missing_bars": {str(stamp): [] for stamp in missing},
                        }
                found_1m = sum(
                    candidate is not None
                    for candidate in interval_reports.get("1m", {}).get("reconstructed_5m_candidates", {}).values()
                )
                day_reports[day_text] = {
                    "missing_timestamps_in_original": [str(stamp) for stamp in missing],
                    "intervals": interval_reports,
                    "complete_5m_candidates_from_1m": found_1m,
                }

            candidate_total = sum(day.get("complete_5m_candidates_from_1m", 0) for day in day_reports.values())
            returned_intervals = sum(
                report.get("status") == "OK"
                for day in day_reports.values()
                for report in day.get("intervals", {}).values()
            )
            reports[symbol] = {
                "status": "CANDIDATES_FOUND" if candidate_total else ("PROBED" if returned_intervals else "INCONCLUSIVE"),
                "missing_count_original": sum(len(v) for v in missing_by_day.values()),
                "complete_5m_candidates_from_1m": candidate_total,
                "successful_interval_requests": returned_intervals,
                "missing_by_day": day_reports,
            }
        except Exception as exc:
            reports[symbol] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

    total_missing = sum(r.get("missing_count_original", 0) for r in reports.values())
    total_candidates = sum(r.get("complete_5m_candidates_from_1m", 0) for r in reports.values())
    total_ok_requests = sum(r.get("successful_interval_requests", 0) for r in reports.values())
    result = {
        "purpose": "test whether alternate Yahoo intraday intervals can expose or reconstruct missing 5m bars",
        "provider": "Yahoo Finance Chart API (query1)",
        "intervals_tested": list(INTERVALS),
        "independent_provider_comparison": False,
        "data_repaired": False,
        "datasets_modified": False,
        "interpretation": (
            "Only a complete set of five 1m bars can produce an exact 5m candidate. "
            "2m/15m/30m data are diagnostic and cannot reconstruct exact 5m OHLCV. "
            "All intervals are the same Yahoo source; no candidate is inserted into the original CSV."
        ),
        "summary": {
            "symbols_with_gaps": sum(r.get("missing_count_original", 0) > 0 for r in reports.values()),
            "missing_5m_timestamps_probed": total_missing,
            "exact_5m_candidates_from_1m": total_candidates,
            "successful_interval_requests": total_ok_requests,
        },
        "session": {"start": session_start, "end_exclusive": session_end},
        "reports": reports,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="data/research")
    parser.add_argument("--symbols", default="RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS,SBIN.NS,AXISBANK.NS,KOTAKBANK.NS,BHARTIARTL.NS,ITC.NS")
    parser.add_argument("--session-start", default="09:15")
    parser.add_argument("--session-end", default="15:15")
    parser.add_argument("--output", default="reports/yahoo_alternate_interval_probe.json")
    args = parser.parse_args()
    result = probe_alternate_yahoo_intervals(
        args.directory, [s.strip() for s in args.symbols.split(",") if s.strip()],
        args.output, args.session_start, args.session_end,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
