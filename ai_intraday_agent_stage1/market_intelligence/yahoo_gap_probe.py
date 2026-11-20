"""Probe missing Yahoo candles via alternate direct Chart API requests.

This is a same-provider recovery investigation, not independent validation.
It does not modify or repair downloaded datasets.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

TIMEZONE = "Asia/Kolkata"
CHART_ENDPOINTS = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
    "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
)
OHLCV = ["timestamp", "open", "high", "low", "close", "volume"]


def expected_grid(day, interval_minutes=5, session_start="09:15", session_end="15:15"):
    tz = ZoneInfo(TIMEZONE)
    start = datetime.combine(day, time.fromisoformat(session_start), tzinfo=tz)
    end = datetime.combine(day, time.fromisoformat(session_end), tzinfo=tz)
    return pd.date_range(start=start, end=end, freq=f"{interval_minutes}min", inclusive="left")


def fetch_chart_day(session, symbol, day, endpoint, timeout=30):
    """Fetch one local calendar day using Yahoo's direct chart endpoint."""
    tz = ZoneInfo(TIMEZONE)
    start = datetime.combine(day, time(0, 0), tzinfo=tz)
    end = start + timedelta(days=1)
    params = {
        "period1": int(start.timestamp()),
        "period2": int(end.timestamp()),
        "interval": "5m",
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


def probe_missing_yahoo_bars(directory, symbols, output, session_start="09:15", session_end="15:15", session=None):
    """Probe only dates with missing expected bars; preserve original CSVs."""
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
                grid = expected_grid(day, session_end=session_end, session_start=session_start)
                observed = set(original.loc[original.timestamp.dt.date == day, "timestamp"])
                missing = [stamp for stamp in grid if stamp not in observed]
                if missing:
                    missing_by_day[str(day)] = missing

            day_reports = {}
            for day_text, missing in missing_by_day.items():
                day = pd.Timestamp(day_text).date()
                endpoint_reports = {}
                for endpoint in CHART_ENDPOINTS:
                    host = endpoint.split("/")[2]
                    try:
                        fetched = fetch_chart_day(session, symbol, day, endpoint)
                        fetched = fetched.set_index("timestamp") if not fetched.empty else pd.DataFrame()
                        candles = {}
                        for stamp in missing:
                            if stamp in fetched.index:
                                row = fetched.loc[stamp]
                                if isinstance(row, pd.DataFrame):
                                    row = row.iloc[0]
                                candles[str(stamp)] = {k: float(row[k]) for k in OHLCV[1:]}
                            else:
                                candles[str(stamp)] = None
                        endpoint_reports[host] = {
                            "status": "OK",
                            "rows_returned": int(len(fetched)),
                            "requested_missing_timestamps": candles,
                        }
                    except Exception as exc:
                        endpoint_reports[host] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

                comparison = {}
                hosts = list(endpoint_reports)
                if len(hosts) == 2:
                    a, b = (endpoint_reports[h] for h in hosts)
                    if a.get("status") == b.get("status") == "OK":
                        for stamp in map(str, missing):
                            ca, cb = a["requested_missing_timestamps"][stamp], b["requested_missing_timestamps"][stamp]
                            comparison[stamp] = {
                                "both_found": ca is not None and cb is not None,
                                "values_agree": ca == cb if ca is not None and cb is not None else None,
                            }
                day_reports[day_text] = {
                    "missing_timestamps_in_original": [str(x) for x in missing],
                    "endpoints": endpoint_reports,
                    "endpoint_comparison": comparison,
                }

            found_by_endpoint = {}
            for host in (e.split("/")[2] for e in CHART_ENDPOINTS):
                found_by_endpoint[host] = sum(
                    value is not None
                    for day_report in day_reports.values()
                    for value in day_report.get("endpoints", {}).get(host, {}).get("requested_missing_timestamps", {}).values()
                )
            reports[symbol] = {
                "status": "GAPS_PROBED" if missing_by_day else "NO_GAPS",
                "missing_count_original": sum(len(v) for v in missing_by_day.values()),
                "missing_by_day": day_reports,
                "missing_candles_found_by_endpoint": found_by_endpoint,
            }
        except Exception as exc:
            reports[symbol] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

    result = {
        "purpose": "investigate missing Yahoo candles using alternate direct Chart API endpoints and smaller per-day requests",
        "provider": "Yahoo Finance Chart API (query1/query2)",
        "independent_provider_comparison": False,
        "data_repaired": False,
        "datasets_modified": False,
        "interpretation": "Recovery candidates are reported only. Both endpoints are Yahoo and are not independent validation. No candles are inserted or fabricated.",
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
    parser.add_argument("--output", default="reports/yahoo_gap_probe.json")
    args = parser.parse_args()
    result = probe_missing_yahoo_bars(
        args.directory, [s.strip() for s in args.symbols.split(",") if s.strip()],
        args.output, args.session_start, args.session_end,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
