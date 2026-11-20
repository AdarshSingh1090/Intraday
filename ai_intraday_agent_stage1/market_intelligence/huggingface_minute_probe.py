"""Probe a public Hugging Face 1-minute dataset for Yahoo 5m gaps.

This source is a public research dataset tagged with Upstox metadata. Its
provenance and filter-index completeness are not assumed; partial responses
cannot be used as recovery candidates. This probe never modifies CSVs.
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
import time

import pandas as pd

TIMEZONE = "Asia/Kolkata"
DATASET_ID = "xxparthparekhxx/indian-stock-market-minute-data"
FILTER_URL = "https://datasets-server.huggingface.co/filter"
OHLCV = ["timestamp", "open", "high", "low", "close", "volume"]


def _read_dataset(path):
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].dt.tz is None:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(TIMEZONE)
    else:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert(TIMEZONE)
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def expected_5m_grid(day, session_start="09:15", session_end="15:15"):
    start = pd.Timestamp(f"{day} {session_start}", tz=TIMEZONE)
    end = pd.Timestamp(f"{day} {session_end}", tz=TIMEZONE)
    return pd.date_range(start, end, freq="5min", inclusive="left")


def aggregate_exact_5m_from_minute(frame, stamp):
    """Aggregate exactly five 1-minute bars with timestamps at the 5m start."""
    if frame is None or frame.empty:
        return None
    indexed = frame.set_index("timestamp")
    expected = pd.date_range(stamp, periods=5, freq="1min")
    if not expected.isin(indexed.index).all():
        return None
    rows = indexed.loc[expected]
    if rows.index.duplicated().any() or rows[OHLCV[1:]].isna().any().any():
        return None
    return {
        "open": float(rows["open"].iloc[0]),
        "high": float(rows["high"].max()),
        "low": float(rows["low"].min()),
        "close": float(rows["close"].iloc[-1]),
        "volume": float(rows["volume"].sum()),
    }


def _compare_aggregated_to_yahoo(minute_frame, original_frame, day, start, end):
    existing = original_frame.set_index("timestamp")
    compared = matches = mismatches = 0
    first_mismatch = None
    for stamp in expected_5m_grid(day):
        if stamp < start or stamp >= end or stamp not in existing.index:
            continue
        candidate = aggregate_exact_5m_from_minute(minute_frame, stamp)
        if candidate is None:
            continue
        row = existing.loc[stamp]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        compared += 1
        equal = all(
            abs(candidate[k] - float(row[k])) <= (1e-9 if k == "volume" else 1e-4)
            for k in OHLCV[1:]
        )
        if equal:
            matches += 1
        else:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {
                    "timestamp": str(stamp),
                    "dataset_1m_aggregate": candidate,
                    "yahoo_5m": {k: float(row[k]) for k in OHLCV[1:]},
                }
    return {
        "status": "COMPARED" if compared else "NOT_COMPARABLE",
        "compared_existing_bars": compared,
        "matches": matches,
        "mismatches": mismatches,
        "first_mismatch": first_mismatch,
        "tolerance": {"prices": 1e-4, "volume": 1e-9},
    }


def fetch_public_minute_window(session, symbol, start, end, timeout=45):
    """Query the dataset viewer filter API for a narrow symbol/time window."""
    start_utc = pd.Timestamp(start).tz_convert("UTC")
    end_utc = pd.Timestamp(end).tz_convert("UTC")
    where = (
        f'"symbol"=\'{symbol}\' AND '
        f'"timestamp">=\'{start_utc.isoformat()}\' AND '
        f'"timestamp"<\'{end_utc.isoformat()}\''
    )
    params = {
        "dataset": DATASET_ID,
        "config": "default",
        "split": "minute",
        "where": where,
        "offset": 0,
        "length": 100,
    }
    response = session.get(FILTER_URL, params=params, headers={"Accept": "application/json"}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    rows = []
    for item in payload.get("rows", []):
        row = item.get("row", item)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=OHLCV)
    else:
        if "timestamp" not in frame:
            raise ValueError("Hugging Face filter response lacks timestamp")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dt.tz_convert(TIMEZONE)
        for col in OHLCV[1:]:
            if col not in frame:
                raise ValueError(f"Hugging Face filter response lacks {col}")
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return {
        "frame": frame,
        "partial": payload.get("partial"),
        "num_rows_total": payload.get("num_rows_total"),
        "num_rows_returned": len(frame),
        "dataset": payload.get("dataset", DATASET_ID),
        "config": payload.get("config", "default"),
        "split": payload.get("split", "minute"),
    }


def probe_public_minute_dataset(directory, symbols, output, session_start="09:15", session_end="15:15", session=None):
    """Investigate Yahoo gaps using public minute data; no repair or mutation."""
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
                grid = expected_5m_grid(day, session_start, session_end)
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
                start = min(missing) - pd.Timedelta(minutes=5)
                end = max(missing) + pd.Timedelta(minutes=10)
                request_report = {"status": "NOT_RUN"}
                try:
                    result = fetch_public_minute_window(
                        session, symbol.upper().removesuffix(".NS"), start, end
                    )
                    frame = result["frame"]
                    partial = result["partial"]
                    status = "OK" if not frame.empty else "EMPTY"
                    raw_candidates = {
                        str(stamp): aggregate_exact_5m_from_minute(frame, stamp)
                        for stamp in missing
                    }
                    raw_candidate_count = sum(v is not None for v in raw_candidates.values())
                    if partial is False:
                        candidates = raw_candidates
                        candidate_status = "CANDIDATES_AVAILABLE" if raw_candidate_count else "NO_COMPLETE_WINDOW"
                    else:
                        candidates = {stamp: None for stamp in raw_candidates}
                        if raw_candidate_count:
                            candidate_status = (
                                "WITHHELD_PARTIAL_INDEX" if partial is True
                                else "WITHHELD_INDEX_COMPLETENESS_UNKNOWN"
                            )
                        else:
                            candidate_status = "NO_COMPLETE_WINDOW"
                    neighbor_validation = _compare_aggregated_to_yahoo(
                        frame, original, day, start, end
                    )
                    request_report = {
                        "status": status,
                        "partial_index": partial,
                        "num_rows_total": result["num_rows_total"],
                        "num_rows_returned": result["num_rows_returned"],
                        "window_start_ist": str(start),
                        "window_end_exclusive_ist": str(end),
                        "raw_complete_candidate_count": raw_candidate_count,
                        "candidate_status": candidate_status,
                        "reconstructed_5m_candidates": candidates,
                        "neighbor_validation_against_yahoo_5m": neighbor_validation,
                        "dataset": result["dataset"],
                        "config": result["config"],
                        "split": result["split"],
                    }
                except Exception as exc:
                    request_report = {
                        "status": "ERROR",
                        "error": f"{type(exc).__name__}: {exc}",
                        "reconstructed_5m_candidates": {str(stamp): None for stamp in missing},
                    }
                day_reports[day_text] = {
                    "missing_timestamps_in_original": [str(stamp) for stamp in missing],
                    "public_dataset_query": request_report,
                }
                time.sleep(0.15)

            candidate_total = sum(
                value is not None
                for day in day_reports.values()
                for value in day["public_dataset_query"].get("reconstructed_5m_candidates", {}).values()
            )
            successful = sum(day["public_dataset_query"].get("status") == "OK" for day in day_reports.values())
            reports[symbol] = {
                "status": "CANDIDATES_FOUND" if candidate_total else ("PROBED" if successful else "INCONCLUSIVE"),
                "missing_count_original": sum(len(v) for v in missing_by_day.values()),
                "complete_5m_candidates_from_public_1m_dataset": candidate_total,
                "successful_queries": successful,
                "missing_by_day": day_reports,
            }
        except Exception as exc:
            reports[symbol] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

    total_missing = sum(r.get("missing_count_original", 0) for r in reports.values())
    total_candidates = sum(r.get("complete_5m_candidates_from_public_1m_dataset", 0) for r in reports.values())
    total_successful = sum(r.get("successful_queries", 0) for r in reports.values())
    result = {
        "purpose": "free public dataset probe for exact missing 5m candidates from 1m candles",
        "provider": "Hugging Face public dataset viewer: " + DATASET_ID,
        "authentication_required": False,
        "cost": "free public access",
        "provenance_warning": "Dataset metadata is tagged upstox; source provenance and completeness have not been independently verified. Filter index may be partial for datasets over 5GB.",
        "independent_provider_comparison": True,
        "data_repaired": False,
        "datasets_modified": False,
        "interpretation": (
            "Exact 5m candidates require five consecutive 1m bars and a complete (partial=false) dataset-viewer index. "
            "Candidates are only reported; no CSVs are modified. Neighbor bars are compared to Yahoo when possible."
        ),
        "summary": {
            "symbols_with_gaps": sum(r.get("missing_count_original", 0) > 0 for r in reports.values()),
            "missing_5m_timestamps_probed": total_missing,
            "exact_5m_candidates_found": total_candidates,
            "successful_public_queries": total_successful,
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
    parser.add_argument("--output", default="reports/huggingface_minute_probe.json")
    args = parser.parse_args()
    result = probe_public_minute_dataset(
        args.directory, [s.strip() for s in args.symbols.split(",") if s.strip()],
        args.output, args.session_start, args.session_end,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
