"""Bounded Parquet row-group column-range probe for one target symbol.

Only selected column chunks from conservative symbol candidate row-groups are
fetched with HTTP Range. It is a manual-review diagnostic; never repairs CSVs.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .huggingface_parquet_range_probe import _range_request

MAX_ROW_DATA_BYTES = 16 * 1024 * 1024
MAX_COLUMN_CHUNK_BYTES = 8 * 1024 * 1024
MAX_FOOTER_BYTES = 1024 * 1024
SELECTED_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
TARGET_BUCKETS_UTC = [
    "2026-10-15T06:15:00+00:00",
    "2026-10-15T06:20:00+00:00",
]
NEIGHBOR_BUCKETS_UTC = [
    "2026-10-15T06:10:00+00:00",
    "2026-10-15T06:25:00+00:00",
]


def _normalize_symbol(value):
    return str(value or "").upper().strip().removesuffix(".NS")


def _select_candidate_groups(report, target_symbol):
    target = _normalize_symbol(target_symbol)
    selected = []
    for file_item in report.get("files", []):
        parsed = file_item.get("metadata_parse") or {}
        for group in parsed.get("candidate_column_chunk_plan", []):
            possible = {_normalize_symbol(x) for x in group.get("symbols_possible", [])}
            if target in possible:
                selected.append({
                    "filename": file_item.get("filename"),
                    "url": file_item.get("url"),
                    "file_size": file_item.get("size"),
                    "footer_length": file_item.get("footer_length_bytes"),
                    "row_group_index": group.get("row_group_index"),
                    "symbols_possible": group.get("symbols_possible", []),
                    "columns": group.get("columns", []),
                    "compressed_bytes_total": group.get("compressed_bytes_total", 0),
                    "all_ranges_valid": group.get("all_ranges_valid", False),
                })
    return selected


def _check_candidate_budget(groups, max_total_bytes=MAX_ROW_DATA_BYTES,
                           max_chunk_bytes=MAX_COLUMN_CHUNK_BYTES):
    total = sum(int(group.get("compressed_bytes_total", 0)) for group in groups)
    chunks = [chunk for group in groups for chunk in group.get("columns", [])]
    if not groups:
        status = "NO_CANDIDATE_GROUPS"
    elif any(not group.get("all_ranges_valid") for group in groups):
        status = "INVALID_COLUMN_RANGES"
    elif any(not chunk.get("valid") for chunk in chunks):
        status = "INVALID_COLUMN_RANGES"
    elif any(int(chunk.get("compressed_size", 0)) > max_chunk_bytes for chunk in chunks):
        status = "SINGLE_CHUNK_CAP_EXCEEDED"
    elif total > max_total_bytes:
        status = "TOTAL_ROW_DATA_CAP_EXCEEDED"
    else:
        status = "WITHIN_BOUNDED_ROW_DATA_CAP"
    return {"status": status, "candidate_group_count": len(groups),
            "candidate_chunk_count": len(chunks), "compressed_bytes_total": total,
            "max_total_row_data_bytes": max_total_bytes,
            "max_single_column_chunk_bytes": max_chunk_bytes}


def _target_window_rows(frame, target_symbol, window_start, window_end):
    if frame.empty or "symbol" not in frame or "timestamp" not in frame:
        return frame.iloc[0:0].copy()
    target = _normalize_symbol(target_symbol)
    frame = frame.copy()
    frame["symbol_norm"] = frame["symbol"].map(_normalize_symbol)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame[(frame["symbol_norm"] == target) & frame["timestamp"].notna()]
    start = pd.Timestamp(window_start)
    end = pd.Timestamp(window_end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    return frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]


def _aggregate_minute_rows(frame, target_symbol, window_start, window_end):
    frame = _target_window_rows(frame, target_symbol, window_start, window_end)
    if frame.empty:
        return {}
    frame["bucket"] = frame["timestamp"].dt.floor("5min")
    result = {}
    for bucket, group in frame.groupby("bucket"):
        group = group.sort_values("timestamp")
        observed = list(group["timestamp"])
        expected = list(pd.date_range(bucket, periods=5, freq="1min"))
        complete = len(group) == 5 and observed == expected
        result[bucket.isoformat()] = {
            "rows": len(group),
            "complete_five_minutes": complete,
            "timestamps": [ts.isoformat() for ts in observed],
            "open": float(group.iloc[0]["open"]),
            "high": float(group["high"].max()),
            "low": float(group["low"].min()),
            "close": float(group.iloc[-1]["close"]),
            "volume": int(group["volume"].sum()),
        }
    return result


def _timestamp_utc(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Kolkata")
    return ts.tz_convert("UTC")


def _compare_yahoo_neighbors(source_bars, yahoo_csv, target_symbol):
    checks = []
    if not yahoo_csv or not Path(yahoo_csv).exists():
        return {"status": "YAHOO_REFERENCE_UNAVAILABLE", "checks": checks, "matches": 0}
    yahoo = pd.read_csv(yahoo_csv)
    ts_col = next((c for c in yahoo.columns if c.lower() in {"timestamp", "datetime", "date", "time"}), None)
    if ts_col is None:
        return {"status": "YAHOO_TIMESTAMP_COLUMN_NOT_FOUND", "checks": checks, "matches": 0}
    yahoo["ts_utc"] = yahoo[ts_col].map(_timestamp_utc)
    for bucket_text in NEIGHBOR_BUCKETS_UTC:
        bucket = pd.Timestamp(bucket_text)
        source = source_bars.get(bucket.isoformat())
        rows = yahoo[yahoo["ts_utc"] == bucket]
        yahoo_row = rows.iloc[0] if not rows.empty else None
        match = False
        detail = None
        if source and source.get("complete_five_minutes") and yahoo_row is not None:
            price_matches = all(
                abs(float(source[field]) - float(yahoo_row[field])) <= 0.01
                for field in ("open", "high", "low", "close")
            )
            volume_matches = int(source["volume"]) == int(float(yahoo_row["volume"]))
            match = price_matches and volume_matches
            detail = {"price_matches": price_matches, "volume_matches": volume_matches}
        checks.append({
            "bucket_utc": bucket.isoformat(),
            "bucket_ist": bucket.tz_convert("Asia/Kolkata").isoformat(),
            "source_bar": source,
            "yahoo_bar": ({
                field: float(yahoo_row[field]) for field in ("open", "high", "low", "close", "volume")
            } if yahoo_row is not None else None),
            "match": match,
            "details": detail,
        })
    matches = sum(check["match"] for check in checks)
    return {"status": "BOTH_NEIGHBORS_MATCH" if matches == 2 else "NEIGHBOR_VALIDATION_FAILED",
            "checks": checks, "matches": matches}


def _write_sparse_parquet(path, file_size, footer_bytes, footer_length, chunks):
    footer_start = file_size - 8 - footer_length
    if footer_length <= 0 or len(footer_bytes) != footer_length + 8:
        raise ValueError("Footer byte length does not match expected metadata plus trailer")
    if footer_bytes[-4:] != b"PAR1" or int.from_bytes(footer_bytes[-8:-4], "little") != footer_length:
        raise ValueError("Parquet footer trailer is invalid")
    for chunk in chunks:
        start = int(chunk["start_offset"])
        end = start + len(chunk["data"])
        if start < 4 or end > footer_start:
            raise ValueError("Selected column chunk overlaps header/footer or exceeds file bounds")
    with open(path, "wb") as handle:
        handle.write(b"PAR1")
        handle.seek(footer_start)
        handle.write(footer_bytes)
        handle.truncate(file_size)
        for chunk in chunks:
            handle.seek(chunk["start_offset"])
            handle.write(chunk["data"])


def probe_candidate_rowgroups(report_path, output, target_symbol="RELIANCE.NS",
                             session=None, max_row_data_bytes=MAX_ROW_DATA_BYTES,
                             max_column_chunk_bytes=MAX_COLUMN_CHUNK_BYTES):
    session = session or requests.Session()
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    groups = _select_candidate_groups(report, target_symbol)
    budget = _check_candidate_budget(groups, max_row_data_bytes, max_column_chunk_bytes)
    result = {
        "dataset": report.get("dataset"),
        "target_symbol": target_symbol,
        "authentication_required": False,
        "full_dataset_download_attempted": False,
        "row_data_download_attempted": False,
        "purpose": "bounded selected Parquet column chunks for manual gap validation; no CSV repair",
        "target_buckets_utc": TARGET_BUCKETS_UTC,
        "neighbor_buckets_utc": NEIGHBOR_BUCKETS_UTC,
        "candidate_groups": [{k: v for k, v in group.items() if k != "columns"} for group in groups],
        "candidate_budget": budget,
        "row_data_bytes_read": 0,
        "footer_metadata_bytes_read": 0,
        "tail_bytes_read": 0,
        "total_range_bytes_read": 0,
        "range_requests": [],
        "target_bars": {},
        "neighbor_validation": None,
        "manual_review_candidate": False,
        "provenance": "Dataset metadata is tagged Upstox; exact source provenance has not been independently verified.",
    }
    if budget["status"] != "WITHIN_BOUNDED_ROW_DATA_CAP":
        result["status"] = budget["status"]
    else:
        # The target symbol is currently expected to fit one candidate group.
        # If multiple files/groups are needed, all selected chunks must fit the cap.
        file_meta = {}
        for group in groups:
            if group["filename"] not in file_meta:
                file_meta[group["filename"]] = group
        fetched = []
        failure = None
        footer_cache = {}
        for group in groups:
            file_size = int(group["file_size"])
            url = group["url"]
            footer_length = int(group.get("footer_length") or 0)
            if footer_length <= 0 or footer_length > MAX_FOOTER_BYTES:
                failure = "FOOTER_SIZE_INVALID_OR_EXCEEDS_CAP"
                break
            if group["filename"] not in footer_cache:
                tail = _range_request(session, url, file_size - 8, file_size - 1)
                result["tail_bytes_read"] += tail.get("bytes_read", 0)
                result["range_requests"].append({"kind": "tail", "filename": group["filename"],
                                                 **{k: v for k, v in tail.items() if k != "data"}})
                if not tail.get("range_valid") or tail.get("data") is None:
                    failure = "TAIL_RANGE_INVALID"
                    break
                if int.from_bytes(tail["data"][:4], "little") != footer_length or tail["data"][-4:] != b"PAR1":
                    failure = "FOOTER_TRAILER_MISMATCH"
                    break
                footer_start = file_size - 8 - footer_length
                footer = _range_request(session, url, footer_start, file_size - 1)
                result["footer_metadata_bytes_read"] += footer.get("bytes_read", 0)
                result["range_requests"].append({"kind": "footer", "filename": group["filename"],
                                                 **{k: v for k, v in footer.items() if k != "data"}})
                if not footer.get("range_valid") or footer.get("data") is None:
                    failure = "FOOTER_RANGE_INVALID"
                    break
                footer_cache[group["filename"]] = footer["data"]
            for chunk in group["columns"]:
                start, end = int(chunk["start_offset"]), int(chunk["end_exclusive"])
                result["row_data_download_attempted"] = True
                response = _range_request(session, url, start, end - 1)
                result["row_data_bytes_read"] += response.get("bytes_read", 0)
                result["range_requests"].append({"kind": "column_chunk", "filename": group["filename"],
                                                 "row_group_index": group["row_group_index"],
                                                 "column": chunk["column"],
                                                 **{k: v for k, v in response.items() if k != "data"}})
                if not response.get("range_valid") or response.get("data") is None:
                    failure = "COLUMN_CHUNK_RANGE_INVALID"
                    break
                fetched.append({**chunk, "filename": group["filename"],
                               "row_group_index": group["row_group_index"], "data": response["data"]})
            if failure:
                break
        result["total_range_bytes_read"] = result["row_data_bytes_read"] + result["footer_metadata_bytes_read"] + result["tail_bytes_read"]
        if failure:
            result["status"] = failure
        else:
            result["row_data_download_attempted"] = bool(fetched)
            tables = []
            with tempfile.TemporaryDirectory() as temp:
                for filename, file_item in file_meta.items():
                    file_size = int(file_item["file_size"])
                    footer_length = int(file_item["footer_length"])
                    footer_bytes = footer_cache[filename]
                    sparse_path = Path(temp) / filename
                    chunks_for_file = [chunk for chunk in fetched if chunk["filename"] == filename]
                    _write_sparse_parquet(sparse_path, file_size, footer_bytes, footer_length, chunks_for_file)
                    try:
                        import pyarrow.parquet as pq
                        parquet = pq.ParquetFile(str(sparse_path))
                        for group in groups:
                            if group["filename"] != filename:
                                continue
                            table = parquet.read_row_group(int(group["row_group_index"]), columns=SELECTED_COLUMNS)
                            tables.append(table.to_pandas())
                    except Exception as exc:
                        failure = f"{type(exc).__name__}: {exc}"
                        break
            if failure:
                result["status"] = "SPARSE_PARQUET_READ_FAILED"
                result["error"] = failure
            elif tables:
                frame = pd.concat(tables, ignore_index=True)
                window_start = "2026-10-15T06:10:00+00:00"
                window_end = "2026-10-15T06:30:00+00:00"
                target_rows = _target_window_rows(frame, target_symbol, window_start, window_end)
                source_bars = _aggregate_minute_rows(frame, target_symbol, window_start, window_end)
                result["rowgroup_rows_loaded"] = int(len(frame))
                result["source_1m_rows_in_window"] = int(len(target_rows))
                result["source_bars_5m"] = source_bars
                result["target_bars"] = {
                    bucket: source_bars.get(bucket, {"rows": 0, "complete_five_minutes": False})
                    for bucket in TARGET_BUCKETS_UTC
                }
                result["target_complete"] = all(
                    bar.get("complete_five_minutes") for bar in result["target_bars"].values()
                )
                yahoo_path = f"data/research/{target_symbol}_5m.csv"
                result["neighbor_validation"] = _compare_yahoo_neighbors(source_bars, yahoo_path, target_symbol)
                result["manual_review_candidate"] = bool(
                    result["target_complete"] and result["neighbor_validation"].get("matches") == 2
                )
                result["status"] = (
                    "MANUAL_REVIEW_CANDIDATE_FOUND" if result["manual_review_candidate"]
                    else "NO_VALIDATED_TARGET_CANDIDATES"
                )
            else:
                result["status"] = "NO_ROWGROUP_TABLES_READ"
    result["interpretation"] = (
        "Only selected symbol candidate row-group column chunks were Range-read under explicit byte caps. "
        "Candidates are for manual review only; source provenance is unverified and no CSVs were modified."
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/huggingface_parquet_range_probe.json")
    parser.add_argument("--symbol", default="RELIANCE.NS")
    parser.add_argument("--output", default="reports/huggingface_parquet_candidate_probe.json")
    parser.add_argument("--max-row-data-bytes", type=int, default=MAX_ROW_DATA_BYTES)
    parser.add_argument("--max-column-chunk-bytes", type=int, default=MAX_COLUMN_CHUNK_BYTES)
    args = parser.parse_args()
    report = probe_candidate_rowgroups(
        args.report, args.output, target_symbol=args.symbol,
        max_row_data_bytes=args.max_row_data_bytes,
        max_column_chunk_bytes=args.max_column_chunk_bytes,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
