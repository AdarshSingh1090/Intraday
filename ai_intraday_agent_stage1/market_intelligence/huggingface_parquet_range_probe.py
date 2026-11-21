"""Bounded HTTP Range access to Parquet footer metadata and row-group statistics.

This diagnostic fetches no Parquet row data. It uses a sparse temporary file to
let PyArrow parse a footer whose bytes were fetched by bounded HTTP Range requests.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests

DATASET_ID = "xxparthparekhxx/indian-stock-market-minute-data"
PARQUET_API = "https://datasets-server.huggingface.co/parquet"
MAX_FILES = 11
MAX_FOOTER_BYTES = 8 * 1024 * 1024
MAX_TOTAL_FOOTER_BYTES = 16 * 1024 * 1024
TIMEOUT = 30
TARGET_START_UTC = "2026-10-15T06:15:00+00:00"
TARGET_END_UTC = "2026-10-15T06:25:00+00:00"
TARGET_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS",
]


def _read_stream_limited(response, expected_bytes):
    data = bytearray()
    for chunk in response.iter_content(chunk_size=8192):
        if not chunk:
            continue
        remaining = expected_bytes + 1 - len(data)
        if remaining <= 0:
            break
        data.extend(chunk[:remaining])
        if len(data) > expected_bytes:
            break
    return bytes(data)


def _range_request(session, url, start, end, timeout=TIMEOUT):
    expected = end - start + 1
    response = session.get(
        url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        timeout=timeout, stream=True,
    )
    status = int(getattr(response, "status_code", 0))
    headers = getattr(response, "headers", {}) or {}
    content_range = headers.get("Content-Range")
    if status != 206:
        response.close()
        return {"status": status, "range_supported": False, "content_range": content_range, "bytes_read": 0, "data": None}
    data = _read_stream_limited(response, expected)
    response.close()
    range_valid = bool(content_range and content_range.startswith(f"bytes {start}-{end}/"))
    return {
        "status": status, "range_supported": True, "content_range": content_range,
        "bytes_read": len(data), "range_valid": range_valid,
        "data": data if range_valid and len(data) == expected else None,
        "truncated_or_oversized": len(data) != expected,
    }


def _json_scalar(value):
    if value is None:
        return None
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"hex": value.hex()}
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def _parse_datetime_stat(value, arrow_type=None):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        unit = "s"
        type_text = str(arrow_type or "").lower()
        if "timestamp[ns" in type_text:
            unit = "ns"
        elif "timestamp[us" in type_text:
            unit = "us"
        elif "timestamp[ms" in type_text:
            unit = "ms"
        if unit == "ns":
            seconds = value / 1_000_000_000
        elif unit == "us":
            seconds = value / 1_000_000
        elif unit == "ms":
            seconds = value / 1_000
        else:
            seconds = value
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    return None


def _symbol_in_stats(target, minimum, maximum):
    if minimum is None or maximum is None:
        return None
    target_norm = str(target).upper().removesuffix(".NS")
    min_norm = str(minimum).upper().removesuffix(".NS")
    max_norm = str(maximum).upper().removesuffix(".NS")
    if min_norm == max_norm:
        return min_norm == target_norm
    # A min/max range can only be used as a conservative pruning hint.
    return min_norm <= target_norm <= max_norm


def _parse_footer_metadata(file_size, footer_range_data, footer_length, target_start, target_end, target_symbols):
    """Parse footer-only bytes from a sparse local file; never fetch row data."""
    if len(footer_range_data) != footer_length + 8:
        return {"status": "INVALID_FOOTER_BYTES", "error": "Footer range length did not match footer length plus trailer"}
    trailer = footer_range_data[footer_length:]
    if trailer[:4] != footer_length.to_bytes(4, "little") or trailer[4:] != b"PAR1":
        return {"status": "INVALID_FOOTER_TRAILER", "error": "Footer length/magic trailer validation failed"}
    metadata_bytes = footer_range_data[:footer_length]
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {"status": "PYARROW_UNAVAILABLE", "error": "PyArrow is not installed in this runner"}

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as temp:
            temp_path = temp.name
        # Sparse file: only the header and footer ranges are written. Row data
        # regions remain holes and are never read by read_metadata().
        with open(temp_path, "wb") as handle:
            handle.write(b"PAR1")
            handle.seek(file_size - 8 - footer_length)
            handle.write(metadata_bytes)
            handle.write(trailer)
            handle.truncate(file_size)
        metadata = pq.read_metadata(temp_path)
        arrow_schema = getattr(metadata, "schema_arrow", None)
        parquet_schema = getattr(metadata, "schema", None)
        parquet_columns = {
            col.name: col for col in parquet_schema
        } if parquet_schema is not None else {}
        fields = []
        if arrow_schema is not None:
            for field in arrow_schema:
                field_type = str(field.type) if field.type is not None else None
                parquet_col = parquet_columns.get(field.name)
                physical = getattr(parquet_col, "physical_type", None) if parquet_col is not None else None
                logical = getattr(parquet_col, "logical_type", None) if parquet_col is not None else None
                if field_type is None or field_type.lower() in {"none", "null"}:
                    field_type = str(logical) if logical is not None and str(logical).lower() not in {"none", "unknown"} else None
                if field_type is None:
                    field_type = str(physical) if physical is not None else "UNKNOWN"
                fields.append({
                    "name": field.name,
                    "type": field_type,
                    "physical_type": str(physical) if physical is not None else None,
                    "logical_type": str(logical) if logical is not None and str(logical).lower() not in {"none", "unknown"} else None,
                })
        else:
            for col in (parquet_schema or []):
                logical = getattr(col, "logical_type", None)
                physical = getattr(col, "physical_type", None)
                logical_text = str(logical) if logical is not None and str(logical).lower() not in {"none", "unknown"} else None
                physical_text = str(physical) if physical is not None else None
                fields.append({
                    "name": col.name,
                    "type": logical_text or physical_text or "UNKNOWN",
                    "physical_type": physical_text,
                    "logical_type": logical_text,
                })
        columns = [field["name"] for field in fields]
        symbol_candidates = [name for name in columns if name.lower() in {"symbol", "ticker", "stock", "stock_symbol", "instrument"}]
        time_candidates = [name for name in columns if name.lower() in {"timestamp", "datetime", "date_time", "time", "date"}]
        symbol_column = symbol_candidates[0] if symbol_candidates else None
        timestamp_column = time_candidates[0] if time_candidates else None
        type_by_name = {field["name"]: field["type"] for field in fields}
        row_groups = []
        candidate_groups = []
        for rg_index in range(metadata.num_row_groups):
            rg = metadata.row_group(rg_index)
            stats_map = {}
            for col_index in range(rg.num_columns):
                column = rg.column(col_index)
                stats = column.statistics
                if stats is None:
                    continue
                key = getattr(column, "path_in_schema", None) or column.name
                stats_map[key] = {
                    "min": _json_scalar(getattr(stats, "min", None)),
                    "max": _json_scalar(getattr(stats, "max", None)),
                    "null_count": getattr(stats, "null_count", None),
                    "has_min_max": bool(getattr(stats, "has_min_max", False)),
                }
            rg_summary = {
                "index": rg_index,
                "num_rows": rg.num_rows,
                "total_byte_size": rg.total_byte_size,
                "statistics": stats_map,
            }
            # Locate stats by exact field/path name, allowing nested path suffixes.
            sym_stats = next((v for k, v in stats_map.items() if k.split(".")[-1].lower() == (symbol_column or "").lower()), None)
            ts_stats = next((v for k, v in stats_map.items() if k.split(".")[-1].lower() == (timestamp_column or "").lower()), None)
            symbol_matches = None
            time_overlaps = None
            if sym_stats and symbol_column:
                symbol_matches = {
                    target: _symbol_in_stats(target, sym_stats.get("min"), sym_stats.get("max"))
                    for target in target_symbols
                }
            if ts_stats and timestamp_column:
                min_dt = _parse_datetime_stat(ts_stats.get("min"), type_by_name.get(timestamp_column))
                max_dt = _parse_datetime_stat(ts_stats.get("max"), type_by_name.get(timestamp_column))
                if min_dt and max_dt:
                    time_overlaps = min_dt < target_end and max_dt >= target_start
                    rg_summary["timestamp_min_utc"] = min_dt.isoformat()
                    rg_summary["timestamp_max_utc"] = max_dt.isoformat()
            rg_summary["target_symbol_matches"] = symbol_matches
            rg_summary["target_window_overlap"] = time_overlaps
            row_groups.append(rg_summary)
            if symbol_matches and time_overlaps:
                matched = [symbol for symbol, match in symbol_matches.items() if match]
                if matched:
                    candidate_groups.append({"row_group_index": rg_index, "symbols_possible": matched})
        timestamp_stats_groups = sum(
            rg.get("timestamp_min_utc") is not None and rg.get("timestamp_max_utc") is not None
            for rg in row_groups
        )
        timestamp_overlap_groups = sum(rg.get("target_window_overlap") is True for rg in row_groups)
        symbol_candidate_groups = sum(
            bool(rg.get("target_symbol_matches") and any(rg["target_symbol_matches"].values()))
            for rg in row_groups
        )
        date_pruning_status = (
            "EFFECTIVE"
            if timestamp_stats_groups and timestamp_overlap_groups < timestamp_stats_groups
            else ("INEFFECTIVE_ALL_GROUPS_OVERLAP_TARGET" if timestamp_stats_groups else "NO_TIMESTAMP_STATS")
        )
        return {
            "status": "METADATA_PARSED",
            "num_rows": metadata.num_rows,
            "num_row_groups": metadata.num_row_groups,
            "schema": fields,
            "symbol_column": symbol_column,
            "timestamp_column": timestamp_column,
            "row_groups": row_groups,
            "candidate_row_groups_for_target": candidate_groups,
            "timestamp_stats_groups": timestamp_stats_groups,
            "timestamp_overlap_groups": timestamp_overlap_groups,
            "date_pruning_status": date_pruning_status,
            "symbol_candidate_groups": symbol_candidate_groups,
            "candidate_pruning_status": (
                "STATS_AVAILABLE" if symbol_column and timestamp_column else "INSUFFICIENT_SCHEMA_OR_STATS"
            ),
            "target_window_utc": {"start": target_start.isoformat(), "end_exclusive": target_end.isoformat()},
            "target_symbols": target_symbols,
        }
    except Exception as exc:
        return {"status": "METADATA_PARSE_FAILED", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def probe_parquet_footer_ranges(
    output, session=None, dataset=DATASET_ID, max_files=MAX_FILES,
    max_total_footer_bytes=MAX_TOTAL_FOOTER_BYTES,
    target_start=TARGET_START_UTC, target_end=TARGET_END_UTC,
    target_symbols=TARGET_SYMBOLS,
):
    session = session or requests.Session()
    max_files = max(1, min(int(max_files), MAX_FILES))
    max_total_footer_bytes = max(8, int(max_total_footer_bytes))
    start_dt = _parse_datetime_stat(target_start)
    end_dt = _parse_datetime_stat(target_end)
    if start_dt is None or end_dt is None or end_dt <= start_dt:
        raise ValueError("Target timestamp range must be valid and end after start")
    report = {
        "dataset": dataset,
        "authentication_required": False,
        "full_dataset_download_attempted": False,
        "row_data_download_attempted": False,
        "purpose": "bounded Parquet footer metadata parsing and row-group stats pruning; no row data",
        "max_files": max_files,
        "max_footer_bytes_per_file": MAX_FOOTER_BYTES,
        "max_total_footer_bytes": max_total_footer_bytes,
        "target_window_utc": {"start": start_dt.isoformat(), "end_exclusive": end_dt.isoformat()},
        "target_symbols": list(target_symbols),
        "files": [],
    }
    try:
        manifest_response = session.get(PARQUET_API, params={"dataset": dataset}, timeout=TIMEOUT)
        manifest_status = int(getattr(manifest_response, "status_code", 200))
        if manifest_status >= 400:
            report["status"] = "MANIFEST_ERROR"
            report["manifest_status"] = manifest_status
            report["error"] = getattr(manifest_response, "text", "")[:500]
        else:
            manifest = manifest_response.json()
            report["manifest_status"] = manifest_status
            report["manifest_partial"] = manifest.get("partial")
            files = [
                entry for entry in manifest.get("parquet_files", [])
                if entry.get("config") == "minute" and entry.get("split") == "train"
            ][:max_files]
            total_bytes_read = 0
            total_footer_bytes = 0
            for entry in files:
                item = {"filename": entry.get("filename"), "size": entry.get("size"), "url": entry.get("url")}
                try:
                    size = int(entry["size"])
                    url = entry["url"]
                    if size < 8:
                        item["status"] = "INVALID_FILE_SIZE"
                    else:
                        tail = _range_request(session, url, size - 8, size - 1)
                        item["tail_range"] = {k: v for k, v in tail.items() if k != "data"}
                        total_bytes_read += tail["bytes_read"]
                        if not tail.get("range_valid") or tail.get("data") is None:
                            item["status"] = "RANGE_UNSUPPORTED_OR_INVALID_TAIL"
                        elif tail["data"][-4:] != b"PAR1":
                            item["status"] = "INVALID_PARQUET_MAGIC"
                        else:
                            footer_length = int.from_bytes(tail["data"][:4], "little", signed=False)
                            item["footer_length_bytes"] = footer_length
                            if footer_length <= 0 or footer_length > size - 8:
                                item["status"] = "INVALID_FOOTER_LENGTH"
                            elif footer_length > MAX_FOOTER_BYTES:
                                item["status"] = "FOOTER_TOO_LARGE_FOR_BOUNDED_PROBE"
                            elif total_footer_bytes + footer_length + 8 > max_total_footer_bytes:
                                item["status"] = "TOTAL_FOOTER_BUDGET_EXCEEDED"
                            else:
                                footer_start = size - 8 - footer_length
                                footer = _range_request(session, url, footer_start, size - 1)
                                item["footer_range"] = {k: v for k, v in footer.items() if k != "data"}
                                total_bytes_read += footer["bytes_read"]
                                if footer.get("range_valid") and footer.get("data") is not None:
                                    total_footer_bytes += footer_length + 8
                                    item["status"] = "FOOTER_RANGE_RETRIEVED"
                                    item["footer_metadata_bytes_retrieved"] = footer_length
                                    item["footer_magic_valid"] = footer["data"][-4:] == b"PAR1"
                                    item["metadata_parse"] = _parse_footer_metadata(
                                        size, footer["data"], footer_length, start_dt, end_dt, list(target_symbols)
                                    )
                                else:
                                    item["status"] = "FOOTER_RANGE_FAILED"
                except Exception as exc:
                    item["status"] = "ERROR"
                    item["error"] = f"{type(exc).__name__}: {exc}"
                report["files"].append(item)
            report["bounded_bytes_read"] = total_bytes_read
            report["footer_metadata_bytes_read"] = total_footer_bytes
            parsed = sum(1 for f in report["files"] if f.get("metadata_parse", {}).get("status") == "METADATA_PARSED")
            report["metadata_files_parsed"] = parsed
            report["status"] = (
                "RANGE_FOOTER_ACCESS_CONFIRMED"
                if any(f.get("status") == "FOOTER_RANGE_RETRIEVED" for f in report["files"])
                else "RANGE_FOOTER_ACCESS_INCONCLUSIVE"
            )
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["interpretation"] = (
        "Footer parsing and row-group statistics are metadata only. Row-group min/max values are pruning hints, "
        "not proof of row presence. Any candidate row groups must be separately range-read and validated. "
        "No Parquet row data was requested."
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/huggingface_parquet_range_probe.json")
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--max-files", type=int, default=MAX_FILES)
    parser.add_argument("--max-total-footer-bytes", type=int, default=MAX_TOTAL_FOOTER_BYTES)
    parser.add_argument("--target-start", default=TARGET_START_UTC)
    parser.add_argument("--target-end", default=TARGET_END_UTC)
    parser.add_argument("--symbols", default=",".join(TARGET_SYMBOLS))
    args = parser.parse_args()
    report = probe_parquet_footer_ranges(
        args.output, dataset=args.dataset, max_files=args.max_files,
        max_total_footer_bytes=args.max_total_footer_bytes,
        target_start=args.target_start, target_end=args.target_end,
        target_symbols=[symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()],
    )
    print(json.dumps({
        "dataset": report.get("dataset"), "status": report.get("status"),
        "manifest_status": report.get("manifest_status"), "manifest_partial": report.get("manifest_partial"),
        "bounded_bytes_read": report.get("bounded_bytes_read"),
        "footer_metadata_bytes_read": report.get("footer_metadata_bytes_read"),
        "metadata_files_parsed": report.get("metadata_files_parsed"),
        "files": report.get("files"),
    }, indent=2))


if __name__ == "__main__":
    main()
