"""Small, no-auth Hugging Face viewer diagnostics; no dataset download.

Captures status, response body, and capability flags for the public Dataset Viewer
API endpoints and metadata-only Hub listing. Intended to explain failed filters
before attempting bounded Parquet range access.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import requests

DATASET_ID = "xxparthparekhxx/indian-stock-market-minute-data"
BASE = "https://datasets-server.huggingface.co"
HUB = "https://huggingface.co"
BODY_LIMIT = 6000
TIMEOUT = 30


def _capture(session, name, url, params=None, timeout=TIMEOUT):
    started = time.perf_counter()
    try:
        response = session.get(url, params=params, timeout=timeout, headers={"User-Agent": "intraday-stage8-diagnostics/1.0"})
        elapsed = round(time.perf_counter() - started, 3)
        body = response.text
        try:
            parsed = response.json()
        except Exception:
            parsed = None
        return {
            "name": name,
            "status_code": response.status_code,
            "ok": response.ok,
            "elapsed_seconds": elapsed,
            "final_url": response.url,
            "content_type": response.headers.get("Content-Type"),
            "server": response.headers.get("Server"),
            "error_code_header": response.headers.get("X-Error-Code"),
            "body_excerpt": body[:BODY_LIMIT],
            "body_truncated": len(body) > BODY_LIMIT,
            "json": parsed if isinstance(parsed, (dict, list)) else None,
        }
    except Exception as exc:
        return {
            "name": name,
            "status_code": None,
            "ok": False,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "body_excerpt": None,
            "json": None,
        }


def _config_split_from_splits(payload):
    if not isinstance(payload, dict):
        return "default", "train"
    entries = payload.get("splits")
    if not isinstance(entries, list):
        return "default", "train"
    # HF reports this dataset as two configs (day/minute), each with split=train.
    for entry in entries:
        if isinstance(entry, dict) and entry.get("config") == "minute":
            return entry.get("config", "minute"), entry.get("split", "train")
    if entries and isinstance(entries[0], dict):
        return entries[0].get("config", "default"), entries[0].get("split", "train")
    return "default", "train"


def _summarize_parquet(payload):
    if not isinstance(payload, dict):
        return {"parseable": False}
    files = payload.get("parquet_files") or payload.get("files") or []
    summary = []
    total_size = 0
    for entry in files:
        if not isinstance(entry, dict):
            continue
        size = entry.get("size")
        if isinstance(size, int):
            total_size += size
        summary.append({
            "filename": entry.get("filename") or entry.get("url"),
            "size": size,
            "split": entry.get("split"),
            "config": entry.get("config"),
            "url_present": bool(entry.get("url")),
        })
    return {
        "parseable": True,
        "file_count": len(summary),
        "total_bytes_reported": total_size,
        "files": summary[:50],
        "files_truncated": len(summary) > 50,
    }


def _summarize_tree(payload):
    if not isinstance(payload, list):
        return {"parseable": False}
    entries = []
    total = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        size = item.get("size")
        if isinstance(size, int):
            total += size
        entries.append({"path": item.get("path"), "size": size, "type": item.get("type")})
    return {"parseable": True, "entry_count": len(entries), "total_bytes_reported": total, "entries": entries[:100], "entries_truncated": len(entries) > 100}


def run_diagnostics(output, session=None, dataset=DATASET_ID):
    session = session or requests.Session()
    report = {
        "dataset": dataset,
        "authentication_required": False,
        "download_attempted": False,
        "large_file_download_attempted": False,
        "diagnostics": {},
        "interpretation": "Endpoint checks and metadata only. No Parquet data files downloaded. HTTP failures are diagnostic, not evidence that source data is absent.",
    }
    report["diagnostics"]["is_valid"] = _capture(session, "is-valid", f"{BASE}/is-valid", {"dataset": dataset})
    report["diagnostics"]["splits"] = _capture(session, "splits", f"{BASE}/splits", {"dataset": dataset})
    report["diagnostics"]["parquet"] = _capture(session, "parquet", f"{BASE}/parquet", {"dataset": dataset})

    config, split = _config_split_from_splits(report["diagnostics"]["splits"].get("json"))
    report["selected_config_split"] = {"config": config, "split": split}

    report["diagnostics"]["rows"] = _capture(
        session, "rows", f"{BASE}/rows",
        {"dataset": dataset, "config": config, "split": split, "offset": 0, "length": 1},
    )
    report["diagnostics"]["first_rows"] = _capture(
        session, "first-rows", f"{BASE}/first-rows",
        {"dataset": dataset, "config": config, "split": split},
    )
    # A deliberately small symbol-only filter isolates route/capability support
    # from timestamp predicate complexity.
    report["diagnostics"]["filter_symbol"] = _capture(
        session, "filter-symbol", f"{BASE}/filter",
        {"dataset": dataset, "config": config, "split": split,
         "where": "\"symbol\"='RELIANCE'", "offset": 0, "length": 5},
    )
    # Bounded target-window query tests the composite predicate used by recovery.
    report["diagnostics"]["filter_target_window"] = _capture(
        session, "filter-target-window", f"{BASE}/filter",
        {"dataset": dataset, "config": "minute", "split": "train",
         "where": "\"symbol\"='RELIANCE' AND \"timestamp\">='2026-10-15T06:10:00Z' AND \"timestamp\"<'2026-10-15T06:30:00Z'",
         "offset": 0, "length": 100},
    )

    report["parquet_manifest_summary"] = _summarize_parquet(report["diagnostics"]["parquet"].get("json"))
    report["diagnostics"]["hub_metadata"] = _capture(
        session, "hub-metadata", f"{HUB}/api/datasets/{dataset}",
    )
    # Metadata-only tree call for converted Parquet split; no file content.
    tree_url = f"{HUB}/api/datasets/{dataset}/tree/refs%2Fconvert%2Fparquet/minute/train"
    report["diagnostics"]["hub_parquet_tree"] = _capture(
        session, "hub-parquet-tree", tree_url, {"recursive": "true", "expand": "true"},
    )
    report["hub_parquet_tree_summary"] = _summarize_tree(report["diagnostics"]["hub_parquet_tree"].get("json"))

    is_valid = report["diagnostics"]["is_valid"].get("json")
    if isinstance(is_valid, dict):
        report["viewer_capabilities"] = is_valid
    filter_result = report["diagnostics"]["filter_symbol"]
    report["filter_outcome"] = {
        "symbol_query_status": filter_result.get("status_code"),
        "symbol_query_ok": filter_result.get("ok"),
        "symbol_query_error_excerpt": filter_result.get("body_excerpt") if not filter_result.get("ok") else None,
        "target_window_status": report["diagnostics"]["filter_target_window"].get("status_code"),
        "target_window_ok": report["diagnostics"]["filter_target_window"].get("ok"),
        "target_window_error_excerpt": report["diagnostics"]["filter_target_window"].get("body_excerpt") if not report["diagnostics"]["filter_target_window"].get("ok") else None,
        "viewer_filter_capability": is_valid.get("filter") if isinstance(is_valid, dict) else None,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/huggingface_api_diagnostics.json")
    parser.add_argument("--dataset", default=DATASET_ID)
    args = parser.parse_args()
    report = run_diagnostics(args.output, dataset=args.dataset)
    print(json.dumps({
        "dataset": report["dataset"],
        "capabilities": report.get("viewer_capabilities"),
        "endpoint_statuses": {name: item.get("status_code") for name, item in report["diagnostics"].items()},
        "filter_outcome": report.get("filter_outcome"),
        "parquet_manifest": report.get("parquet_manifest_summary"),
        "tree": report.get("hub_parquet_tree_summary"),
        "download_attempted": report["download_attempted"],
    }, indent=2))


if __name__ == "__main__":
    main()
