"""Bounded HTTP Range probe for public Hugging Face Parquet footers.

Only reads the 8-byte Parquet tail and (if supported) the footer metadata range
for at most two shards. Never downloads Parquet row data or the full dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

DATASET_ID = "xxparthparekhxx/indian-stock-market-minute-data"
PARQUET_API = "https://datasets-server.huggingface.co/parquet"
MAX_FILES = 2
MAX_FOOTER_BYTES = 8 * 1024 * 1024
TIMEOUT = 30


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


def probe_parquet_footer_ranges(output, session=None, dataset=DATASET_ID, max_files=MAX_FILES):
    session = session or requests.Session()
    report = {
        "dataset": dataset,
        "authentication_required": False,
        "full_dataset_download_attempted": False,
        "row_data_download_attempted": False,
        "purpose": "verify bounded HTTP Range access to Parquet footer metadata only",
        "max_files": max_files,
        "max_footer_bytes_per_file": MAX_FOOTER_BYTES,
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
                            else:
                                footer_start = size - 8 - footer_length
                                footer = _range_request(session, url, footer_start, size - 1)
                                item["footer_range"] = {k: v for k, v in footer.items() if k != "data"}
                                total_bytes_read += footer["bytes_read"]
                                if footer.get("range_valid") and footer.get("data") is not None:
                                    item["status"] = "FOOTER_RANGE_RETRIEVED"
                                    item["footer_metadata_bytes_retrieved"] = footer_length
                                    item["footer_magic_valid"] = footer["data"][-4:] == b"PAR1"
                                else:
                                    item["status"] = "FOOTER_RANGE_FAILED"
                except Exception as exc:
                    item["status"] = "ERROR"
                    item["error"] = f"{type(exc).__name__}: {exc}"
                report["files"].append(item)
            report["bounded_bytes_read"] = total_bytes_read
            report["status"] = (
                "RANGE_FOOTER_ACCESS_CONFIRMED"
                if any(f.get("status") == "FOOTER_RANGE_RETRIEVED" for f in report["files"])
                else "RANGE_FOOTER_ACCESS_INCONCLUSIVE"
            )
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["interpretation"] = (
        "Footer-range access confirms only that bounded metadata bytes can be fetched. "
        "It does not prove the footer is parseable, row-group stats are useful, or the target rows are recoverable. "
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
    args = parser.parse_args()
    report = probe_parquet_footer_ranges(args.output, dataset=args.dataset, max_files=args.max_files)
    print(json.dumps({k: report.get(k) for k in ("dataset", "status", "manifest_status", "manifest_partial", "bounded_bytes_read", "files")}, indent=2))


if __name__ == "__main__":
    main()
