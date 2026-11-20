import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from market_intelligence.huggingface_parquet_range_probe import probe_parquet_footer_ranges


class RangeResponse:
    def __init__(self, status_code, content, content_range=None, json_payload=None):
        self.status_code = status_code
        self._content = content
        self.headers = {"Content-Range": content_range} if content_range else {}
        self._json = json_payload
        self.closed = False
        self.read_calls = 0
    def iter_content(self, chunk_size=8192):
        self.read_calls += 1
        yield self._content
    def close(self):
        self.closed = True
    def json(self):
        return self._json


class HuggingFaceParquetRangeProbeTests(unittest.TestCase):
    def test_reads_only_tail_and_footer_ranges_and_reports_bounded_bytes(self):
        file_size = 2048
        metadata = b"small fake metadata payload"
        tail = len(metadata).to_bytes(4, "little") + b"PAR1"
        footer = metadata + tail
        tail_start = file_size - 8
        footer_start = file_size - len(footer)
        manifest = {"partial": False, "parquet_files": [{
            "config": "minute", "split": "train", "filename": "0000.parquet",
            "size": file_size, "url": "https://example.test/file.parquet"
        }]}
        session = Mock()
        session.get.side_effect = [
            RangeResponse(200, b"", json_payload=manifest),
            RangeResponse(206, tail, f"bytes {tail_start}-{file_size-1}/{file_size}"),
            RangeResponse(206, footer, f"bytes {footer_start}-{file_size-1}/{file_size}"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            result = probe_parquet_footer_ranges(Path(temp)/"report.json", session=session)
        self.assertEqual(result["status"], "RANGE_FOOTER_ACCESS_CONFIRMED")
        self.assertEqual(result["bounded_bytes_read"], len(tail)+len(footer))
        self.assertEqual(len(result["files"]), 1)
        self.assertEqual(result["files"][0]["status"], "FOOTER_RANGE_RETRIEVED")
        calls = session.get.call_args_list
        self.assertEqual(calls[1].kwargs["headers"]["Range"], f"bytes={tail_start}-{file_size-1}")
        self.assertEqual(calls[2].kwargs["headers"]["Range"], f"bytes={footer_start}-{file_size-1}")
        self.assertTrue(calls[1].kwargs["stream"])
        self.assertFalse(result["row_data_download_attempted"])

    def test_server_ignoring_range_does_not_read_body_or_request_footer(self):
        manifest = {"partial": False, "parquet_files": [{
            "config": "minute", "split": "train", "filename": "0000.parquet",
            "size": 2048, "url": "https://example.test/file.parquet"
        }]}
        ignored = RangeResponse(200, b"SHOULD_NOT_BE_READ")
        session = Mock()
        session.get.side_effect = [RangeResponse(200, b"", json_payload=manifest), ignored]
        with tempfile.TemporaryDirectory() as temp:
            result = probe_parquet_footer_ranges(Path(temp)/"report.json", session=session)
        self.assertEqual(result["status"], "RANGE_FOOTER_ACCESS_INCONCLUSIVE")
        self.assertEqual(result["files"][0]["status"], "RANGE_UNSUPPORTED_OR_INVALID_TAIL")
        self.assertEqual(ignored.read_calls, 0)
        self.assertEqual(session.get.call_count, 2)

    def test_oversized_footer_is_not_requested(self):
        file_size = 100_000_000
        footer_length = 9 * 1024 * 1024
        tail = footer_length.to_bytes(4, "little") + b"PAR1"
        manifest = {"partial": False, "parquet_files": [{
            "config": "minute", "split": "train", "filename": "0000.parquet",
            "size": file_size, "url": "https://example.test/file.parquet"
        }]}
        session = Mock()
        session.get.side_effect = [
            RangeResponse(200, b"", json_payload=manifest),
            RangeResponse(206, tail, f"bytes {file_size-8}-{file_size-1}/{file_size}"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            result = probe_parquet_footer_ranges(Path(temp)/"report.json", session=session)
        self.assertEqual(result["files"][0]["status"], "FOOTER_TOO_LARGE_FOR_BOUNDED_PROBE")
        self.assertEqual(session.get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
