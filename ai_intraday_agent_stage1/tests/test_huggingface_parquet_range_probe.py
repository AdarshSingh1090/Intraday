import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import types
from datetime import datetime, timezone

from market_intelligence.huggingface_parquet_range_probe import probe_parquet_footer_ranges, _column_chunk_range, _summarize_candidate_chunk_plan, MAX_CANDIDATE_COLUMN_CHUNK_BYTES, MAX_CANDIDATE_CHUNK_PLAN_BYTES


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

    def test_footer_parser_uses_sparse_file_and_reports_rowgroup_stats(self):
        file_size = 4096
        metadata_bytes = b"PARQUET_META"
        footer = metadata_bytes + len(metadata_bytes).to_bytes(4, "little") + b"PAR1"
        tail = len(metadata_bytes).to_bytes(4, "little") + b"PAR1"
        manifest = {"partial": False, "parquet_files": [{
            "config": "minute", "split": "train", "filename": "0000.parquet",
            "size": file_size, "url": "https://example.test/file.parquet"
        }]}

        class Stats:
            def __init__(self, minimum, maximum):
                self.min, self.max, self.null_count, self.has_min_max = minimum, maximum, 0, True
        class Column:
            def __init__(self, name, stats):
                self.path_in_schema, self.statistics = name, stats
        class Group:
            num_rows, total_byte_size, num_columns = 10, 100, 2
            def column(self, i):
                return [Column("symbol", Stats("RELIANCE", "RELIANCE")),
                        Column("timestamp", Stats(datetime(2026,10,15,6,15,tzinfo=timezone.utc),
                                                datetime(2026,10,15,6,24,tzinfo=timezone.utc)))][i]
        class Field:
            def __init__(self, name, typ):
                self.name, self.type = name, typ
        class ParquetColumn:
            def __init__(self, name, physical, logical=None):
                self.name, self.physical_type, self.logical_type = name, physical, logical
        class Schema:
            def __iter__(self):
                return iter([Field("symbol", "string"), Field("timestamp", "timestamp[us, tz=UTC]"), Field("open", None)])
        class Meta:
            num_rows, num_row_groups = 10, 1
            schema_arrow = Schema()
            schema = [ParquetColumn("symbol", "BYTE_ARRAY", "String"),
                      ParquetColumn("timestamp", "INT64", "Timestamp"),
                      ParquetColumn("open", "DOUBLE", None)]
            def row_group(self, i):
                return Group()
        pyarrow = types.ModuleType("pyarrow")
        parquet = types.ModuleType("pyarrow.parquet")
        def read_metadata(path):
            with open(path, "rb") as f:
                f.seek(0)
                self.assertEqual(f.read(4), b"PAR1")
                f.seek(file_size - 8)
                self.assertEqual(f.read(8), len(metadata_bytes).to_bytes(4, "little")+b"PAR1")
            return Meta()
        parquet.read_metadata = read_metadata
        pyarrow.parquet = parquet
        session = Mock()
        session.get.side_effect = [
            RangeResponse(200, b"", json_payload=manifest),
            RangeResponse(206, tail, f"bytes {file_size-8}-{file_size-1}/{file_size}"),
            RangeResponse(206, footer, f"bytes {file_size-len(footer)}-{file_size-1}/{file_size}"),
        ]
        with patch.dict(sys.modules, {"pyarrow": pyarrow, "pyarrow.parquet": parquet}):
            with tempfile.TemporaryDirectory() as temp:
                result = probe_parquet_footer_ranges(Path(temp)/"report.json", session=session)
        parsed = result["files"][0]["metadata_parse"]
        self.assertEqual(parsed["status"], "METADATA_PARSED")
        self.assertEqual(parsed["num_row_groups"], 1)
        self.assertEqual(parsed["candidate_row_groups_for_target"][0]["symbols_possible"], ["RELIANCE.NS"])
        open_field = next(field for field in parsed["schema"] if field["name"] == "open")
        self.assertEqual(open_field["type"], "DOUBLE")
        self.assertEqual(open_field["physical_type"], "DOUBLE")
        self.assertEqual(parsed["date_pruning_status"], "INEFFECTIVE_ALL_GROUPS_OVERLAP_TARGET")
        self.assertFalse(result["row_data_download_attempted"])

    def test_total_footer_budget_prevents_later_range_requests(self):
        file_size = 2048
        metadata = b"x" * 40
        footer = metadata + len(metadata).to_bytes(4, "little") + b"PAR1"
        tail = len(metadata).to_bytes(4, "little") + b"PAR1"
        manifest = {"partial": False, "parquet_files": [
            {"config": "minute", "split": "train", "filename": "0000.parquet", "size": file_size, "url": "https://example.test/0"},
            {"config": "minute", "split": "train", "filename": "0001.parquet", "size": file_size, "url": "https://example.test/1"},
        ]}
        session = Mock()
        session.get.side_effect = [
            RangeResponse(200, b"", json_payload=manifest),
            RangeResponse(206, tail, f"bytes {file_size-8}-{file_size-1}/{file_size}"),
            RangeResponse(206, footer, f"bytes {file_size-len(footer)}-{file_size-1}/{file_size}"),
            RangeResponse(206, tail, f"bytes {file_size-8}-{file_size-1}/{file_size}"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            result = probe_parquet_footer_ranges(Path(temp)/"report.json", session=session, max_total_footer_bytes=60)
        self.assertEqual(result["files"][0]["status"], "FOOTER_RANGE_RETRIEVED")
        self.assertEqual(result["files"][1]["status"], "TOTAL_FOOTER_BUDGET_EXCEEDED")
        self.assertEqual(session.get.call_count, 4)


    def test_column_chunk_range_uses_dictionary_offset_and_respects_footer(self):
        class Chunk:
            dictionary_page_offset = 100
            data_page_offset = 150
            total_compressed_size = 80
            total_uncompressed_size = 120
            compression = "SNAPPY"
        result = _column_chunk_range(Chunk(), file_size=1000, footer_start=900)
        self.assertTrue(result["valid"])
        self.assertEqual(result["start_offset"],100)
        self.assertEqual(result["end_exclusive"],180)
        self.assertEqual(result["compressed_size"],80)
        self.assertEqual(result["compression"],"SNAPPY")

    def test_column_chunk_range_rejects_missing_offsets_and_footer_overlap(self):
        class Missing:
            dictionary_page_offset = -1
            data_page_offset = -1
            total_compressed_size = 100
        self.assertFalse(_column_chunk_range(Missing(),1000,900)["valid"])
        class Overlap:
            dictionary_page_offset = 850
            data_page_offset = 860
            total_compressed_size = 100
        result = _column_chunk_range(Overlap(),1000,900)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"],"range_overlaps_footer_or_exceeds_file")

    def test_candidate_chunk_plan_caps_and_empty_status(self):
        self.assertEqual(_summarize_candidate_chunk_plan([])["status"], "NO_CANDIDATE_CHUNKS")
        valid = [{"columns":[{"valid":True,"compressed_size":100}],
                  "compressed_bytes_total":100,"all_ranges_valid":True}]
        self.assertEqual(_summarize_candidate_chunk_plan(valid)["status"], "WITHIN_BOUNDED_PLAN_CAP")
        single = [{"columns":[{"valid":True,"compressed_size":MAX_CANDIDATE_COLUMN_CHUNK_BYTES+1}],
                   "compressed_bytes_total":MAX_CANDIDATE_COLUMN_CHUNK_BYTES+1,"all_ranges_valid":True}]
        self.assertEqual(_summarize_candidate_chunk_plan(single)["status"], "SINGLE_CHUNK_CAP_EXCEEDED")
        total = [{"columns":[{"valid":True,"compressed_size":MAX_CANDIDATE_CHUNK_PLAN_BYTES//2+1}],
                   "compressed_bytes_total":MAX_CANDIDATE_CHUNK_PLAN_BYTES//2+1,"all_ranges_valid":True},
                  {"columns":[{"valid":True,"compressed_size":MAX_CANDIDATE_CHUNK_PLAN_BYTES//2+1}],
                   "compressed_bytes_total":MAX_CANDIDATE_CHUNK_PLAN_BYTES//2+1,"all_ranges_valid":True}]
        self.assertEqual(_summarize_candidate_chunk_plan(total)["status"], "TOTAL_PLAN_CAP_EXCEEDED")

if __name__ == "__main__":
    unittest.main()
