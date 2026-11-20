import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from market_intelligence.huggingface_api_diagnostics import (
    run_diagnostics, _config_split_from_splits, _summarize_parquet
)


class Response:
    def __init__(self, status, body, url="https://example.test", headers=None):
        self.status_code=status
        self._body=body
        self.url=url
        self.headers=headers or {"Content-Type":"application/json"}
        self.ok=status < 400
        self.text=body if isinstance(body,str) else __import__("json").dumps(body)
    def json(self):
        if isinstance(self._body,str):
            raise ValueError("not json")
        return self._body


class HuggingFaceApiDiagnosticsTests(unittest.TestCase):
    def test_selects_minute_split_from_metadata(self):
        self.assertEqual(_config_split_from_splits({"splits":[
            {"config":"default","split":"day"},
            {"config":"default","split":"minute"}
        ]}), ("default","minute"))
        self.assertEqual(_config_split_from_splits({}), ("default","minute"))

    def test_summarizes_parquet_files_without_downloading(self):
        result = _summarize_parquet({"parquet_files":[
            {"filename":"a.parquet","size":123,"url":"https://x"},
            {"filename":"b.parquet","size":456}
        ]})
        self.assertEqual(result["file_count"],2)
        self.assertEqual(result["total_bytes_reported"],579)
        self.assertTrue(result["files"][0]["url_present"])

    def test_captures_404_body_and_endpoint_statuses(self):
        session=Mock()
        def get(url,params=None,**kwargs):
            if url.endswith("/splits"):
                return Response(200,{"splits":[{"config":"default","split":"minute"}]},url)
            if url.endswith("/is-valid"):
                return Response(200,{"viewer":True,"filter":False},url)
            if url.endswith("/parquet"):
                return Response(200,{"parquet_files":[{"filename":"shard.parquet","size":10}]},url)
            if url.endswith("/filter"):
                return Response(404,{"error":"filter route unavailable"},url)
            if url.endswith("/rows") or url.endswith("/first-rows"):
                return Response(200,{"rows":[]},url)
            if "/api/datasets/" in url and "/tree/" in url:
                return Response(200,[{"path":"0000.parquet","size":10,"type":"file"}],url)
            return Response(200,{"id":"public/dataset"},url)
        session.get.side_effect=get
        with tempfile.TemporaryDirectory() as d:
            report=run_diagnostics(Path(d)/"report.json",session=session,dataset="owner/name")
            self.assertEqual(report["filter_outcome"]["status_code"],404)
            self.assertIn("filter route unavailable",report["filter_outcome"]["error_excerpt"])
            self.assertFalse(report["download_attempted"])
            self.assertEqual(report["hub_parquet_tree_summary"]["entry_count"],1)
            self.assertTrue(Path(d,"report.json").exists())
        self.assertEqual(session.get.call_count,7)

    def test_network_error_is_recorded_not_raised(self):
        session=Mock()
        session.get.side_effect=RuntimeError("blocked")
        with tempfile.TemporaryDirectory() as d:
            report=run_diagnostics(Path(d)/"report.json",session=session)
            self.assertIsNone(report["diagnostics"]["is_valid"]["status_code"])
            self.assertIn("blocked",report["diagnostics"]["is_valid"]["error"])
            self.assertFalse(report["download_attempted"])


if __name__ == "__main__":
    unittest.main()
