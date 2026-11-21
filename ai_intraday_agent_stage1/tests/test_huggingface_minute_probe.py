import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from market_intelligence.huggingface_minute_probe import (
    aggregate_exact_5m_from_minute,
    fetch_public_minute_window,
    probe_public_minute_dataset,
    expected_5m_grid,
)


class FakeResponse:
    def __init__(self, payload, status_code=200, text=None, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.text = text if text is not None else str(payload)
        self.headers = headers or {}
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
    def json(self):
        return self.payload


class HuggingFaceMinuteProbeTests(unittest.TestCase):
    def test_filter_response_parses_utc_rows_and_query(self):
        payload = {"rows": [{"row_idx": i, "row": {
            "symbol": "RELIANCE", "timestamp": f"2026-10-15T06:{15+i:02d}:00Z",
            "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 100, "oi": 0
        }} for i in range(5)], "partial": False, "num_rows_total": 5}
        session = Mock()
        session.get.return_value = FakeResponse(payload)
        start = pd.Timestamp("2026-10-15 11:45:00+05:30")
        end = start + pd.Timedelta(minutes=5)
        result = fetch_public_minute_window(session, "RELIANCE", start, end)
        self.assertEqual(len(result["frame"]), 5)
        self.assertEqual(result["frame"].timestamp.iloc[0].strftime("%H:%M"), "11:45")
        params = session.get.call_args.kwargs["params"]
        self.assertIn('"symbol"=\'RELIANCE\'', params["where"])
        self.assertIn('"timestamp">=', params["where"])
        self.assertEqual(params["config"], "minute")
        self.assertEqual(params["split"], "train")
        self.assertEqual(params["length"], 100)
        # Viewer responses omit config/split; report should preserve request context.
        self.assertEqual(result["config"], "minute")
        self.assertEqual(result["split"], "train")
        self.assertEqual(result["requested_config"], "minute")
        self.assertEqual(result["requested_split"], "train")
        self.assertIsNone(result["response_config"])
        self.assertIsNone(result["response_split"])

    def test_aggregates_only_complete_exact_5m_window(self):
        stamp = expected_5m_grid(pd.Timestamp("2026-10-15").date())[30]
        frame = pd.DataFrame({
            "timestamp": pd.date_range(stamp, periods=5, freq="1min"),
            "open": [100, 100.1, 100.2, 100.3, 100.4],
            "high": [101, 101.5, 102, 101.4, 101.6],
            "low": [99, 99.5, 98.8, 99.2, 99.4],
            "close": [100.1, 100.2, 100.3, 100.4, 100.5],
            "volume": [100, 200, 150, 250, 300],
        })
        candidate = aggregate_exact_5m_from_minute(frame, stamp)
        self.assertEqual(candidate, {"open": 100.0, "high": 102.0, "low": 98.8, "close": 100.5, "volume": 1000.0})
        self.assertIsNone(aggregate_exact_5m_from_minute(frame.iloc[:-1], stamp))

    def test_probe_candidate_neighbor_matches_and_never_mutates(self):
        day = pd.Timestamp("2026-10-15").date()
        grid = expected_5m_grid(day)
        missing = [grid[30], grid[31]]
        original = pd.DataFrame({
            "timestamp": grid.delete([30, 31]).astype(str), "open": 100, "high": 101,
            "low": 99, "close": 100.5, "volume": 2000
        })
        window_start = missing[0] - pd.Timedelta(minutes=5)
        window_end = missing[-1] + pd.Timedelta(minutes=10)
        minute_timestamps = pd.date_range(window_start, window_end, freq="1min", inclusive="left")
        minute_rows = [{
            "symbol": "RELIANCE",
            "timestamp": stamp.tz_convert("UTC").isoformat(),
            "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 400, "oi": 0,
        } for stamp in minute_timestamps]
        payload = {"rows": [{"row_idx": i, "row": row} for i, row in enumerate(minute_rows)],
                   "partial": False, "num_rows_total": len(minute_rows)}
        session = Mock()
        session.get.return_value = FakeResponse(payload)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "RELIANCE.NS_5m.csv"
            original.to_csv(source, index=False)
            report = probe_public_minute_dataset(temp, ["RELIANCE.NS"], Path(temp)/"report.json", session=session)
            item = report["reports"]["RELIANCE.NS"]
            day_report = item["missing_by_day"]["2026-10-15"]["public_dataset_query"]
            self.assertEqual(item["complete_5m_candidates_from_public_1m_dataset"], 2)
            self.assertEqual(day_report["status"], "OK")
            self.assertFalse(day_report["partial_index"])
            self.assertEqual(day_report["neighbor_validation_against_yahoo_5m"]["matches"], 2)
            self.assertEqual(day_report["candidate_status"], "CANDIDATES_AVAILABLE_DIAGNOSTIC_ONLY")
            self.assertFalse(report["independent_provider_comparison"])
            self.assertFalse(report["source_provenance_verified"])
            self.assertFalse(report["datasets_modified"])
            self.assertEqual(len(pd.read_csv(source)), len(original))
            self.assertTrue((Path(temp)/"report.json").exists())

    def test_probe_withholds_candidate_when_filter_index_partial(self):
        day = pd.Timestamp("2026-10-15").date()
        grid = expected_5m_grid(day)
        missing = grid[30]
        original = pd.DataFrame({
            "timestamp": grid.delete(30).astype(str), "open": 100, "high": 101,
            "low": 99, "close": 100.5, "volume": 2000
        })
        minute_stamps = pd.date_range(missing, periods=5, freq="1min")
        payload = {"rows": [{"row_idx": i, "row": {
            "symbol": "RELIANCE", "timestamp": t.tz_convert("UTC").isoformat(),
            "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 100, "oi": 0
        }} for i, t in enumerate(minute_stamps)], "partial": True, "num_rows_total": 5}
        session = Mock()
        session.get.return_value = FakeResponse(payload)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "RELIANCE.NS_5m.csv"
            original.to_csv(source, index=False)
            report = probe_public_minute_dataset(temp, ["RELIANCE.NS"], Path(temp)/"report.json", session=session)
            query = report["reports"]["RELIANCE.NS"]["missing_by_day"]["2026-10-15"]["public_dataset_query"]
            self.assertEqual(query["raw_complete_candidate_count"], 1)
            self.assertEqual(query["candidate_status"], "WITHHELD_PARTIAL_INDEX")
            self.assertEqual(query["raw_candidates_unverified"][str(missing)]["open"], 100)
            self.assertIsNone(query["reconstructed_5m_candidates"][str(missing)])
            self.assertIsNone(query["reconstructed_5m_candidates"][str(missing)])
            self.assertEqual(report["summary"]["exact_5m_candidates_found"], 0)
            self.assertFalse(report["datasets_modified"])
            self.assertEqual(len(pd.read_csv(source)), len(original))

    def test_candidate_withheld_when_neighbor_validation_mismatches(self):
        day = pd.Timestamp("2026-10-15").date()
        grid = expected_5m_grid(day)
        missing = [grid[30], grid[31]]
        original = pd.DataFrame({
            "timestamp": grid.delete([30, 31]).astype(str), "open": 100, "high": 101,
            "low": 99, "close": 100.5, "volume": 2000
        })
        window_start = missing[0] - pd.Timedelta(minutes=5)
        window_end = missing[-1] + pd.Timedelta(minutes=10)
        minute_timestamps = pd.date_range(window_start, window_end, freq="1min", inclusive="left")
        minute_rows = [{
            "symbol": "RELIANCE",
            "timestamp": stamp.tz_convert("UTC").isoformat(),
            "open": 101, "high": 102, "low": 100, "close": 101.5, "volume": 400, "oi": 0,
        } for stamp in minute_timestamps]
        payload = {"rows": [{"row_idx": i, "row": row} for i, row in enumerate(minute_rows)],
                   "partial": False, "num_rows_total": len(minute_rows)}
        session = Mock()
        session.get.return_value = FakeResponse(payload)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "RELIANCE.NS_5m.csv"
            original.to_csv(source, index=False)
            report = probe_public_minute_dataset(temp, ["RELIANCE.NS"], Path(temp)/"report.json", session=session)
            query = report["reports"]["RELIANCE.NS"]["missing_by_day"]["2026-10-15"]["public_dataset_query"]
            self.assertEqual(query["candidate_status"], "WITHHELD_NEIGHBOR_MISMATCH_OR_UNVALIDATED")
            self.assertEqual(report["summary"]["exact_5m_candidates_found"], 0)
            self.assertEqual(query["raw_complete_candidate_count"], 2)
            self.assertIsNone(query["reconstructed_5m_candidates"][str(missing[0])])
            self.assertIsNotNone(query["raw_candidates_unverified"][str(missing[0])])
            self.assertEqual(query["neighbor_validation_against_yahoo_5m"]["mismatches"], 2)
            self.assertEqual(len(query["neighbor_validation_against_yahoo_5m"]["mismatch_examples"]), 2)
            self.assertFalse(report["datasets_modified"])
            self.assertEqual(len(pd.read_csv(source)), len(original))


    def test_filter_retries_transient_500_and_reports_attempts(self):
        payload = {"rows": [{"row": {"timestamp": "2026-10-15T06:15:00Z",
            "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 10}}],
            "partial": False, "num_rows_total": 1}
        session = Mock()
        session.get.side_effect = [
            FakeResponse({"error": "dataset index is loading"}, 500,
                         text="dataset index is loading"),
            FakeResponse(payload, 200),
        ]
        sleeper = Mock()
        start = pd.Timestamp("2026-10-15 11:45:00+05:30")
        result = fetch_public_minute_window(
            session, "RELIANCE", start, start + pd.Timedelta(minutes=5),
            sleep_func=sleeper,
        )
        self.assertEqual(result["request_attempts"], 2)
        self.assertEqual(result["retry_statuses"], [500])
        self.assertEqual(len(result["frame"]), 1)
        self.assertEqual(session.get.call_count, 2)
        sleeper.assert_called_once_with(2)

    def test_terminal_404_is_not_retried(self):
        session = Mock()
        session.get.return_value = FakeResponse({"error": "not found"}, 404, text="not found")
        start = pd.Timestamp("2026-10-15 11:45:00+05:30")
        with self.assertRaises(Exception) as raised:
            fetch_public_minute_window(
                session, "RELIANCE", start, start + pd.Timedelta(minutes=5),
                sleep_func=Mock(),
            )
        self.assertEqual(getattr(raised.exception, "status_code", None), 404)
        self.assertEqual(session.get.call_count, 1)

    def test_index_loading_circuit_breaker_skips_redundant_symbol_queries(self):
        day = pd.Timestamp("2026-10-15").date()
        grid = expected_5m_grid(day)
        original = pd.DataFrame({
            "timestamp": grid.delete([30, 31]).astype(str), "open": 100, "high": 101,
            "low": 99, "close": 100.5, "volume": 2000
        })
        session = Mock()
        session.get.side_effect = [
            FakeResponse({"error": "the dataset index is loading"}, 500,
                         text="the dataset index is loading"),
            FakeResponse({"error": "the dataset index is loading"}, 500,
                         text="the dataset index is loading"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            for symbol in ["RELIANCE.NS", "TCS.NS"]:
                original.to_csv(Path(temp) / f"{symbol}_5m.csv", index=False)
            report = probe_public_minute_dataset(
                temp, ["RELIANCE.NS", "TCS.NS"], Path(temp)/"report.json",
                session=session,
            )
            first = report["reports"]["RELIANCE.NS"]["missing_by_day"]["2026-10-15"]["public_dataset_query"]
            second = report["reports"]["TCS.NS"]["missing_by_day"]["2026-10-15"]["public_dataset_query"]
            self.assertEqual(first["http_status"], 500)
            self.assertTrue(first["index_loading_error"])
            self.assertEqual(first["request_attempts"], 2)
            self.assertEqual(second["status"], "SKIPPED_INDEX_LOADING")
            self.assertEqual(session.get.call_count, 2)
            self.assertTrue(report["summary"]["dataset_index_loading_circuit_breaker_triggered"])

    def test_provider_error_is_inconclusive_and_source_unchanged(self):
        day = pd.Timestamp("2026-10-15").date()
        grid = expected_5m_grid(day)
        original = pd.DataFrame({
            "timestamp": grid.delete([30, 31]).astype(str), "open": 100, "high": 101,
            "low": 99, "close": 100.5, "volume": 2000
        })
        session = Mock()
        session.get.side_effect = RuntimeError("viewer unavailable")
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "RELIANCE.NS_5m.csv"
            original.to_csv(source, index=False)
            report = probe_public_minute_dataset(temp, ["RELIANCE.NS"], Path(temp)/"report.json", session=session)
            item = report["reports"]["RELIANCE.NS"]
            self.assertEqual(item["status"], "INCONCLUSIVE")
            self.assertEqual(item["successful_queries"], 0)
            self.assertEqual(item["missing_by_day"]["2026-10-15"]["public_dataset_query"]["status"], "ERROR")
            self.assertEqual(len(pd.read_csv(source)), len(original))


if __name__ == "__main__":
    unittest.main()
