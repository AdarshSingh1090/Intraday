import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from market_intelligence.yahoo_interval_probe import (
    aggregate_exact_5m_from_1m,
    fetch_chart_day,
    probe_alternate_yahoo_intervals,
    expected_grid,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
    def raise_for_status(self):
        pass
    def json(self):
        return self.payload


class YahooAlternateIntervalProbeTests(unittest.TestCase):
    def test_fetch_chart_parses_interval_and_ist_timestamps(self):
        stamps = [int(pd.Timestamp(f"2026-10-15 11:{m:02d}:00+05:30").timestamp()) for m in range(45, 50)]
        payload = {"chart": {"result": [{
            "timestamp": stamps,
            "indicators": {"quote": [{"open": [100, 100.1, 100.2, 100.3, 100.4],
                "high": [101, 101.1, 101.2, 101.3, 101.4], "low": [99, 99.1, 99.2, 99.3, 99.4],
                "close": [100, 100.1, 100.2, 100.3, 100.4], "volume": [100]*5}]}
        }], "error": None}}
        session = Mock()
        session.get.return_value = FakeResponse(payload)
        frame = fetch_chart_day(session, "RELIANCE.NS", pd.Timestamp("2026-10-15").date(), "1m")
        self.assertEqual(len(frame), 5)
        self.assertEqual(frame.timestamp.iloc[0].strftime("%H:%M"), "11:45")
        self.assertEqual(session.get.call_args.kwargs["params"]["interval"], "1m")

    def test_aggregates_only_exact_complete_five_minute_window(self):
        stamp = expected_grid(pd.Timestamp("2026-10-15").date())[30]
        minute_frame = pd.DataFrame({
            "timestamp": pd.date_range(stamp, periods=5, freq="1min"),
            "open": [100, 100.1, 100.2, 100.3, 100.4],
            "high": [101, 101.5, 102, 101.4, 101.6],
            "low": [99, 99.5, 98.8, 99.2, 99.4],
            "close": [100.1, 100.2, 100.3, 100.4, 100.5],
            "volume": [100, 200, 150, 250, 300],
        })
        candidate = aggregate_exact_5m_from_1m(minute_frame, stamp)
        self.assertEqual(candidate, {"open": 100.0, "high": 102.0, "low": 98.8, "close": 100.5, "volume": 1000.0})
        self.assertIsNone(aggregate_exact_5m_from_1m(minute_frame.iloc[:-1], stamp))

    def test_probe_reports_candidate_without_mutating_source(self):
        day = pd.Timestamp("2026-10-15").date()
        grid = expected_grid(day)
        missing = grid[30]
        original = pd.DataFrame({
            "timestamp": grid.delete(30).astype(str), "open": 100, "high": 101,
            "low": 99, "close": 100.5, "volume": 2000
        })
        minute_stamps = [int(pd.Timestamp(f"2026-10-15 11:{m:02d}:00+05:30").timestamp()) for m in range(45, 50)]
        minute_quote = {"open": [100]*5, "high": [101]*5, "low": [99]*5, "close": [100.5]*5, "volume": [400]*5}
        payload_1m = {"chart": {"result": [{
            "timestamp": minute_stamps, "indicators": {"quote": [minute_quote]}
        }], "error": None}}
        empty_payload = {"chart": {"result": None, "error": None}}
        session = Mock()
        def fake_get(url, params, **kwargs):
            interval = params["interval"]
            return FakeResponse(payload_1m if interval == "1m" else empty_payload)
        session.get.side_effect = fake_get
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "RELIANCE.NS_5m.csv"
            original.to_csv(source, index=False)
            out = Path(temp) / "report.json"
            report = probe_alternate_yahoo_intervals(temp, ["RELIANCE.NS"], out, session=session)
            item = report["reports"]["RELIANCE.NS"]
            self.assertEqual(item["complete_5m_candidates_from_1m"], 1)
            self.assertEqual(item["missing_by_day"]["2026-10-15"]["intervals"]["1m"]["status"], "OK")
            self.assertEqual(item["missing_by_day"]["2026-10-15"]["intervals"]["2m"]["status"], "EMPTY")
            self.assertEqual(report["summary"]["exact_5m_candidates_from_1m"], 1)
            self.assertFalse(report["datasets_modified"])
            self.assertEqual(len(pd.read_csv(source)), len(original))
            self.assertTrue(out.exists())

    def test_provider_errors_are_reported_as_inconclusive(self):
        day = pd.Timestamp("2026-10-15").date()
        grid = expected_grid(day)
        original = pd.DataFrame({
            "timestamp": grid.delete(30).astype(str), "open": 100, "high": 101,
            "low": 99, "close": 100.5, "volume": 2000
        })
        session = Mock()
        session.get.side_effect = RuntimeError("rate limited")
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "RELIANCE.NS_5m.csv"
            original.to_csv(source, index=False)
            report = probe_alternate_yahoo_intervals(temp, ["RELIANCE.NS"], Path(temp)/"report.json", session=session)
            item = report["reports"]["RELIANCE.NS"]
            self.assertEqual(item["status"], "INCONCLUSIVE")
            self.assertEqual(item["successful_interval_requests"], 0)
            self.assertTrue(all(v["status"] == "ERROR" for v in item["missing_by_day"]["2026-10-15"]["intervals"].values()))
            self.assertEqual(len(pd.read_csv(source)), len(original))


if __name__ == "__main__":
    unittest.main()
