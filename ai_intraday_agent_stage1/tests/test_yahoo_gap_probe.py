import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from market_intelligence.yahoo_gap_probe import fetch_chart_day, probe_missing_yahoo_bars, expected_grid


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
    def raise_for_status(self):
        pass
    def json(self):
        return self.payload


class YahooGapProbeTests(unittest.TestCase):
    def test_fetch_chart_parses_candle_timezone_and_values(self):
        stamp = int(pd.Timestamp("2026-10-15 11:45:00+05:30").timestamp())
        payload = {"chart": {"result": [{
            "timestamp": [stamp],
            "indicators": {"quote": [{"open": [100], "high": [101], "low": [99], "close": [100.5], "volume": [2000]}]}
        }], "error": None}}
        session = Mock()
        session.get.return_value = FakeResponse(payload)
        result = fetch_chart_day(session, "RELIANCE.NS", pd.Timestamp("2026-10-15").date(),
                                 "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}")
        self.assertEqual(result.iloc[0]["timestamp"].strftime("%Y-%m-%d %H:%M"), "2026-10-15 11:45")
        self.assertEqual(result.iloc[0]["volume"], 2000)
        self.assertEqual(session.get.call_args.kwargs["params"]["interval"], "5m")

    def test_probe_reports_recovered_candidate_without_mutating_dataset(self):
        day = pd.Timestamp("2026-10-15").date()
        grid = expected_grid(day)
        missing = grid[30]
        original = pd.DataFrame({
            "timestamp": grid.delete(30).astype(str), "open": 100, "high": 101,
            "low": 99, "close": 100.5, "volume": 2000
        })
        payload = {"chart": {"result": [{
            "timestamp": [int(missing.timestamp())],
            "indicators": {"quote": [{"open": [100], "high": [101], "low": [99], "close": [100.5], "volume": [2000]}]}
        }], "error": None}}
        session = Mock()
        session.get.return_value = FakeResponse(payload)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "RELIANCE.NS_5m.csv"
            original.to_csv(source, index=False)
            out = Path(temp) / "report.json"
            report = probe_missing_yahoo_bars(temp, ["RELIANCE.NS"], out, session=session)
            item = report["reports"]["RELIANCE.NS"]
            self.assertEqual(item["missing_count_original"], 1)
            self.assertEqual(item["missing_candles_found_by_endpoint"]["query1.finance.yahoo.com"], 1)
            self.assertEqual(item["missing_candles_found_by_endpoint"]["query2.finance.yahoo.com"], 1)
            self.assertFalse(report["datasets_modified"])
            self.assertEqual(len(pd.read_csv(source)), len(original))
            self.assertTrue(out.exists())
            self.assertIn("not independent", report["interpretation"].lower())


if __name__ == "__main__":
    unittest.main()
