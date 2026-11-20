import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_intelligence.research_data import (
    ResearchFetchConfig, normalize_yahoo_frame, audit_research_candles,
    fetch_multi_symbol_research_data,
)


def sample_frame(n=3):
    return pd.DataFrame({
        "Datetime": pd.date_range("2026-01-05 09:15", periods=n, freq="5min"),
        "Open": [100.0] * n, "High": [101.0] * n,
        "Low": [99.0] * n, "Close": [100.5] * n, "Volume": [1000] * n,
    })


class ResearchDataTests(unittest.TestCase):
    def test_normalization_timezone_and_columns(self):
        frame = normalize_yahoo_frame(sample_frame())
        self.assertEqual(list(frame.columns), ["timestamp", "open", "high", "low", "close", "volume"])
        self.assertEqual(str(frame.timestamp.dt.tz), "Asia/Kolkata")
        self.assertEqual(len(frame), 3)

    def test_audit_passes_valid_dataset(self):
        report = audit_research_candles(normalize_yahoo_frame(sample_frame()))
        self.assertTrue(report["eligible"])
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["coverage"]["trading_dates"], 1)

    def test_audit_flags_duplicate_and_ohlc(self):
        frame = normalize_yahoo_frame(sample_frame())
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
        frame.loc[2, "high"] = 98
        report = audit_research_candles(frame)
        self.assertFalse(report["eligible"])
        self.assertTrue(any("duplicate" in issue for issue in report["issues"]))
        self.assertTrue(any("inconsistent OHLC" in issue for issue in report["issues"]))

    def test_fetch_isolated_per_symbol_and_manifest_saved(self):
        calls = []
        def fake_fetch(ticker, interval, start, end):
            calls.append((ticker, interval, start, end))
            if ticker == "BAD.NS":
                raise RuntimeError("provider failure")
            return sample_frame()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = fetch_multi_symbol_research_data(
                ["GOOD.NS", "BAD.NS"],
                ResearchFetchConfig(5, "2026-01-01", "2026-02-01"),
                tmp, fetcher=fake_fetch,
            )
            self.assertEqual(manifest["successful_symbols"], ["GOOD.NS"])
            self.assertEqual(manifest["failed_or_rejected_symbols"], ["BAD.NS"])
            self.assertTrue(Path(tmp, "GOOD.NS_5m.csv").exists())
            self.assertTrue(Path(tmp, "manifest.json").exists())
            self.assertEqual(calls[0][1], "5m")

    def test_invalid_interval_rejected_before_network(self):
        with self.assertRaises(ValueError):
            fetch_multi_symbol_research_data(
                ["ABC.NS"], ResearchFetchConfig(3, "2026-01-01", "2026-02-01"), "."
            )

    def test_empty_data_is_recorded_as_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = fetch_multi_symbol_research_data(
                ["EMPTY.NS"], ResearchFetchConfig(5, "2026-01-01", "2026-02-01"),
                tmp, fetcher=lambda *args: pd.DataFrame(),
            )
            self.assertEqual(result["results"]["EMPTY.NS"]["status"], "REJECTED")


if __name__ == "__main__":
    unittest.main()
