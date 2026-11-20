import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from market_intelligence.provider_probe import (
    compare_frames,
    expected_session_grid,
    run_probe,
    summarize_provider_frame,
)


def frame(times):
    return pd.DataFrame({
        "Datetime": times,
        "Open": [100.0] * len(times),
        "High": [101.0] * len(times),
        "Low": [99.0] * len(times),
        "Close": [100.5] * len(times),
        "Volume": [1000] * len(times),
    })


class ProviderProbeTests(unittest.TestCase):
    def test_expected_grid_is_75_five_minute_bars(self):
        grid = expected_session_grid(pd.Timestamp("2026-11-19").date())
        self.assertEqual(len(grid), 75)
        self.assertEqual(grid[0].strftime("%H:%M"), "09:15")
        self.assertEqual(grid[-1].strftime("%H:%M"), "15:25")

    def test_missing_completed_bar_is_reported_without_repair(self):
        grid = expected_session_grid(pd.Timestamp("2026-11-19").date())
        times = grid.delete(39)  # 12:30 candle absent
        report = summarize_provider_frame(frame(times), as_of="2026-11-20 16:00:00+05:30")
        self.assertEqual(report["missing_completed_bars"], 1)
        self.assertEqual(report["missing_time_frequency"], {"12:30": 1})

    def test_provider_comparison_reports_timestamp_disagreement(self):
        grid = expected_session_grid(pd.Timestamp("2026-11-19").date())
        a, b = frame(grid.delete(39)), frame(grid)
        report = compare_frames(a, b, as_of="2026-11-20 16:00:00+05:30")
        self.assertEqual(report["timestamp_comparison"]["primary_only_timestamps"], 0)
        self.assertEqual(report["timestamp_comparison"]["alternate_only_timestamps"], 1)
        self.assertEqual(report["alternate"]["missing_completed_bars"], 0)

    def test_probe_report_discloses_same_provider_scope(self):
        grid = expected_session_grid(pd.Timestamp("2026-11-19").date())
        fake_yfinance = SimpleNamespace(
            __version__="test-version",
            Ticker=lambda symbol: SimpleNamespace(history=lambda **kwargs: frame(grid)),
            download=lambda *args, **kwargs: frame(grid),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "probe.json"
            with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
                report = run_probe(["TEST.NS"], "2026-11-19", "2026-11-20", output)

            self.assertEqual(report["yfinance_version"], "test-version")
            self.assertFalse(report["independent_provider_comparison"])
            self.assertIn("not an independent provider", report["comparison_scope"])
            self.assertIn("independent market-data provider", report["interpretation_warning"])
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
