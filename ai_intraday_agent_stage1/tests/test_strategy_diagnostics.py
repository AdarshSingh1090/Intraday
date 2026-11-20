import unittest
from unittest.mock import patch
import pandas as pd

from market_intelligence.strategy_diagnostics import diagnose_symbol, diagnose_directory, FILTERS
from market_intelligence.strategy import StrategyConfig
from market_intelligence.indicators import calculate_indicators


def make_frame(n=140, mode="up"):
    timestamps = pd.date_range("2026-11-01 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
    if mode == "up":
        close = [100+i*0.15 for i in range(n)]
    elif mode == "down":
        close = [200-i*0.15 for i in range(n)]
    else:
        close = [100+((i%4)-1.5)*0.03 for i in range(n)]
    open_ = [c-0.02 if mode != "down" else c+0.02 for c in close]
    return pd.DataFrame({"timestamp":timestamps, "open":open_, "high":[max(o,c)+0.05 for o,c in zip(open_,close)], "low":[min(o,c)-0.05 for o,c in zip(open_,close)], "close":close, "volume":[1000+i for i in range(n)]})


class StrategyDiagnosticTests(unittest.TestCase):
    def test_counts_all_filters_and_scenarios(self):
        result = diagnose_symbol(make_frame(), "TEST")
        self.assertEqual(set(result["filter_summary"]), set(FILTERS))
        self.assertIn("all_filters_pass", result["scenarios"])
        self.assertIn("remove_rsi_and_relative_volume", result["scenarios"])
        self.assertIn("strategy_baseline_hard_filters_plus_minimum_score", result["scenarios"])
        self.assertGreater(result["directional_observations"], 0)

    def test_diagnostics_fail_loudly_if_shared_slope_missing(self):
        frame = make_frame()
        original = calculate_indicators

        def without_slope(data):
            return original(data).drop(columns=["ema_20_slope_percent"])

        with patch("market_intelligence.strategy_diagnostics.calculate_indicators", side_effect=without_slope):
            with self.assertRaisesRegex(ValueError, "indicator pipeline missing required indicators.*ema_20_slope_percent"):
                diagnose_symbol(frame, "TEST")

    def test_no_auto_winner_field(self):
        result = diagnose_symbol(make_frame(), "TEST")
        self.assertNotIn("winner", result)
        self.assertNotIn("best_config", result)

    def test_long_only_disables_short_diagnostics(self):
        result = diagnose_symbol(make_frame(mode="down"), "TEST", StrategyConfig(allow_short=False))
        self.assertEqual(result["directional_observations"], 0)

    def test_directory_aggregate(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            make_frame().to_csv(Path(tmp)/"TEST.NS_5m.csv", index=False)
            result = diagnose_directory(tmp)
            self.assertIn("aggregate", result)
            self.assertIn("TEST.NS", result["symbols"])

    def test_empty_dataset_rejected(self):
        with self.assertRaises(ValueError):
            diagnose_symbol(pd.DataFrame(), "EMPTY")


    def test_strategy_baseline_requires_regime_ema_alignment(self):
        # Positive EMA slope alone is not enough to establish a bullish regime.
        # If close is below EMA20, the actual strategy classifies this row as
        # sideways and must not count it as a baseline setup, even when the
        # other five scored checks pass.
        def frame(aligned):
            close = 100.5 if aligned else 99.5
            open_price = 101.0 if aligned else 99.0
            return pd.DataFrame([{
                "timestamp": pd.Timestamp("2026-11-01 09:15", tz="Asia/Kolkata"),
                "open": open_price, "high": 101.2, "low": 98.8, "close": close,
                "volume": 1500, "ema_20": 100.0, "ema_50": 99.0,
                "rsi_14": 60.0, "atr_14": 1.0,
                "ema_20_slope_percent": 0.02, "ema20_vs_ema50_percent": 0.10,
                "relative_volume": 1.5,
            }])

        for aligned, expected in ((False, 0), (True, 1)):
            data = frame(aligned)
            with patch("market_intelligence.strategy_diagnostics.calculate_indicators", return_value=data.copy()):
                with patch("market_intelligence.strategy_diagnostics.calculate_screening_metrics", return_value={"data": data.copy()}):
                    result = diagnose_symbol(data, "TEST", StrategyConfig())
            row = result["diagnostic_rows"][0]
            self.assertEqual(row["ema_alignment"], aligned)
            self.assertEqual(row["score"], 5)
            self.assertEqual(row["baseline_strategy_pass"], bool(expected))
            self.assertEqual(result["scenarios"]["strategy_baseline_hard_filters_plus_minimum_score"]["passing_observations"], expected)
if __name__ == "__main__":
    unittest.main()
