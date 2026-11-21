import unittest
import pandas as pd

from market_intelligence.evaluation import evaluate_market_data


def sample():
    ts = pd.date_range("2026-11-20 09:15", periods=60, freq="5min", tz="Asia/Kolkata")
    close = [100 + i * 0.1 for i in range(60)]
    return pd.DataFrame({
        "timestamp": ts,
        "open": [x - 0.02 for x in close],
        "high": [x + 0.05 for x in close],
        "low": [x - 0.05 for x in close],
        "close": close,
        "volume": [1000.0] * 60,
    })


class IntegratedEvaluationTests(unittest.TestCase):
    def test_clean_data_ready_for_strategy(self):
        df = sample()
        result = evaluate_market_data(df, now=df.timestamp.iloc[-1] + pd.Timedelta(minutes=1))
        self.assertTrue(result["trade_eligible"])
        self.assertEqual(result["status"], "READY_FOR_STRATEGY")
        self.assertIsNotNone(result["technical_analysis"])
        self.assertIsNotNone(result["screening"])

    def test_quality_failure_blocks_strategy_eligibility(self):
        df = sample()
        df.loc[df.index[-1], "volume"] = 0
        result = evaluate_market_data(df, now=df.timestamp.iloc[-1] + pd.Timedelta(minutes=1))
        self.assertFalse(result["trade_eligible"])
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(result["blocking_reasons"])

    def test_empty_data_safe(self):
        result = evaluate_market_data(pd.DataFrame())
        self.assertFalse(result["trade_eligible"])
        self.assertEqual(result["status"], "no_data")


if __name__ == "__main__":
    unittest.main()
