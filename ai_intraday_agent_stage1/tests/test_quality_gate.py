import unittest
import pandas as pd

from market_intelligence.quality_gate import assess_candles, QualityConfig


def sample():
    timestamps = pd.date_range("2026-11-20 09:15", periods=25, freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame({
        "timestamp": timestamps, "open": [100.0] * 25, "high": [101.0] * 25,
        "low": [99.0] * 25, "close": [100.5] * 25, "volume": [1000.0] * 25,
    })


class QualityGateTests(unittest.TestCase):
    def test_clean_recent_data_is_eligible(self):
        df = sample()
        result = assess_candles(df, now=df.timestamp.iloc[-1] + pd.Timedelta(minutes=1))
        self.assertTrue(result["trade_eligible"])
        self.assertEqual(result["status"], "PASSED")
        self.assertEqual(result["quality_score"], 100.0)

    def test_zero_volume_latest_candle_blocks(self):
        df = sample()
        df.loc[df.index[-1], "volume"] = 0
        result = assess_candles(df, now=df.timestamp.iloc[-1] + pd.Timedelta(minutes=1))
        self.assertFalse(result["trade_eligible"])
        self.assertIn("ZERO_VOLUME", result["latest"]["flags"])

    def test_flat_low_volume_candle_flagged(self):
        df = sample()
        df.loc[df.index[-1], ["open", "high", "low", "close", "volume"]] = [100, 100, 100, 100, 100]
        result = assess_candles(df, now=df.timestamp.iloc[-1] + pd.Timedelta(minutes=1))
        self.assertIn("FLAT_LOW_VOLUME", result["latest"]["flags"])
        self.assertFalse(result["trade_eligible"])

    def test_flat_high_volume_candle_flagged(self):
        df = sample()
        df.loc[df.index[-1], ["open", "high", "low", "close", "volume"]] = [100, 100, 100, 100, 5000]
        result = assess_candles(df, now=df.timestamp.iloc[-1] + pd.Timedelta(minutes=1))
        self.assertIn("FLAT_HIGH_VOLUME", result["latest"]["flags"])
        self.assertFalse(result["trade_eligible"])

    def test_missing_interval_blocks(self):
        df = sample().drop(index=10).reset_index(drop=True)
        result = assess_candles(df, now=df.timestamp.iloc[-1] + pd.Timedelta(minutes=1))
        self.assertFalse(result["trade_eligible"])
        self.assertEqual(result["summary"]["missing_candles_within_observed_ranges"], 1)

    def test_stale_data_blocks(self):
        df = sample()
        result = assess_candles(df, now=df.timestamp.iloc[-1] + pd.Timedelta(minutes=20),
                                config=QualityConfig(freshness_minutes=10))
        self.assertFalse(result["trade_eligible"])
        self.assertTrue(any("stale" in x.lower() for x in result["blocking_reasons"]))

    def test_duplicate_timestamps_flagged(self):
        df = sample()
        df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
        result = assess_candles(df, now=df.timestamp.iloc[-1] + pd.Timedelta(minutes=1))
        self.assertGreater(result["summary"]["duplicate_timestamps"], 0)

    def test_invalid_ohlc_blocks_latest(self):
        df = sample()
        df.loc[df.index[-1], "high"] = 98
        result = assess_candles(df, now=df.timestamp.iloc[-1] + pd.Timedelta(minutes=1))
        self.assertFalse(result["trade_eligible"])
        self.assertIn("INVALID_OHLC", result["latest"]["flags"])

    def test_input_is_not_mutated(self):
        df = sample()
        before = df.copy(deep=True)
        assess_candles(df, now=df.timestamp.iloc[-1] + pd.Timedelta(minutes=1))
        pd.testing.assert_frame_equal(df, before)

    def test_empty_data_returns_safe_result(self):
        result = assess_candles(pd.DataFrame())
        self.assertFalse(result["trade_eligible"])
        self.assertEqual(result["status"], "no_data")


if ( __name__ == "__main__" ):
    unittest.main()
