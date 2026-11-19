import unittest
import pandas as pd

from market_intelligence.strategy import generate_setup, StrategyConfig


def make_evaluation(direction="LONG", score_all=True):
    n = 60
    close = [100 + i * 0.2 for i in range(n)]
    open_ = [x - 0.05 for x in close]
    if direction == "SHORT":
        close = [200 - i * 0.2 for i in range(n)]
        open_ = [x + 0.05 for x in close]
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-11-20 09:15", periods=n, freq="5min", tz="Asia/Kolkata"),
        "open": open_, "high": [max(o,c)+0.1 for o,c in zip(open_,close)],
        "low": [min(o,c)-0.1 for o,c in zip(open_,close)], "close": close,
        "volume": [1000]*n
    })
    # Indicators intentionally constructed for deterministic strategy tests.
    if direction == "LONG":
        df["ema_20"] = [x - 0.5 for x in close]
        df["ema_50"] = [x - 1.0 for x in close]
        df["rsi_14"] = [60.0]*n
        df["atr_14"] = [0.5]*n
        df["atr_percent"] = [0.25]*n
        df["ema_20_slope_percent"] = [0.1]*n
        regime = {"trend":"bullish","trend_strength":"strong","momentum":"bullish","volatility":"normal"}
    else:
        df["ema_20"] = [x + 0.5 for x in close]
        df["ema_50"] = [x + 1.0 for x in close]
        df["rsi_14"] = [40.0]*n
        df["atr_14"] = [0.5]*n
        df["atr_percent"] = [0.25]*n
        df["ema_20_slope_percent"] = [-0.1]*n
        regime = {"trend":"bearish","trend_strength":"strong","momentum":"bearish","volatility":"normal"}
    return {
        "trade_eligible": True, "blocking_reasons": [],
        "technical_analysis": {"regime":regime,"indicators":df},
        "screening": {"metrics":{"relative_volume":1.5}},
        "symbol":"TEST"
    }


class StrategyTests(unittest.TestCase):
    def test_valid_long_candidate(self):
        result=generate_setup(make_evaluation("LONG"))
        self.assertEqual(result["signal"],"LONG")
        self.assertEqual(result["status"],"CANDIDATE")
        self.assertEqual(result["candidate"]["reward_risk_ratio"],2.0)
        self.assertGreater(result["candidate"]["entry_reference_price"],result["candidate"]["stop_loss"])

    def test_valid_short_candidate(self):
        result=generate_setup(make_evaluation("SHORT"))
        self.assertEqual(result["signal"],"SHORT")
        self.assertLess(result["candidate"]["entry_reference_price"],result["candidate"]["stop_loss"])

    def test_quality_gate_blocks(self):
        evaluation=make_evaluation()
        evaluation["trade_eligible"]=False
        evaluation["blocking_reasons"]=["stale"]
        result=generate_setup(evaluation)
        self.assertEqual(result["status"],"BLOCKED")
        self.assertEqual(result["signal"],"NO_TRADE")

    def test_rsi_outside_range_no_setup(self):
        evaluation=make_evaluation()
        evaluation["technical_analysis"]["indicators"].loc[59,"rsi_14"]=75
        result=generate_setup(evaluation)
        self.assertEqual(result["status"],"NO_SETUP")

    def test_relative_volume_required(self):
        evaluation=make_evaluation()
        evaluation["screening"]["metrics"]["relative_volume"]=0.8
        result=generate_setup(evaluation)
        self.assertEqual(result["status"],"NO_SETUP")

    def test_insufficient_indicators(self):
        evaluation=make_evaluation()
        evaluation["technical_analysis"]["indicators"].loc[59,"atr_14"]=float("nan")
        result=generate_setup(evaluation)
        self.assertEqual(result["status"],"INSUFFICIENT_DATA")

    def test_no_short_when_disabled(self):
        result=generate_setup(make_evaluation("SHORT"),config=StrategyConfig(allow_short=False))
        self.assertEqual(result["signal"],"NO_TRADE")


if __name__ == "__main__":
    unittest.main()
