import unittest
import pandas as pd
from unittest.mock import patch

from market_intelligence.backtester import backtest, calculate_metrics, BacktestConfig


def candles(n=70):
    ts = pd.date_range("2026-01-05 09:15", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamp": ts,
        "open": [100.0]*n, "high": [101.0]*n,
        "low": [99.0]*n, "close": [100.0]*n,
        "volume": [1000]*n,
    })


class BacktesterTests(unittest.TestCase):
    def test_empty_input_rejected(self):
        result=backtest(pd.DataFrame())
        self.assertEqual(result["status"],"REJECTED")

    def test_missing_columns_rejected(self):
        result=backtest(pd.DataFrame({"timestamp":["2026-01-01"]}))
        self.assertEqual(result["status"],"REJECTED")

    def test_invalid_configuration_rejected(self):
        result=backtest(candles(65),config=BacktestConfig(budget="not-a-number"))
        self.assertEqual(result["status"],"REJECTED")
        self.assertIn("Invalid backtest configuration",result["reason"])

    def test_insufficient_history_rejected(self):
        result=backtest(candles(30),config=BacktestConfig(warmup_bars=60))
        self.assertEqual(result["status"],"REJECTED")

    def test_metrics_calculated(self):
        trades=[
            {"net_pnl":100,"gross_pnl":110,"estimated_fees":10,"entry_timestamp":"2026-01-01 09:20"},
            {"net_pnl":-50,"gross_pnl":-45,"estimated_fees":5,"entry_timestamp":"2026-01-01 09:30"},
            {"net_pnl":25,"gross_pnl":30,"estimated_fees":5,"entry_timestamp":"2026-01-02 09:20"},
        ]
        m=calculate_metrics(trades)
        self.assertEqual(m["trades"],3)
        self.assertEqual(m["win_rate_pct"],66.67)
        self.assertEqual(m["net_pnl"],75)
        self.assertEqual(m["profit_factor"],2.5)
        self.assertEqual(m["max_drawdown"],50)
        self.assertEqual(m["trades_per_day"],1.5)
        self.assertEqual(m["expectancy_per_trade"],25)
        self.assertEqual(m["ending_equity"],8975)
        self.assertEqual(m["total_return_pct"],0.8427)
        self.assertIsNotNone(m["expectancy_bootstrap_95_ci"])

    def test_duplicate_timestamps_are_rejected_not_silently_deduplicated(self):
        data=candles(65)
        data.loc[1, "timestamp"] = data.loc[0, "timestamp"]
        result=backtest(data,config=BacktestConfig(warmup_bars=60))
        self.assertEqual(result["status"],"REJECTED")
        self.assertIn("duplicate timestamps",result["reason"])

    def test_inconsistent_ohlc_is_rejected(self):
        data=candles(65)
        data.loc[2, "high"] = 98
        result=backtest(data,config=BacktestConfig(warmup_bars=60))
        self.assertEqual(result["status"],"REJECTED")
        self.assertIn("inconsistent OHLC",result["reason"])

    def test_regular_session_filter_and_timezone_normalization(self):
        data=candles(65)
        data.loc[len(data)] = [pd.Timestamp("2026-01-05 08:30"),100,101,99,100,1000]
        data.loc[len(data)] = [pd.Timestamp("2026-01-05 15:40"),100,101,99,100,1000]
        result=backtest(data,config=BacktestConfig(warmup_bars=60))
        self.assertEqual(result["status"],"COMPLETED")
        self.assertEqual(result["excluded_out_of_session_bars"],2)
        self.assertEqual(result["market_timezone"],"Asia/Kolkata")

    def test_date_bounds_exclude_out_of_period_signals(self):
        data=candles(65)
        result=backtest(data,config=BacktestConfig(warmup_bars=60,start_date="2026-01-06",end_date="2026-01-06"))
        self.assertEqual(result["status"],"COMPLETED")
        self.assertEqual(result["metrics"]["trades"],0)

    def test_completed_result_contains_benchmarks(self):
        data=candles(65)
        with patch("market_intelligence.backtester.evaluate_market_data", return_value={"trade_eligible":False,"blocking_reasons":["test"]}):
            result=backtest(data,config=BacktestConfig(warmup_bars=60))
        self.assertEqual(result["benchmarks"]["status"],"COMPLETED")
        self.assertIn("buy_and_hold",result["benchmarks"]["benchmarks"])

    def test_walk_forward_uses_only_future_bars_for_fill(self):
        data=candles(65)
        fake_candidate={"symbol":"TEST","direction":"LONG","signal_timestamp":str(data.iloc[60].timestamp),
                       "entry_reference_price":100,"stop_loss":98,"target":104}
        fake_risk={"approved":True,"quantity":1,"risk_budget":100}
        fake_sim={"status":"SIMULATED","symbol":"TEST","direction":"LONG","quantity":1,
                  "signal_timestamp":fake_candidate["signal_timestamp"],"entry_timestamp":str(data.iloc[61].timestamp),
                  "exit_timestamp":str(data.iloc[64].timestamp),"gross_pnl":1,"estimated_fees":0,"net_pnl":1}
        with patch("market_intelligence.backtester.evaluate_market_data", return_value={"trade_eligible":True,"blocking_reasons":[]}), \
             patch("market_intelligence.backtester.generate_setup", return_value={"status":"CANDIDATE","candidate":fake_candidate}), \
             patch("market_intelligence.backtester.assess_candidate", return_value=fake_risk), \
             patch("market_intelligence.backtester.simulate_candidate", return_value=fake_sim) as sim:
            result=backtest(data,config=BacktestConfig(warmup_bars=60))
            args=sim.call_args_list[0].args
            forward=args[1]
            self.assertGreater(pd.to_datetime(forward.iloc[0].timestamp),pd.to_datetime(fake_candidate["signal_timestamp"]))
            self.assertEqual(pd.to_datetime(forward.iloc[0].timestamp),pd.to_datetime(data.iloc[61].timestamp))
        self.assertEqual(result["status"],"COMPLETED")
        self.assertEqual(result["metrics"]["trades"],1)
        self.assertTrue(result["paper_only"])


if __name__ == "__main__":
    unittest.main()
