import unittest
from market_intelligence.performance import evaluate_performance, segment_performance


class PerformanceTests(unittest.TestCase):
    def test_equity_drawdown_and_return(self):
        trades = [
            {"net_pnl": 100, "gross_pnl": 110, "estimated_fees": 10, "entry_timestamp": "2026-01-01", "exit_timestamp": "2026-01-01"},
            {"net_pnl": -200, "gross_pnl": -190, "estimated_fees": 10, "entry_timestamp": "2026-01-02", "exit_timestamp": "2026-01-02"},
            {"net_pnl": 50, "gross_pnl": 55, "estimated_fees": 5, "entry_timestamp": "2026-01-03", "exit_timestamp": "2026-01-03"},
        ]
        m = evaluate_performance(trades, 1000, bootstrap_iterations=200)
        self.assertEqual(m["net_pnl"], -50)
        self.assertEqual(m["ending_equity"], 950)
        self.assertEqual(m["max_drawdown"], 200)
        self.assertAlmostEqual(m["max_drawdown_pct"], 18.1818, places=3)
        self.assertEqual(m["trades"], 3)
        self.assertIsNotNone(m["expectancy_bootstrap_95_ci"])

    def test_segment_attribution(self):
        trades=[
            {"net_pnl":10,"gross_pnl":12,"estimated_fees":2,"symbol":"AAA","direction":"LONG","exit_reason":"TARGET"},
            {"net_pnl":-5,"gross_pnl":-3,"estimated_fees":2,"symbol":"AAA","direction":"SHORT","exit_reason":"STOP"},
            {"net_pnl":20,"gross_pnl":22,"estimated_fees":2,"symbol":"BBB","direction":"LONG","exit_reason":"TARGET"},
        ]
        segments=segment_performance(trades,group_by=("symbol","direction"),starting_equity=1000,bootstrap_iterations=100)
        self.assertEqual(segments["symbol"]["AAA"]["trades"],2)
        self.assertEqual(segments["direction"]["LONG"]["trades"],2)
        self.assertEqual(segments["direction"]["SHORT"]["net_pnl"],-5)

    def test_metrics_sort_trades_chronologically(self):
        trades=[
            {"net_pnl":100,"gross_pnl":100,"estimated_fees":0,"entry_timestamp":"2026-01-02 09:20"},
            {"net_pnl":-50,"gross_pnl":-50,"estimated_fees":0,"entry_timestamp":"2026-01-01 09:20"},
        ]
        m=evaluate_performance(trades,1000,bootstrap_iterations=0)
        self.assertEqual(m["max_drawdown"],50)
        self.assertEqual(list(m["daily_net_pnl"]),["2026-01-01","2026-01-02"])

    def test_empty_trades(self):
        m = evaluate_performance([], 8900)
        self.assertEqual(m["trades"], 0)
        self.assertEqual(m["net_pnl"], 0)
        self.assertEqual(m["ending_equity"], 8900)

    def test_invalid_starting_equity(self):
        with self.assertRaises(ValueError):
            evaluate_performance([], 0)

    def test_non_finite_pnl_is_ignored_as_zero(self):
        m = evaluate_performance([{"net_pnl": float("nan")}], 1000)
        self.assertEqual(m["net_pnl"], 0)


if __name__ == "__main__":
    unittest.main()
