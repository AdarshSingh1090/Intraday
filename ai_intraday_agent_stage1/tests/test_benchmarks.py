import unittest
import pandas as pd
from market_intelligence.benchmarks import benchmark_summary


class BenchmarkTests(unittest.TestCase):
    def test_baselines_returned_and_costs_applied(self):
        df=pd.DataFrame({
            "timestamp":pd.date_range("2026-01-01 09:15",periods=4,freq="5min"),
            "open":[100,101,102,103],"close":[101,102,103,104],
            "high":[101,102,103,104],"low":[100,101,102,103]
        })
        result=benchmark_summary(df,1000,slippage_bps=0,fee_rate=0,flat_fee=0)
        self.assertEqual(result["status"],"COMPLETED")
        self.assertEqual(result["benchmarks"]["no_trade"]["net_pnl"],0)
        self.assertGreater(result["benchmarks"]["buy_and_hold"]["net_pnl"],0)
        self.assertEqual(len(result["benchmarks"]["daily_open_close"]["daily_results"]),1)

    def test_empty_data_safe(self):
        result=benchmark_summary(pd.DataFrame())
        self.assertEqual(result["status"],"NO_DATA")


if __name__=="__main__":
    unittest.main()
