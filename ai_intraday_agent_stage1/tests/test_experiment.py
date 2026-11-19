import unittest
import pandas as pd
from market_intelligence.experiment import chronological_split, walk_forward_splits, run_parameter_sensitivity, run_walk_forward_backtests, run_multi_symbol_backtests


def dataset(days=40):
    rows=[]
    for day in pd.date_range("2026-01-01", periods=days, freq="B"):
        rows.append({"timestamp": day + pd.Timedelta(hours=9, minutes=15), "value": 1})
    return pd.DataFrame(rows)


class ExperimentTests(unittest.TestCase):
    def test_chronological_split_has_no_date_overlap(self):
        split=chronological_split(dataset())
        self.assertEqual(split["method"], "chronological_by_trading_date")
        train=set(split["train"].timestamp.dt.date)
        val=set(split["validation"].timestamp.dt.date)
        test=set(split["test"].timestamp.dt.date)
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertLess(max(split["train"].timestamp), min(split["validation"].timestamp))
        self.assertLess(max(split["validation"].timestamp), min(split["test"].timestamp))
        self.assertEqual(min(split["validation_with_warmup"].timestamp), min(split["train"].timestamp))
        self.assertEqual(max(split["validation_with_warmup"].timestamp), max(split["validation"].timestamp))
        self.assertEqual(max(split["test_with_warmup"].timestamp), max(split["test"].timestamp))

    def test_walk_forward_generates_ordered_folds(self):
        folds=walk_forward_splits(dataset(), min_train_days=20, validation_days=5, step_days=5)
        self.assertGreaterEqual(len(folds), 3)
        for fold in folds:
            self.assertLess(max(fold["train"].timestamp), min(fold["validation"].timestamp))
            self.assertEqual(max(fold["validation_with_warmup"].timestamp), max(fold["validation"].timestamp))

    def test_walk_forward_runner_passes_period_bounds(self):
        class Config:
            def __init__(self, start, end): self.start_date=start; self.end_date=end
        calls=[]
        def fake_backtest(data, symbol, config):
            calls.append((config.start_date,config.end_date,len(data)))
            return {"status":"COMPLETED","metrics":{"trades":0},"trades":[]}
        result=run_walk_forward_backtests(dataset(30),fake_backtest,
            lambda start,end: Config(start,end),symbol="TEST",
            min_train_days=10,validation_days=5,step_days=5)
        self.assertGreaterEqual(len(result["folds"]),3)
        self.assertTrue(all(call[2] > 5 for call in calls))

    def test_multi_symbol_runner(self):
        def fake(data,symbol,config):
            return {"status":"COMPLETED","trades":[{"net_pnl":1,"gross_pnl":1,"estimated_fees":0,"symbol":symbol}]}
        result=run_multi_symbol_backtests({"AAA":dataset(5),"BBB":dataset(5)},fake,lambda s:None)
        self.assertEqual(len(result["symbols"]),2)
        self.assertEqual(len(result["combined_trades"]),2)
        self.assertIn("not a portfolio",result["warning"])

    def test_invalid_split_rejected(self):
        with self.assertRaises(ValueError):
            chronological_split(dataset(2))

    def test_sensitivity_runner_does_not_select_winner(self):
        class Config:
            def __init__(self, x): self.x=x
        configs=[Config(1),Config(2)]
        def fake(data, config):
            return {"status":"COMPLETED","metrics":{"net_pnl":config.x*10}}
        result=run_parameter_sensitivity(dataset(),configs,fake,metric_keys=["net_pnl"])
        self.assertEqual(len(result),2)
        self.assertEqual(result.net_pnl.tolist(),[10,20])


if __name__ == "__main__":
    unittest.main()
