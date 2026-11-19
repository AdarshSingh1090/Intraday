import unittest
import pandas as pd
from market_intelligence.experiment import chronological_split, walk_forward_splits, run_parameter_sensitivity


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

    def test_walk_forward_generates_ordered_folds(self):
        folds=walk_forward_splits(dataset(), min_train_days=20, validation_days=5, step_days=5)
        self.assertGreaterEqual(len(folds), 3)
        for fold in folds:
            self.assertLess(max(fold["train"].timestamp), min(fold["validation"].timestamp))

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
