import unittest
import main


class BacktestCliTests(unittest.TestCase):
    def test_backtest_command_is_registered(self):
        args = main.build_parser().parse_args([
            "backtest", "--file", "history.csv", "--symbol", "RELIANCE",
            "--interval", "5", "--report-json", "reports/run.json",
            "--trades-csv", "reports/trades.csv",
        ])
        self.assertEqual(args.cmd, "backtest")
        self.assertEqual(args.symbol, "RELIANCE")
        self.assertEqual(args.interval, 5)
        self.assertEqual(args.report_json, "reports/run.json")
        self.assertTrue(callable(args.func))


if __name__ == "__main__":
    unittest.main()
