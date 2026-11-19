import unittest
import main


class BacktestCliTests(unittest.TestCase):
    def test_backtest_command_is_registered(self):
        args = main.build_parser().parse_args([
            "backtest", "--file", "history.csv", "--symbol", "RELIANCE",
            "--interval", "5", "--start-date", "2026-01-01", "--end-date", "2026-01-31",
            "--market-timezone", "Asia/Kolkata", "--session-start", "09:15", "--session-end", "15:30",
            "--report-json", "reports/run.json",
            "--trades-csv", "reports/trades.csv",
        ])
        self.assertEqual(args.cmd, "backtest")
        self.assertEqual(args.symbol, "RELIANCE")
        self.assertEqual(args.interval, 5)
        self.assertEqual(args.start_date, "2026-01-01")
        self.assertEqual(args.end_date, "2026-01-31")
        self.assertEqual(args.market_timezone, "Asia/Kolkata")
        self.assertEqual(args.session_start, "09:15")
        self.assertEqual(args.report_json, "reports/run.json")
        self.assertTrue(callable(args.func))


if __name__ == "__main__":
    unittest.main()
