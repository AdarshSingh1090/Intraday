import unittest
import main


class BacktestCliTests(unittest.TestCase):
    def test_research_command_is_registered(self):
        args = main.build_parser().parse_args([
            "research", "--file", "history.csv", "--symbol", "RELIANCE",
            "--train-fraction", "0.6", "--validation-fraction", "0.2",
            "--report-json", "reports/experiment.json",
        ])
        self.assertEqual(args.cmd, "research")
        self.assertEqual(args.train_fraction, 0.6)
        self.assertEqual(args.validation_fraction, 0.2)
        self.assertTrue(callable(args.func))

    def test_research_cli_prints_backtest_diagnostics(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from io import StringIO
        import contextlib

        args = SimpleNamespace(
            file="history.csv", interval=5, warmup_bars=60, budget=8900,
            max_trades_per_day=3, daily_loss_limit_pct=2.0,
            risk_per_trade_pct=0.5, max_capital_per_trade_pct=35.0,
            max_portfolio_exposure_pct=70.0, slippage_bps=5.0,
            fee_rate=0.0003, flat_fee=0.0, long_only=False,
            market_timezone="Asia/Kolkata", session_start="09:15",
            session_end="15:30", symbol="TEST", train_fraction=0.6,
            validation_fraction=0.2, report_json=None,
        )
        diagnostics = {
            "evaluation_windows": 10, "quality_blocked_windows": 4,
            "candidate_setups": 2, "risk_rejected": 1, "simulation_rejected": 1,
            "quality_blocking_reason_counts": {"gap": 4},
            "setup_reason_counts": {"RSI failed": 3},
            "risk_rejection_reason_counts": {"risk too high": 1},
            "simulation_rejection_reason_counts": {"gap invalidated": 1},
        }
        report = {
            "splits": {"train": ["a", "b"], "validation": ["c"], "test": ["d"]},
            "results": {
                "train": {"date_range": ["a", "b"], "metrics": {"trades": 0},
                          "result": {"diagnostics": diagnostics}},
            },
        }
        output = StringIO()
        with patch("main.pd.read_csv"), patch("main.run_chronological_experiment", return_value=report):
            with contextlib.redirect_stdout(output):
                main.cmd_research(args)
        text = output.getvalue()
        self.assertIn("Backtest diagnostics:", text)
        self.assertIn("quality_blocked_windows: 4", text)
        self.assertIn("setup rejection reasons: RSI failed (3)", text)
        self.assertIn("simulation rejection reasons: gap invalidated (1)", text)

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
