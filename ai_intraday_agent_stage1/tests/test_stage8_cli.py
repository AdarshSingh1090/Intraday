import unittest

from main import build_parser


class Stage8CliTests(unittest.TestCase):
    def test_fetch_research_data_parser(self):
        args = build_parser().parse_args([
            "fetch-research-data",
            "--symbols", "RELIANCE.NS,TCS.NS",
            "--interval", "5",
            "--start-date", "2026-01-01",
            "--end-date", "2026-02-01",
            "--output-dir", "data/research",
        ])
        self.assertEqual(args.cmd, "fetch-research-data")
        self.assertEqual(args.symbols, "RELIANCE.NS,TCS.NS")
        self.assertEqual(args.interval, 5)
        self.assertEqual(args.start_date, "2026-01-01")
        self.assertEqual(args.end_date, "2026-02-01")
        self.assertEqual(args.func.__name__, "cmd_fetch_research_data")


if __name__ == "__main__":
    unittest.main()
