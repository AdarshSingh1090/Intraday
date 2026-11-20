import unittest
import tempfile
from pathlib import Path
import pandas as pd

from market_intelligence.signal_diagnostics import diagnose_file


class SignalDiagnosticsTests(unittest.TestCase):
    def test_records_no_setup_and_insufficient_reasons_separately(self):
        n = 100
        close = [100 + i * 0.1 + (i % 5) * 0.01 for i in range(n)]
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-11-20 09:15", periods=n, freq="5min", tz="Asia/Kolkata"),
            "open": [p - 0.02 for p in close],
            "high": [p + 0.1 for p in close],
            "low": [p - 0.1 for p in close],
            "close": close,
            "volume": [1000] * n,
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            df.to_csv(path, index=False)
            result = diagnose_file(path, "TEST", warmup=60)
        self.assertEqual(result["status_counts"].get("INSUFFICIENT_DATA", 0), 0)
        self.assertTrue(result["status_counts"].get("NO_SETUP", 0) > 0)
        self.assertEqual(result["insufficient_data_reasons"], {})
        self.assertTrue(result["no_setup_reasons"])
        self.assertEqual(result["execution"], "none")


if __name__ == "__main__":
    unittest.main()
