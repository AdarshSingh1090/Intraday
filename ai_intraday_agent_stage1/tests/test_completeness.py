import tempfile
import unittest
from pathlib import Path
import pandas as pd

from market_intelligence.completeness import (
    audit_intraday_completeness, audit_research_directory,
)


class CompletenessAuditTests(unittest.TestCase):
    def full_session(self):
        ts = pd.date_range("2026-01-05 09:15", "2026-01-05 15:25", freq="5min", tz="Asia/Kolkata")
        return pd.DataFrame({"timestamp": ts, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})

    def test_full_nse_session_75_bars_passes(self):
        report = audit_intraday_completeness(self.full_session(), 5)
        self.assertTrue(report["eligible"])
        self.assertEqual(report["expected_bars_on_observed_dates"], 75)
        self.assertEqual(report["missing_bars"], 0)

    def test_missing_bar_is_detected(self):
        frame = self.full_session().drop(index=20).reset_index(drop=True)
        report = audit_intraday_completeness(frame, 5)
        self.assertFalse(report["eligible"])
        self.assertEqual(report["missing_bars"], 1)
        self.assertTrue(any("missing bars: 1" in x for x in report["issues"]))

    def test_out_of_session_and_alignment_are_detected(self):
        frame = self.full_session()
        frame.loc[0, "timestamp"] = pd.Timestamp("2026-01-05 09:17", tz="Asia/Kolkata")
        frame.loc[1, "timestamp"] = pd.Timestamp("2026-01-05 08:55", tz="Asia/Kolkata")
        report = audit_intraday_completeness(frame, 5)
        self.assertFalse(report["eligible"])
        self.assertGreater(report["outside_session_bars"], 0)
        self.assertGreater(report["misaligned_bars"], 0)

    def test_cross_symbol_timestamp_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.full_session()
            b = a.drop(index=10).reset_index(drop=True)
            a.to_csv(Path(tmp, "A.csv"), index=False)
            b.to_csv(Path(tmp, "B.csv"), index=False)
            report = audit_research_directory(tmp, 5, "A")
            self.assertFalse(report["eligible"])
            self.assertEqual(report["synchronization"]["B"]["missing_reference_timestamps"], 1)
            self.assertTrue(Path(tmp, "completeness_audit.json").exists())

    def test_empty_directory_is_ineligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = audit_research_directory(tmp)
            self.assertFalse(report["eligible"])
            self.assertIn("no CSV datasets found", report["issues"])


if __name__ == "__main__":
    unittest.main()
