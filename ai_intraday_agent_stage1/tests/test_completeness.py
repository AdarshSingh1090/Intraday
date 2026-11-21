import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from market_intelligence.completeness import (
    audit_intraday_completeness, audit_research_directory,
)
from market_intelligence.session_calendar import audit_whole_sessions


class CompletenessAuditTests(unittest.TestCase):
    def full_session(self, date="2026-01-05"):
        ts = pd.date_range(f"{date} 09:15", f"{date} 15:25", freq="5min", tz="Asia/Kolkata")
        return pd.DataFrame({"timestamp": ts, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})

    def test_full_nse_session_75_bars_passes(self):
        report = audit_intraday_completeness(self.full_session(), 5, as_of="2026-01-05 16:00+05:30")
        self.assertTrue(report["eligible"])
        self.assertEqual(report["expected_bars_on_observed_dates"], 75)
        self.assertEqual(report["missing_bars"], 0)
        self.assertFalse(report["per_date"]["2026-01-05"]["partial_session"])

    def test_invalid_timestamp_row_blocks_completeness(self):
        frame = self.full_session()
        frame["timestamp"] = frame["timestamp"].astype(object)
        frame.loc[20, "timestamp"] = "not-a-timestamp"
        report = audit_intraday_completeness(frame, 5, as_of="2026-01-05 16:00+05:30")
        self.assertFalse(report["eligible"])
        self.assertEqual(report["invalid_timestamps"], 1)
        self.assertEqual(report["missing_bars"], 1)
        self.assertTrue(any("invalid timestamps: 1" in issue for issue in report["issues"]))

    def test_directory_audit_propagates_invalid_timestamp_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self.full_session()
            frame["timestamp"] = frame["timestamp"].astype(object)
            frame.loc[20, "timestamp"] = "not-a-timestamp"
            frame.to_csv(Path(tmp, "A.csv"), index=False)
            report = audit_research_directory(tmp, 5, "A", as_of="2026-01-05 16:00+05:30")
            self.assertFalse(report["eligible"])
            self.assertEqual(report["symbols"]["A"]["completeness"]["invalid_timestamps"], 1)
            self.assertTrue(any("invalid timestamps" in issue for issue in report["issues"]))

    def test_missing_bar_in_completed_session_is_detected(self):
        frame = self.full_session().drop(index=20).reset_index(drop=True)
        report = audit_intraday_completeness(frame, 5, as_of="2026-01-05 16:00+05:30")
        self.assertFalse(report["eligible"])
        self.assertEqual(report["missing_bars"], 1)
        self.assertTrue(any("missing completed bars: 1" in x for x in report["issues"]))

    def test_partial_current_session_does_not_count_future_session_slots_as_missing(self):
        frame = self.full_session().iloc[:53].reset_index(drop=True)  # through 13:35; forming candle
        report = audit_intraday_completeness(frame, 5, as_of="2026-01-05 13:37+05:30")
        day = report["per_date"]["2026-01-05"]
        self.assertTrue(report["eligible"])
        self.assertTrue(day["partial_session"])
        self.assertEqual(day["full_session_expected_bars"], 75)
        self.assertEqual(day["expected_bars"], 52)  # through 13:30, completed candle starts
        self.assertEqual(day["observed_completed_bars"], 52)
        self.assertEqual(day["uncompleted_bars_excluded"], 1)
        self.assertEqual(day["missing_bars"], 0)

    def test_missing_completed_bar_before_partial_cutoff_is_detected(self):
        frame = self.full_session().iloc[:53].drop(index=51).reset_index(drop=True)
        report = audit_intraday_completeness(frame, 5, as_of="2026-01-05 13:37+05:30")
        self.assertFalse(report["eligible"])
        self.assertEqual(report["missing_bars"], 1)

    def test_out_of_session_and_alignment_are_detected(self):
        frame = self.full_session()
        frame.loc[0, "timestamp"] = pd.Timestamp("2026-01-05 09:17", tz="Asia/Kolkata")
        frame.loc[1, "timestamp"] = pd.Timestamp("2026-01-05 08:55", tz="Asia/Kolkata")
        report = audit_intraday_completeness(frame, 5, as_of="2026-01-05 16:00+05:30")
        self.assertFalse(report["eligible"])
        self.assertGreater(report["outside_session_bars"], 0)
        self.assertGreater(report["misaligned_bars"], 0)

    def test_cross_symbol_timestamp_mismatch_on_completed_bars(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.full_session()
            b = a.drop(index=10).reset_index(drop=True)
            a.to_csv(Path(tmp, "A.csv"), index=False)
            b.to_csv(Path(tmp, "B.csv"), index=False)
            report = audit_research_directory(tmp, 5, "A", as_of="2026-01-05 16:00+05:30")
            self.assertFalse(report["eligible"])
            self.assertEqual(report["synchronization"]["B"]["missing_reference_timestamps"], 1)
            self.assertTrue(any("B: completeness audit failed" in issue for issue in report["issues"]))
            self.assertTrue(any("B: timestamp synchronization mismatch" in issue for issue in report["issues"]))
            self.assertTrue(Path(tmp, "completeness_audit.json").exists())

    def test_sync_ignores_unfinished_bar_for_current_partial_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.full_session().iloc[:53].reset_index(drop=True)  # 13:35 unfinished
            b = self.full_session().iloc[:52].reset_index(drop=True)  # through 13:30
            a.to_csv(Path(tmp, "A.csv"), index=False)
            b.to_csv(Path(tmp, "B.csv"), index=False)
            report = audit_research_directory(tmp, 5, "A", as_of="2026-01-05 13:37+05:30")
            self.assertEqual(report["synchronization"]["B"]["missing_reference_timestamps"], 0)
            self.assertEqual(report["synchronization"]["B"]["extra_timestamps_vs_reference"], 0)

    def test_manifest_research_eligibility_reflects_completeness_and_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.full_session()
            b = a.drop(index=10).reset_index(drop=True)
            a.to_csv(Path(tmp, "A.csv"), index=False)
            b.to_csv(Path(tmp, "B.csv"), index=False)
            manifest = {
                "results": {
                    "A": {"status": "SAVED", "audit": {"eligible": True}},
                    "B": {"status": "SAVED", "audit": {"eligible": True}},
                }
            }
            Path(tmp, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = audit_research_directory(tmp, 5, "A", as_of="2026-01-05 16:00+05:30")
            updated = json.loads(Path(tmp, "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(updated["completeness_audit"]["eligible"])
            self.assertEqual(updated["completeness_audit"]["audited_symbols"], 2)
            self.assertTrue(updated["results"]["A"]["research_eligibility"]["eligible"])
            self.assertFalse(updated["results"]["B"]["research_eligibility"]["eligible"])
            self.assertEqual(updated["results"]["B"]["research_eligibility"]["completeness"]["missing_bars"], 1)
            self.assertTrue(Path(report["manifest_updated"]).exists())

    def cas_session(self, date="2026-01-05"):
        continuous = pd.date_range(f"{date} 09:15", f"{date} 15:10", freq="5min", tz="Asia/Kolkata")
        post = pd.date_range(f"{date} 15:15", periods=3, freq="5min", tz="Asia/Kolkata")
        ts = continuous.append(post)
        return pd.DataFrame({"timestamp": ts, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})

    def test_cas_session_excludes_auction_rows_and_uses_72_continuous_bars(self):
        report = audit_intraday_completeness(self.cas_session(), 5, session_end="15:15", as_of="2026-01-05 16:00+05:30")
        self.assertTrue(report["eligible"])
        self.assertEqual(report["expected_bars_on_observed_dates"], 72)
        self.assertEqual(report["missing_bars"], 0)
        self.assertEqual(report["post_session_observations_excluded"], 3)
        self.assertEqual(report["per_date"]["2026-01-05"]["post_session_bars_excluded"], 3)

    def test_cross_symbol_sync_ignores_post_session_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.cas_session()
            b = a.iloc[:72].reset_index(drop=True)
            a.to_csv(Path(tmp, "A.csv"), index=False)
            b.to_csv(Path(tmp, "B.csv"), index=False)
            report = audit_research_directory(tmp, 5, "A", session_end="15:15", as_of="2026-01-05 16:00+05:30")
            self.assertTrue(report["eligible"])
            self.assertEqual(report["synchronization"]["B"]["missing_reference_timestamps"], 0)
            self.assertEqual(report["synchronization"]["B"]["extra_timestamps_vs_reference"], 0)


    def test_missing_whole_session_across_all_symbols_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = pd.concat([self.full_session("2026-01-05"), self.full_session("2026-01-07")], ignore_index=True)
            b = a.copy()
            a.to_csv(Path(tmp, "A.csv"), index=False)
            b.to_csv(Path(tmp, "B.csv"), index=False)
            report = audit_research_directory(tmp, 5, "A", as_of="2026-01-08 16:00+05:30")
            calendar = report["whole_session_calendar_audit"]
            self.assertFalse(report["eligible"])
            self.assertEqual(calendar["missing_sessions_union"], ["2026-01-06"])
            self.assertEqual(calendar["per_symbol"]["A"]["missing_sessions"], ["2026-01-06"])
            self.assertEqual(calendar["per_symbol"]["B"]["missing_sessions"], ["2026-01-06"])
            self.assertTrue(any("missing full trading sessions" in issue for issue in report["issues"]))

    def test_nse_holiday_is_not_counted_as_missing_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = pd.concat([self.full_session("2026-01-23"), self.full_session("2026-01-27")], ignore_index=True)
            frame.to_csv(Path(tmp, "A.csv"), index=False)
            report = audit_research_directory(tmp, 5, "A", as_of="2026-01-28 16:00+05:30")
            calendar = report["whole_session_calendar_audit"]
            self.assertTrue(report["eligible"])
            self.assertEqual(calendar["missing_sessions_union"], [])
            self.assertIn("2026-01-26", calendar["holidays_excluded"])

    def test_manifest_requested_range_catches_trailing_missing_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = pd.concat([self.full_session("2026-01-05"), self.full_session("2026-01-07")], ignore_index=True)
            frame.to_csv(Path(tmp, "A.NS_5m.csv"), index=False)
            manifest = {
                "start_date": "2026-01-05",
                "end_date": "2026-01-08",
                "results": {"A.NS": {"status": "SAVED", "audit": {"eligible": True}}},
            }
            Path(tmp, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = audit_research_directory(tmp, 5, "A.NS_5m", as_of="2026-01-08 16:00+05:30")
            updated = json.loads(Path(tmp, "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["whole_session_calendar_audit"]["missing_sessions_union"], ["2026-01-06"])
            self.assertFalse(updated["results"]["A.NS"]["research_eligibility"]["eligible"])
            self.assertFalse(updated["completeness_audit"]["eligible"])

    def test_unknown_calendar_year_blocks_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self.full_session("2027-01-04")
            frame.to_csv(Path(tmp, "A.csv"), index=False)
            report = audit_research_directory(tmp, 5, "A", as_of="2027-01-05 16:00+05:30")
            calendar = report["whole_session_calendar_audit"]
            self.assertFalse(report["eligible"])
            self.assertEqual(calendar["unsupported_years"], [2027])
            self.assertTrue(any("official session calendar unavailable" in issue for issue in report["issues"]))

    def test_special_weekend_sessions_are_excluded_from_regular_session_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = pd.concat([self.full_session("2026-11-06"), self.full_session("2026-11-09")], ignore_index=True)
            frame.to_csv(Path(tmp, "A.csv"), index=False)
            report = audit_research_directory(tmp, 5, "A", as_of="2026-11-10 16:00+05:30")
            calendar = report["whole_session_calendar_audit"]
            self.assertTrue(report["eligible"])
            self.assertIn("2026-11-08", calendar["special_sessions_excluded"])
            self.assertNotIn("2026-11-08", calendar["expected_sessions"])

    def test_current_session_is_pending_until_regular_session_close(self):
        report = audit_whole_sessions(
            {"A": {pd.Timestamp("2026-01-02").date()}},
            start_date="2026-01-02",
            end_date_exclusive="2026-01-06",
            as_of="2026-01-05 12:00+05:30",
            session_end="15:15",
        )
        self.assertTrue(report["eligible"])
        self.assertEqual(report["missing_sessions_union"], [])
        self.assertEqual(report["pending_current_session"], ["2026-01-05"])
        self.assertEqual(report["per_symbol"]["A"]["observed_sessions_in_range"], 1)

    def test_empty_directory_is_ineligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = audit_research_directory(tmp, as_of="2026-01-05 16:00+05:30")
            self.assertFalse(report["eligible"])
            self.assertIn("no CSV datasets found", report["issues"])


if __name__ == "__main__":
    unittest.main()
