import unittest
from market_intelligence.research_gate import evaluate_research_gate


class ResearchGateTests(unittest.TestCase):
    def setUp(self):
        self.audit = {"eligible": True, "issues": [], "symbols": {}}
        self.manifest = {
            "symbols_requested": ["A.NS", "B.NS"],
            "results": {
                "A.NS": {"status": "SAVED", "audit": {"eligible": True, "issues": []}},
                "B.NS": {"status": "SAVED", "audit": {"eligible": True, "issues": []}},
            },
        }

    def test_eligible_only_when_both_audits_pass(self):
        report = evaluate_research_gate(self.audit, self.manifest)
        self.assertTrue(report["eligible"])
        self.assertEqual(report["status"], "ELIGIBLE")

    def test_rejected_acquisition_blocks_even_if_completeness_passes(self):
        self.manifest["results"]["B.NS"] = {"status": "REJECTED", "audit": {"eligible": False, "issues": ["bad OHLC"]}}
        report = evaluate_research_gate(self.audit, self.manifest)
        self.assertFalse(report["eligible"])
        self.assertEqual(report["status"], "BLOCKED_ACQUISITION")
        self.assertIn("B.NS: acquisition status 'REJECTED'", report["acquisition_issues"])

    def test_missing_requested_symbol_result_blocks(self):
        del self.manifest["results"]["B.NS"]
        report = evaluate_research_gate(self.audit, self.manifest)
        self.assertFalse(report["eligible"])
        self.assertTrue(any("B.NS: missing acquisition result" in issue for issue in report["issues"]))

    def test_missing_manifest_blocks(self):
        report = evaluate_research_gate(self.audit, None)
        self.assertFalse(report["eligible"])
        self.assertEqual(report["status"], "BLOCKED_ACQUISITION")

    def test_both_audits_fail(self):
        self.audit["eligible"] = False
        self.audit["issues"] = ["missing bars"]
        self.manifest["results"]["B.NS"]["status"] = "ERROR"
        report = evaluate_research_gate(self.audit, self.manifest)
        self.assertEqual(report["status"], "BLOCKED_DATA_QUALITY_AND_ACQUISITION")
        self.assertFalse(report["eligible"])


if __name__ == "__main__":
    unittest.main()
