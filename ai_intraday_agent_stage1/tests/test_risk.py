import unittest
from market_intelligence.risk import assess_candidate, RiskConfig


def candidate(direction="LONG", entry=100, stop=98, target=104):
    return {"symbol":"TEST","direction":direction,"entry_reference_price":entry,
            "stop_loss":stop,"target":target}


class RiskTests(unittest.TestCase):
    def test_sizes_with_risk_and_capital_constraints(self):
        result=assess_candidate(candidate(), RiskConfig(budget=8900))
        self.assertTrue(result["approved"])
        self.assertEqual(result["quantity"], 22) # ₹44.50 risk cap / ₹2 risk per share
        self.assertLessEqual(result["total_risk"], result["risk_budget"])
        self.assertLessEqual(result["capital_required"], 8900*0.35)

    def test_short_candidate(self):
        result=assess_candidate(candidate("SHORT",100,102,96))
        self.assertTrue(result["approved"])
        self.assertEqual(result["direction"], "SHORT")

    def test_wrong_stop_target_geometry_rejected(self):
        result=assess_candidate(candidate("LONG",100,102,104))
        self.assertFalse(result["approved"])

    def test_daily_loss_limit_halts(self):
        result=assess_candidate(candidate(), daily_pnl=-178)
        self.assertFalse(result["approved"])
        self.assertIn("Daily loss limit reached; paper trading halted for the day.", result["reasons"])

    def test_trade_count_limit(self):
        result=assess_candidate(candidate(), trades_today=3)
        self.assertFalse(result["approved"])

    def test_exposure_limit_restricts_quantity(self):
        result=assess_candidate(candidate(), open_exposure=6000)
        self.assertTrue(result["approved"])
        self.assertLessEqual(result["capital_required"], 8900*0.70-6000)

    def test_zero_quantity_rejected(self):
        result=assess_candidate(candidate(entry=100000,stop=99998,target=100004))
        self.assertFalse(result["approved"])
        self.assertEqual(result["quantity"], 0)

    def test_missing_candidate_rejected(self):
        result=assess_candidate(None)
        self.assertFalse(result["approved"])

    def test_invalid_direction_rejected(self):
        result=assess_candidate({"direction":"HOLD","entry_reference_price":100,"stop_loss":98,"target":104})
        self.assertFalse(result["approved"])

    def test_reward_risk_below_minimum_rejected(self):
        result=assess_candidate(candidate(target=102.5))
        self.assertFalse(result["approved"])
        self.assertIn("Reward/risk below minimum 2.0:1.", result["reasons"])


if __name__ == "__main__":
    unittest.main()
