import unittest
import pandas as pd
from market_intelligence.paper_simulator import simulate_candidate, SimulationConfig


def c(direction="LONG"):
    return {"symbol":"TEST","direction":direction,"entry_reference_price":100,
            "stop_loss":98 if direction=="LONG" else 102,
            "target":104 if direction=="LONG" else 96,"quantity":10,
            "signal_timestamp":"2026-01-01 09:15"}


def bars(rows):
    return pd.DataFrame(rows,columns=["timestamp","open","high","low","close","volume"])


class PaperSimulatorTests(unittest.TestCase):
    def test_long_target_hit(self):
        result=simulate_candidate(c(),bars([["09:20",100,105,99,104,1000]]),SimulationConfig(0,0,0))
        self.assertEqual(result["exit_reason"],"TARGET")
        self.assertEqual(result["net_pnl"],40)

    def test_long_stop_hit(self):
        result=simulate_candidate(c(),bars([["09:20",100,101,97,99,1000]]),SimulationConfig(0,0,0))
        self.assertEqual(result["exit_reason"],"STOP")
        self.assertEqual(result["net_pnl"],-20)

    def test_both_hit_same_bar_stop_first(self):
        result=simulate_candidate(c(),bars([["09:20",100,105,97,103,1000]]),SimulationConfig(0,0,0))
        self.assertIn("STOP",result["exit_reason"])
        self.assertEqual(result["net_pnl"],-20)

    def test_stop_gap(self):
        result=simulate_candidate(c(),bars([["09:20",97,98,96,97,1000]]),SimulationConfig(0,0,0))
        self.assertEqual(result["exit_reason"],"STOP_GAP")
        self.assertEqual(result["net_pnl"],-30)

    def test_signal_candle_cannot_be_used_as_forward_data(self):
        result=simulate_candidate(c(),bars([["2026-01-01 09:15",100,105,99,104,1000]]),SimulationConfig(0,0,0))
        self.assertEqual(result["status"],"REJECTED")
        self.assertIn("strictly after",result["reason"])

    def test_entry_gap_past_target_is_rejected(self):
        result=simulate_candidate(c(),bars([["09:20",105,106,104,105,1000]]),SimulationConfig(0,0,0))
        self.assertEqual(result["status"],"REJECTED")

    def test_actual_entry_risk_cannot_exceed_approved_budget(self):
        candidate=c()
        candidate["risk_budget"]=20
        result=simulate_candidate(candidate,bars([["09:20",101,102,100,101,1000]]),SimulationConfig(0,0,0))
        self.assertEqual(result["status"],"REJECTED")
        self.assertIn("actual entry risk exceeds",result["reason"])

    def test_end_of_data_exit(self):
        result=simulate_candidate(c(),bars([["09:20",100,101,99,103,1000],["09:25",103,103,102,103,1000]]),SimulationConfig(0,0,0))
        self.assertEqual(result["exit_reason"],"END_OF_DATA")
        self.assertEqual(result["net_pnl"],30)

    def test_short_target(self):
        result=simulate_candidate(c("SHORT"),bars([["09:20",100,97,95,96,1000]]),SimulationConfig(0,0,0))
        self.assertEqual(result["exit_reason"],"TARGET")
        self.assertEqual(result["net_pnl"],40)

    def test_slippage_and_fees_reduce_profit(self):
        result=simulate_candidate(c(),bars([["09:20",100,105,99,104,1000]]))
        self.assertLess(result["net_pnl"],result["gross_pnl"])

    def test_invalid_candidate(self):
        result=simulate_candidate({},bars([["09:20",100,101,99,100,1000]]))
        self.assertEqual(result["status"],"REJECTED")

    def test_no_bars(self):
        result=simulate_candidate(c(),pd.DataFrame())
        self.assertEqual(result["status"],"REJECTED")


if __name__=="__main__":
    unittest.main()
