import tempfile, unittest
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
from market_intelligence.nse_chart_probe import normalize, probe

class NSEChartProbeTests(unittest.TestCase):
    def frame(self,rows):
        return pd.DataFrame([{"datetime":ts,"open":o,"high":h,"low":l,"close":c,"volume":v} for ts,o,h,l,c,v in rows])
    def test_normalize_naive_timestamps_as_ist(self):
        f=normalize(self.frame([("2026-10-15 11:45:00",100,101,99,100.5,1000)]))
        self.assertEqual(str(f.iloc[0].timestamp),"2026-10-15 11:45:00+05:30")
    def test_candidate_requires_complete_targets_and_matching_neighbors(self):
        day="2026-10-15"; rows=[(f"{day} 11:40:00",100,101,99,100.5,1000),(f"{day} 11:45:00",100.5,102,100,101.5,1500),(f"{day} 11:50:00",101.5,103,101,102.5,1600),(f"{day} 11:55:00",102.5,104,102,103.5,1700)]
        fetch=Mock(return_value=self.frame(rows))
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/"RELIANCE.NS_5m.csv"
            pd.DataFrame([{"timestamp":f"{day} 11:40:00+05:30","open":100,"high":101,"low":99,"close":100.5,"volume":1000},{"timestamp":f"{day} 11:55:00+05:30","open":102.5,"high":104,"low":102,"close":103.5,"volume":1700}]).to_csv(src,index=False)
            r=probe(d,["RELIANCE.NS"],Path(d)/"report.json",fetcher=fetch,delay=0)["reports"]["RELIANCE.NS"]
            self.assertEqual(r["candidate_status"],"CANDIDATES_FOR_MANUAL_REVIEW")
            self.assertEqual(r["target_bars"]["2026-10-15 11:45:00+05:30"]["close"],101.5)
            self.assertTrue(r["neighbor_validation"]["all_match"])
            self.assertFalse(r["csv_modified"]);self.assertEqual(len(pd.read_csv(src)),2)
    def test_withholds_when_missing_target_or_neighbor_mismatch(self):
        day="2026-10-15";fetch=Mock(return_value=self.frame([(f"{day} 11:40:00",101,102,100,101.5,1000),(f"{day} 11:45:00",100.5,102,100,101.5,1500)]))
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/"RELIANCE.NS_5m.csv"
            pd.DataFrame([{"timestamp":f"{day} 11:40:00+05:30","open":100,"high":101,"low":99,"close":100.5,"volume":1000},{"timestamp":f"{day} 11:55:00+05:30","open":102.5,"high":104,"low":102,"close":103.5,"volume":1700}]).to_csv(src,index=False)
            r=probe(d,["RELIANCE.NS"],Path(d)/"report.json",fetcher=fetch,delay=0)["reports"]["RELIANCE.NS"]
            self.assertEqual(r["candidate_status"],"NO_COMPLETE_TARGET_BARS");self.assertFalse(r["target_complete"])
            self.assertFalse(r["neighbor_validation"]["all_match"]);self.assertFalse(r["csv_modified"])
if __name__=="__main__": unittest.main()
