import tempfile
import unittest
from pathlib import Path
import pandas as pd

from market_intelligence.tradingview_tvdatafeed_probe import probe, _normalize_frame, _matches


class TradingViewProbeTests(unittest.TestCase):
    def _make_frame(self, date="2026-10-15"):
        times=["11:40","11:45","11:50","11:55"]
        rows=[]
        for i,t in enumerate(times):
            rows.append({"datetime":pd.Timestamp(f"{date} {t}:00",tz="Asia/Kolkata"),
                         "open":100+i,"high":101+i,"low":99+i,"close":100.5+i,"volume":500+i})
        return pd.DataFrame(rows).set_index("datetime")

    def test_normalize_naive_index_assumes_ist(self):
        frame=pd.DataFrame({"open":[1],"high":[2],"low":[0.5],"close":[1.5],"volume":[10]},
                          index=[pd.Timestamp("2026-10-15 11:45:00")])
        out=_normalize_frame(frame)
        self.assertEqual(str(out.iloc[0]["timestamp"].tz),"Asia/Kolkata")
        self.assertEqual(out.iloc[0]["timestamp"].hour,11)

    def test_candidate_requires_both_neighbor_matches(self):
        frame=self._make_frame()
        yahoo=frame.loc[["2026-10-15 11:40:00+05:30","2026-10-15 11:55:00+05:30"]].reset_index()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"RELIANCE.NS_5m.csv";yahoo.to_csv(path,index=False)
            result=probe(tmp,"RELIANCE.NS",Path(tmp)/"report.json",fetcher=lambda *args:frame)
        self.assertTrue(result["target_complete"])
        self.assertEqual(result["neighbor_validation"]["matches"],2)
        self.assertTrue(result["manual_review_candidate"])
        self.assertFalse(result["csv_modified"])

    def test_neighbor_mismatch_withholds_candidate(self):
        frame=self._make_frame()
        yahoo=frame.loc[["2026-10-15 11:40:00+05:30","2026-10-15 11:55:00+05:30"]].reset_index()
        yahoo.loc[0,"volume"]+=1
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"RELIANCE.NS_5m.csv";yahoo.to_csv(path,index=False)
            result=probe(tmp,"RELIANCE.NS",Path(tmp)/"report.json",fetcher=lambda *args:frame)
        self.assertTrue(result["target_complete"])
        self.assertEqual(result["neighbor_validation"]["matches"],1)
        self.assertFalse(result["manual_review_candidate"])
        self.assertEqual(result["candidate_status"],"WITHHELD_NEIGHBOR_MISMATCH")

    def test_empty_provider_result_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=probe(tmp,"RELIANCE.NS",Path(tmp)/"report.json",fetcher=lambda *args:pd.DataFrame())
        self.assertEqual(result["status"],"EMPTY_OR_UNAVAILABLE")
        self.assertEqual(result["candidate_status"],"NO_COMPLETE_TARGET_BARS")
        self.assertFalse(result["manual_review_candidate"])

    def test_exact_match_requires_volume_and_price(self):
        self.assertTrue(_matches({"open":100,"high":101,"low":99,"close":100.5,"volume":500},
                                 {"open":100.005,"high":101,"low":99,"close":100.5,"volume":500}))
        self.assertFalse(_matches({"open":100,"high":101,"low":99,"close":100.5,"volume":500},
                                  {"open":100,"high":101,"low":99,"close":100.5,"volume":501}))

if __name__=="__main__":
    unittest.main()
