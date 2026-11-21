import tempfile
import unittest
from pathlib import Path
import pandas as pd
from market_intelligence.nse_epoch_probe import normalize, probe

class NSEEpochProbeTests(unittest.TestCase):
    def make_rows(self, day, times):
        return [{"time": int(pd.Timestamp(f"{day} {hm}", tz="Asia/Kolkata").tz_convert("UTC").timestamp()*1000),
                 "open":100.0,"high":101.0,"low":99.0,"close":100.5,"volume":1000}
                for hm in times]

    def test_epoch_ms_parsed_as_utc_then_converted_to_ist(self):
        raw={"payload":{"status":"ok","data":self.make_rows("2026-10-15",["11:45"])}}
        f=normalize(raw)
        self.assertEqual(str(f.iloc[0].timestamp),"2026-10-15 11:45:00+05:30")

    def test_target_requires_both_valid_bars_and_exact_neighbor_matches(self):
        target=self.make_rows("2026-10-15",["11:40","11:45","11:50","11:55"])
        sanity=self.make_rows("2026-11-19",["09:15","12:00","15:10"])
        def fetch(symbol,start,end,interval):
            return {"payload":{"status":"ok","data":sanity if start.date()==pd.Timestamp("2026-11-19").date() else target}}
        with tempfile.TemporaryDirectory() as d:
            rows=[]
            for day,hm in [("2026-10-15",x) for x in ("11:40","11:55")]+[("2026-11-19",x) for x in ("09:15","12:00","15:10")]:
                ts=pd.Timestamp(f"{day} {hm}",tz="Asia/Kolkata")
                rows.append({"timestamp":ts.isoformat(),"open":100,"high":101,"low":99,"close":100.5,"volume":1000})
            csv=Path(d)/"RELIANCE.NS_5m.csv";pd.DataFrame(rows).to_csv(csv,index=False)
            result=probe(d,Path(d)/"report.json",raw_fetcher=fetch,delay=0)
            self.assertEqual(result["provider_health"],"RECENT_SANITY_DATA_RETURNED")
            self.assertEqual(result["target_query"]["candidate_status"],"CANDIDATE_FOR_MANUAL_REVIEW")
            self.assertFalse(result["datasets_modified"])
            self.assertEqual(len(pd.read_csv(csv)),5)

    def test_empty_recent_sanity_is_inconclusive(self):
        def fetch(*args): return {"payload":{"status":"ok","data":[]}}
        with tempfile.TemporaryDirectory() as d:
            result=probe(d,Path(d)/"report.json",raw_fetcher=fetch,delay=0)
            self.assertEqual(result["provider_health"],"RECENT_SANITY_EMPTY_OR_UNAVAILABLE")
            self.assertEqual(result["target_query"]["candidate_status"],"TARGET_EMPTY_OR_UNAVAILABLE")
            self.assertEqual(result["manual_review_candidates"],0)

    def test_neighbor_mismatch_withholds_candidate(self):
        target=self.make_rows("2026-10-15",["11:40","11:45","11:50","11:55"]);target[-1]["close"]=101.0
        sanity=self.make_rows("2026-11-19",["09:15","12:00","15:10"])
        def fetch(symbol,start,end,interval):
            return {"payload":{"status":"ok","data":sanity if start.date()==pd.Timestamp("2026-11-19").date() else target}}
        with tempfile.TemporaryDirectory() as d:
            rows=[]
            for day,hm in [("2026-10-15",x) for x in ("11:40","11:55")]+[("2026-11-19",x) for x in ("09:15","12:00","15:10")]:
                ts=pd.Timestamp(f"{day} {hm}",tz="Asia/Kolkata")
                rows.append({"timestamp":ts.isoformat(),"open":100,"high":101,"low":99,"close":100.5,"volume":1000})
            pd.DataFrame(rows).to_csv(Path(d)/"RELIANCE.NS_5m.csv",index=False)
            result=probe(d,Path(d)/"report.json",raw_fetcher=fetch,delay=0)
            self.assertEqual(result["target_query"]["candidate_status"],"WITHHELD_VALIDATION_FAILED")
            self.assertEqual(result["target_query"]["neighbor_validation"]["matches"],1)

if __name__=="__main__": unittest.main()
