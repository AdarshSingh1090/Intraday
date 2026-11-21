import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from market_intelligence.huggingface_parquet_candidate_probe import (
    _select_candidate_groups, _check_candidate_budget, _aggregate_minute_rows,
    _compare_yahoo_neighbors, MAX_ROW_DATA_BYTES, MAX_COLUMN_CHUNK_BYTES,
)


class CandidateProbeTests(unittest.TestCase):
    def test_selects_only_target_symbol_candidate_groups(self):
        report={"files":[{"filename":"0007.parquet","url":"u","size":1000,"footer_length_bytes":50,
          "metadata_parse":{"candidate_column_chunk_plan":[
            {"row_group_index":26,"symbols_possible":["RELIANCE.NS"],"columns":[],"compressed_bytes_total":10,"all_ranges_valid":True},
            {"row_group_index":72,"symbols_possible":["SBIN.NS"],"columns":[],"compressed_bytes_total":20,"all_ranges_valid":True},
          ]}}]}
        groups=_select_candidate_groups(report,"RELIANCE")
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0]["row_group_index"],26)
        self.assertEqual(groups[0]["filename"],"0007.parquet")

    def test_budget_enforces_per_chunk_and_total_caps(self):
        valid=[{"columns":[{"valid":True,"compressed_size":100}],
                "compressed_bytes_total":100,"all_ranges_valid":True}]
        self.assertEqual(_check_candidate_budget(valid)["status"],"WITHIN_BOUNDED_ROW_DATA_CAP")
        big_chunk=[{"columns":[{"valid":True,"compressed_size":MAX_COLUMN_CHUNK_BYTES+1}],
                    "compressed_bytes_total":MAX_COLUMN_CHUNK_BYTES+1,"all_ranges_valid":True}]
        self.assertEqual(_check_candidate_budget(big_chunk)["status"],"SINGLE_CHUNK_CAP_EXCEEDED")
        over=[{"columns":[{"valid":True,"compressed_size":MAX_ROW_DATA_BYTES//2+1}],
               "compressed_bytes_total":MAX_ROW_DATA_BYTES//2+1,"all_ranges_valid":True} for _ in range(2)]
        self.assertEqual(_check_candidate_budget(over)["status"],"TOTAL_ROW_DATA_CAP_EXCEEDED")
        self.assertEqual(_check_candidate_budget([])["status"],"NO_CANDIDATE_GROUPS")

    def test_aggregation_requires_exactly_five_consecutive_minutes(self):
        start=pd.Timestamp("2026-10-15T06:15:00Z")
        rows=[{"symbol":"RELIANCE","timestamp":start+pd.Timedelta(minutes=i),
               "open":100+i,"high":101+i,"low":99+i,"close":100.5+i,"volume":10+i}
              for i in range(5)]
        frame=pd.DataFrame(rows)
        bars=_aggregate_minute_rows(frame,"RELIANCE.NS",start,start+pd.Timedelta(minutes=5))
        bar=bars[start.isoformat()]
        self.assertTrue(bar["complete_five_minutes"])
        self.assertEqual(bar["rows"],5)
        self.assertEqual(bar["open"],100)
        self.assertEqual(bar["high"],105)
        self.assertEqual(bar["low"],99)
        self.assertEqual(bar["close"],104.5)
        self.assertEqual(bar["volume"],60)
        incomplete=_aggregate_minute_rows(frame.iloc[:-1],"RELIANCE.NS",start,start+pd.Timedelta(minutes=5))
        self.assertFalse(incomplete[start.isoformat()]["complete_five_minutes"])

    def test_neighbor_validation_requires_exact_volume_and_price_tolerance(self):
        buckets=[
            "2026-10-15T06:10:00+00:00","2026-10-15T06:25:00+00:00"
        ]
        source={b:{"complete_five_minutes":True,"open":100.0,"high":101.0,"low":99.0,"close":100.5,"volume":500}
                for b in buckets}
        yahoo=pd.DataFrame([{"timestamp":b,"open":100.005,"high":101.0,"low":99.0,"close":100.5,"volume":500}
                            for b in buckets])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"yahoo.csv";yahoo.to_csv(path,index=False)
            result=_compare_yahoo_neighbors(source,path,"RELIANCE.NS")
        self.assertEqual(result["matches"],2)
        self.assertEqual(result["status"],"BOTH_NEIGHBORS_MATCH")
        yahoo.loc[0,"volume"]=501
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"yahoo.csv";yahoo.to_csv(path,index=False)
            result=_compare_yahoo_neighbors(source,path,"RELIANCE.NS")
        self.assertEqual(result["matches"],1)

if __name__=="__main__":
    unittest.main()
