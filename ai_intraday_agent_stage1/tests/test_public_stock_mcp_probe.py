import unittest
import pandas as pd
from market_intelligence.public_stock_mcp_probe import build_candle_args,normalize_rows,aggregate_five,_parse_response,_extract_rows

class PublicStockMCPProbeTests(unittest.TestCase):
    def test_build_args_uses_schema_and_bounded_limit(self):
        schema={"properties":{"dataset":{"enum":["stocks","index"]},"symbol":{},"interval":{"enum":["1minute","5minute"]},"start":{},"end":{},"limit":{"maximum":200},"desc":{}},
                "required":["dataset","symbol","interval","start","end"]}
        args,missing=build_candle_args(schema)
        self.assertEqual(args,{"dataset":"stocks","symbol":"RELIANCE","interval":"1minute","start":"2026-10-15","end":"2026-10-15","limit":200,"desc":False})
        self.assertEqual(missing,[])

    def test_latest_query_omits_optional_date_bounds_and_is_bounded(self):
        schema={"properties":{"dataset":{"enum":["stocks","index"]},"symbol":{},"interval":{"enum":["1minute"]},
                "start":{},"end":{},"desc":{},"limit":{"maximum":1000}},
                "required":["dataset","symbol","interval"]}
        args,missing=build_latest_candle_args(schema)
        self.assertEqual(args,{"dataset":"stocks","symbol":"RELIANCE","interval":"1minute","desc":True,"limit":3})
        self.assertEqual(missing,[])

    def test_latest_query_rejects_required_date_bounds(self):
        schema={"properties":{"symbol":{},"interval":{},"start":{},"end":{},"desc":{}},
                "required":["symbol","interval","start","end"]}
        args,missing=build_latest_candle_args(schema)
        self.assertIn("latest query requires optional date bounds",missing)

    def test_list_symbols_args_are_bounded(self):
        schema={"properties":{"dataset":{"enum":["stocks","index"]},"limit":{"maximum":50}},
                "required":["dataset"]}
        args,missing=build_list_symbols_args(schema)
        self.assertEqual(args,{"dataset":"stocks","limit":50})
        self.assertEqual(missing,[])

    def test_parse_json_and_sse(self):
        class Resp:
            headers={"content-type":"text/event-stream"}
            text='event: message\ndata: {"jsonrpc":"2.0","id":2,"result":{"ok":true}}\n\n'
            def json(self):return {"jsonrpc":"2.0","id":1}
        self.assertTrue(_parse_response(Resp())["result"]["ok"])

    def test_extract_structured_content_rows(self):
        result={"structuredContent":{"candles":[{"timestamp":"2026-10-15 11:45:00","open":1}]}}
        self.assertEqual(_extract_rows(result),result["structuredContent"]["candles"])

    def test_epoch_rows_convert_to_ist(self):
        t=pd.Timestamp("2026-10-15 11:45",tz="Asia/Kolkata").tz_convert("UTC").timestamp()
        rows=[{"timestamp":int(t*1000),"open":100,"high":101,"low":99,"close":100.5,"volume":10}]
        df=normalize_rows(rows)
        self.assertEqual(str(df.iloc[0].timestamp),"2026-10-15 11:45:00+05:30")

    def test_timestamp_seconds_floor_to_minute(self):
        row={"timestamp":"2026-10-15 11:45:59+05:30","open":100,"high":101,"low":99,"close":100.5,"volume":10}
        df=normalize_rows([row]);self.assertEqual(str(df.iloc[0].timestamp),"2026-10-15 11:45:00+05:30")

    def test_aggregation_requires_five_consecutive_minutes(self):
        rows=[]
        for i in range(5):
            t=pd.Timestamp("2026-10-15 11:45",tz="Asia/Kolkata")+pd.Timedelta(minutes=i)
            rows.append({"timestamp":t,"open":100+i,"high":102+i,"low":99+i,"close":101+i,"volume":10})
        df=pd.DataFrame(rows);agg=aggregate_five(df,pd.Timestamp("2026-10-15 11:45",tz="Asia/Kolkata"))
        self.assertEqual(agg["open"],100);self.assertEqual(agg["close"],105);self.assertEqual(agg["high"],106);self.assertEqual(agg["low"],99);self.assertEqual(agg["volume"],50)
        self.assertIsNone(aggregate_five(df.iloc[:-1],pd.Timestamp("2026-10-15 11:45",tz="Asia/Kolkata")))

if __name__=="__main__":unittest.main()
