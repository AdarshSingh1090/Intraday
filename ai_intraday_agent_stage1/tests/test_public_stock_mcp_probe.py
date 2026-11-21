import unittest
import json
import pandas as pd
from market_intelligence.public_stock_mcp_probe import build_candle_args,build_latest_candle_args,build_list_symbols_args,summarize_tool_result,normalize_rows,aggregate_five,_parse_response,_extract_rows

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

    def test_summarize_parses_each_text_item_as_symbol_metadata(self):
        result={"content":[
            {"type":"text","text":json.dumps({"symbol":"ABB","rows":1031866,"start":"2015-02-02T09:15:00+05:30","end":"2026-04-08T12:56:00+05:30"})},
            {"type":"text","text":json.dumps({"symbol":"RELIANCE","rows":1200,"start":"2026-01-01","end":"2026-04-08"})},
            {"type":"text","text":json.dumps({"symbol":"RELIANCE","rows":1200,"start":"2026-01-01","end":"2026-04-08"})},
        ]}
        summary=summarize_tool_result(result)
        self.assertEqual(summary["text_content_item_count"],3)
        self.assertEqual(summary["decoded_text_payload_count"],3)
        self.assertEqual(summary["symbol_record_count"],3)
        self.assertEqual(summary["symbol_count"],2)
        self.assertEqual(summary["sample_symbols"],["ABB","RELIANCE"])
        self.assertTrue(summary["contains_reliance"])
        self.assertEqual(summary["reported_rows_total"],1034266)
        self.assertEqual(summary["reported_end_max"],"2026-04-08")
        self.assertEqual(summary["target_symbol_metadata"][0]["end"],"2026-04-08")
        self.assertFalse(summary["target_symbol_metadata"][0]["target_date_within_reported_range"])
        later=summarize_tool_result(result,target_date="2026-04-08")
        self.assertTrue(later["target_symbol_metadata"][0]["target_date_within_reported_range"])

    def test_summarize_supports_structured_symbol_list_and_deduplicates(self):
        result={"structuredContent":{"symbols":[{"ticker":"RELIANCE.NS"},{"ticker":"TCS"},{"ticker":"RELIANCE.NS"}]},
                "content":[{"type":"text","text":"ignored fallback"}]}
        summary=summarize_tool_result(result)
        self.assertEqual(summary["symbol_count"],2)
        self.assertEqual(summary["sample_symbols"],["RELIANCE.NS","TCS"])
        self.assertEqual(summary["decoded_text_payload_count"],0)

    def test_summarize_falls_back_when_structured_content_has_no_symbol_records(self):
        result={"structuredContent":{"dataset":"stocks","synced":True},
                "content":[
                    {"type":"text","text":json.dumps({"symbol":"RELIANCE","rows":100,"start":"2025-01-01","end":"2026-04-08"})},
                    {"type":"text","text":json.dumps({"symbol":"TCS","rows":200,"start":"2025-01-01","end":"2026-04-08"})}
                ]}
        summary=summarize_tool_result(result)
        self.assertEqual(summary["symbol_count"],2)
        self.assertTrue(summary["contains_reliance"])
        self.assertEqual(summary["target_symbol_metadata"][0]["rows"],100)

if __name__=="__main__":unittest.main()
