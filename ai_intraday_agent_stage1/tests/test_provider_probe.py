import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from market_intelligence.provider_probe import (
    compare_frames, expected_session_grid, run_independent_probe, run_probe,
    summarize_provider_frame,
)


def frame(times):
    return pd.DataFrame({"Datetime": times, "Open": [100.0]*len(times), "High": [101.0]*len(times),
        "Low": [99.0]*len(times), "Close": [100.5]*len(times), "Volume": [1000]*len(times)})


class FakeResponse:
    def __init__(self, content=b"", payload=None, status=200): self.content=content; self.payload=payload; self.status=status
    def raise_for_status(self):
        if self.status >= 400: raise RuntimeError(f"HTTP {self.status}")
    def json(self): return self.payload


class FakeSession:
    def __init__(self, instruments, candles): self.instruments=instruments; self.candles=candles; self.calls=[]
    def get(self, url, headers=None, timeout=None):
        self.calls.append((url,headers,timeout))
        if url.endswith("NSE.json.gz"): return FakeResponse(content=gzip.compress(json.dumps(self.instruments).encode()))
        return FakeResponse(payload={"data":{"candles":self.candles}})


class ProviderProbeTests(unittest.TestCase):
    def test_expected_grid_is_75_five_minute_bars(self):
        grid=expected_session_grid(pd.Timestamp("2026-11-19").date())
        self.assertEqual(len(grid),75); self.assertEqual(grid[0].strftime("%H:%M"),"09:15"); self.assertEqual(grid[-1].strftime("%H:%M"),"15:25")

    def test_cas_expected_grid_is_72_continuous_bars(self):
        grid = expected_session_grid(pd.Timestamp("2026-11-19").date(), session_end="15:15")
        self.assertEqual(len(grid), 72)
        self.assertEqual(grid[0].strftime("%H:%M"), "09:15")
        self.assertEqual(grid[-1].strftime("%H:%M"), "15:10")

    def test_cas_probe_reports_post_session_rows_separately(self):
        continuous = expected_session_grid(pd.Timestamp("2026-11-19").date(), session_end="15:15")
        post = pd.date_range("2026-11-19 15:15", periods=3, freq="5min", tz="Asia/Kolkata")
        report = summarize_provider_frame(frame(continuous.append(post)), as_of="2026-11-20 16:00+05:30", session_end="15:15")
        self.assertEqual(report["missing_completed_bars"], 0)
        self.assertEqual(report["continuous_session_rows"], 72)
        self.assertEqual(report["post_session_observations_excluded"], 3)
        self.assertEqual(report["post_session_time_frequency"], {"15:15": 1, "15:20": 1, "15:25": 1})

    def test_cas_interface_comparison_ignores_post_session_timestamp_differences(self):
        continuous = expected_session_grid(pd.Timestamp("2026-11-19").date(), session_end="15:15")
        post = pd.date_range("2026-11-19 15:15", periods=3, freq="5min", tz="Asia/Kolkata")
        report = compare_frames(frame(continuous.append(post)), frame(continuous), as_of="2026-11-20 16:00+05:30", session_end="15:15")
        self.assertEqual(report["timestamp_comparison"]["primary_only_timestamps"], 0)
        self.assertEqual(report["timestamp_comparison"]["matched_timestamps"], 72)
        self.assertEqual(report["raw_timestamp_comparison"]["primary_only_timestamps"], 3)


    def test_missing_completed_bar_is_reported_without_repair(self):
        grid=expected_session_grid(pd.Timestamp("2026-11-19").date()); report=summarize_provider_frame(frame(grid.delete(39)),as_of="2026-11-20 16:00:00+05:30")
        self.assertEqual(report["missing_completed_bars"],1); self.assertEqual(report["missing_time_frequency"],{"12:30":1})

    def test_provider_comparison_reports_timestamp_disagreement(self):
        grid=expected_session_grid(pd.Timestamp("2026-11-19").date()); a,b=frame(grid.delete(39)),frame(grid)
        report=compare_frames(a,b,as_of="2026-11-20 16:00:00+05:30")
        self.assertEqual(report["timestamp_comparison"]["primary_only_timestamps"],0); self.assertEqual(report["timestamp_comparison"]["alternate_only_timestamps"],1)
        self.assertEqual(report["alternate"]["missing_completed_bars"],0)

    def test_probe_report_discloses_same_provider_scope(self):
        grid=expected_session_grid(pd.Timestamp("2026-11-19").date())
        fake_yfinance=SimpleNamespace(__version__="test-version",Ticker=lambda symbol:SimpleNamespace(history=lambda **kwargs:frame(grid)),download=lambda *args,**kwargs:frame(grid))
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"probe.json"
            with patch.dict(sys.modules,{"yfinance":fake_yfinance}): report=run_probe(["TEST.NS"],"2026-11-19","2026-11-20",out)
            self.assertEqual(report["yfinance_version"],"test-version"); self.assertFalse(report["independent_provider_comparison"])
            self.assertTrue(out.exists())

    def test_independent_probe_skips_without_token(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"skip.json"; r=run_independent_probe(["RELIANCE.NS"],"2026-11-19","2026-11-20",d,out)
            self.assertEqual(r["status"],"SKIPPED_NO_TOKEN"); self.assertTrue(out.exists()); self.assertNotIn("secret-token",out.read_text())

    def test_upstox_probe_compares_missing_timestamp_and_auth_header(self):
        grid=expected_session_grid(pd.Timestamp("2026-11-19").date()); yahoo=frame(grid.delete(39))
        instruments=[{"exchange":"NSE","segment":"NSE_EQ","instrument_type":"EQ","trading_symbol":"RELIANCE","instrument_key":"NSE_EQ|INE002A01018"}]
        candles=[[str(t),100,101,99,100.5,1000,0] for t in grid]; session=FakeSession(instruments,candles)
        with tempfile.TemporaryDirectory() as d:
            Path(d,"RELIANCE.NS_5m.csv").write_text(yahoo.to_csv(index=False)); out=Path(d)/"probe.json"
            r=run_independent_probe(["RELIANCE.NS"],"2026-11-19","2026-11-20",d,out,"secret-token",session)
            rep=r["reports"]["RELIANCE.NS"]
            self.assertEqual(r["status"],"COMPLETED"); self.assertEqual(rep["upstox"]["missing_completed_bars"],0); self.assertEqual(rep["yahoo"]["missing_completed_bars"],1)
            self.assertEqual(rep["timestamp_comparison"]["upstox_only_timestamps"],1); self.assertTrue(any(h and h.get("Authorization")=="Bearer secret-token" for _,h,_ in session.calls))
            self.assertNotIn("secret-token",out.read_text())


if __name__ == "__main__": unittest.main()
