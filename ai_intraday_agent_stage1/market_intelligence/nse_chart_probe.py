"""No-auth NSE chart API probe for the Oct 15 Yahoo gaps; diagnostic only."""
from __future__ import annotations
import argparse, json, time
from datetime import datetime, date, time as dtime
from pathlib import Path
import pandas as pd

TZ = "Asia/Kolkata"
FIELDS = ["open", "high", "low", "close", "volume"]

def fetch_nse_chart(symbol, start, end, interval=5):
    from nsemine import historical
    return historical.get_stock_historical_data(symbol, start, end, interval=interval, raw=False)

def normalize(frame):
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty or "datetime" not in frame:
        return pd.DataFrame(columns=["timestamp", *FIELDS])
    out=frame.copy()
    out["timestamp"]=pd.to_datetime(out["datetime"],errors="coerce")
    if out["timestamp"].dt.tz is None: out["timestamp"]=out["timestamp"].dt.tz_localize(TZ)
    else: out["timestamp"]=out["timestamp"].dt.tz_convert(TZ)
    for col in FIELDS:
        if col not in out: out[col]=pd.NA
        out[col]=pd.to_numeric(out[col],errors="coerce")
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

def read_yahoo(path):
    if not path.exists(): return pd.DataFrame(columns=["timestamp",*FIELDS])
    f=pd.read_csv(path); f["timestamp"]=pd.to_datetime(f["timestamp"],errors="coerce")
    if f["timestamp"].dt.tz is None: f["timestamp"]=f["timestamp"].dt.tz_localize(TZ)
    else: f["timestamp"]=f["timestamp"].dt.tz_convert(TZ)
    return f.dropna(subset=["timestamp"])

def bar_at(frame, stamp):
    rows=frame.loc[frame.timestamp==stamp]
    if rows.empty: return None
    row=rows.iloc[0]
    vals={k:float(row[k]) for k in FIELDS if pd.notna(row[k])}
    return vals if len(vals)==len(FIELDS) else None

def matches(a,b):
    if a is None or b is None: return False
    return all(abs(a[k]-b[k]) <= (1e-9 if k=="volume" else 1e-4) for k in FIELDS)

def probe(directory,symbols,output,day="2026-10-15",fetcher=None,delay=0.5):
    fetcher=fetcher or fetch_nse_chart; directory=Path(directory); reports={}
    targets=[pd.Timestamp(f"{day} {t}",tz=TZ) for t in ("11:45","11:50")]
    neighbors=[pd.Timestamp(f"{day} {t}",tz=TZ) for t in ("11:40","11:55")]
    start=datetime.combine(date.fromisoformat(day),dtime(11,35)); end=datetime.combine(date.fromisoformat(day),dtime(12,5))
    for symbol in symbols:
        original=read_yahoo(directory/f"{symbol}_5m.csv")
        try:
            fetched=normalize(fetcher(symbol.removesuffix(".NS"),start,end,5))
            candidates={str(t):bar_at(fetched,t) for t in targets}
            checks=[{"timestamp":str(t),"nse_chart":bar_at(fetched,t),"yahoo_5m":bar_at(original,t),"match":matches(bar_at(fetched,t),bar_at(original,t))} for t in neighbors]
            complete=all(candidates[str(t)] is not None for t in targets)
            neighbor_match=bool(checks) and all(x["match"] for x in checks)
            status="OK" if not fetched.empty else "EMPTY"
            candidate_status=("CANDIDATES_FOR_MANUAL_REVIEW" if complete and neighbor_match else
                              "WITHHELD_NEIGHBOR_MISMATCH" if complete else
                              "NO_COMPLETE_TARGET_BARS" if not fetched.empty else "EMPTY_OR_UNAVAILABLE")
            reports[symbol]={"status":status,"rows_returned":len(fetched),"timestamps_ist":[str(x) for x in fetched.timestamp],
                "target_bars":candidates,"target_complete":complete,
                "neighbor_validation":{"compared":len(checks),"matches":sum(x["match"] for x in checks),
                    "mismatches":sum(not x["match"] for x in checks),"checks":checks,"all_match":neighbor_match},
                "candidate_status":candidate_status,"csv_modified":False}
        except Exception as exc:
            reports[symbol]={"status":"ERROR","error":f"{type(exc).__name__}: {exc}",
                "candidate_status":"INCONCLUSIVE_PROVIDER_ERROR","target_bars":{str(t):None for t in targets},"csv_modified":False}
        if delay: time.sleep(delay)
    count=sum(x.get("candidate_status")=="CANDIDATES_FOR_MANUAL_REVIEW" for x in reports.values())
    result={"provider":"NSE public charting endpoint charting.nseindia.com via unofficial nsemine wrapper",
        "authentication_required":False,"separate_exchange_endpoint_from_yahoo":True,"date":day,
        "targets_ist":[str(t) for t in targets],"query_window_ist":{"start":str(start),"end_exclusive":str(end)},
        "symbols_requested":symbols,"symbols_with_manual_review_candidates":count,"data_repaired":False,
        "datasets_modified":False,"interpretation":"Diagnostic only. Candidates require both target bars and exact neighbor checks; no CSV is changed.",
        "reports":reports}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--directory",default="data/research")
    p.add_argument("--symbols",default="RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS,SBIN.NS,AXISBANK.NS,KOTAKBANK.NS,BHARTIARTL.NS,ITC.NS")
    p.add_argument("--date",default="2026-10-15");p.add_argument("--output",default="reports/nse_chart_probe.json");a=p.parse_args()
    print(json.dumps(probe(a.directory,[s.strip() for s in a.symbols.split(",") if s.strip()],a.output,a.date),indent=2,default=str))
if __name__=="__main__": main()
