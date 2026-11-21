"""Bounded diagnostic for NSE chart API timestamp conversion; no dataset mutation."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import pandas as pd

TZ = "Asia/Kolkata"
FIELDS = ["open", "high", "low", "close", "volume"]


def fetch_raw_chart(symbol, start, end, interval=5):
    """Use nsemine token/session helpers, but send true UTC epoch seconds."""
    from nsemine import historical
    from nsemine.bin import scraper
    from nsemine.utilities import urls
    lookup = historical.__get_script_token(symbol=symbol, segment="EQ")
    if not lookup:
        return {"diagnostic_status": "TOKEN_LOOKUP_FAILED"}
    resolved_symbol, token, symbol_type = lookup
    params = {"chartType": "I", "fromDate": int(start.tz_convert("UTC").timestamp()),
              "symbol": resolved_symbol, "symbolType": symbol_type, "timeInterval": int(interval),
              "toDate": int(end.tz_convert("UTC").timestamp()), "token": token}
    response = scraper.get_request(url=urls.nse_chart_url, params=params)
    if response is None:
        return {"diagnostic_status": "NO_HTTP_RESPONSE", "request_epoch": {"fromDate": params["fromDate"], "toDate": params["toDate"]}}
    try:
        payload = response.json()
    except Exception as exc:
        return {"diagnostic_status": "NON_JSON_RESPONSE", "http_status": getattr(response, "status_code", None),
                "content_type": getattr(response, "headers", {}).get("content-type"),
                "error": f"{type(exc).__name__}: {exc}"}
    return {"diagnostic_status": "HTTP_RESPONSE", "http_status": getattr(response, "status_code", None),
            "content_type": getattr(response, "headers", {}).get("content-type"), "payload": payload,
            "request_epoch": {"fromDate": params["fromDate"], "toDate": params["toDate"],
                              "start_utc": start.tz_convert("UTC").isoformat(), "end_utc": end.tz_convert("UTC").isoformat()}}


def payload_of(raw):
    return raw.get("payload") if isinstance(raw, dict) and (raw.get("diagnostic_status") == "HTTP_RESPONSE" or isinstance(raw.get("payload"), dict)) else raw


def summarize(raw):
    payload = payload_of(raw)
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    message = (payload.get("message") or payload.get("error") or payload.get("msg")) if isinstance(payload, dict) else None
    if isinstance(message, (dict, list)): message = json.dumps(message, default=str)
    return {"status": raw.get("diagnostic_status", "RAW_PAYLOAD") if isinstance(raw, dict) else "NO_PAYLOAD",
            "http_status": raw.get("http_status") if isinstance(raw, dict) else None,
            "content_type": raw.get("content_type") if isinstance(raw, dict) else None,
            "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
            "api_status": payload.get("status") if isinstance(payload, dict) else None,
            "message": str(message)[:300] if message is not None else None,
            "rows": len(rows) if isinstance(rows, list) else 0,
            "first_row_keys": sorted(rows[0].keys()) if rows and isinstance(rows[0], dict) else [],
            "request_epoch": raw.get("request_epoch") if isinstance(raw, dict) else None}


def normalize(raw):
    payload = payload_of(raw)
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data: return pd.DataFrame(columns=["timestamp", *FIELDS])
    df = pd.DataFrame(data)
    if "time" not in df: return pd.DataFrame(columns=["timestamp", *FIELDS])
    t = df["time"]
    if pd.api.types.is_numeric_dtype(t):
        nums = pd.to_numeric(t, errors="coerce")
        unit = "ms" if nums.abs().max(skipna=True) >= 1e12 else "s"
        df["timestamp"] = pd.to_datetime(nums, unit=unit, utc=True, errors="coerce").dt.tz_convert(TZ)
    else:
        parsed = pd.to_datetime(t, errors="coerce")
        df["timestamp"] = parsed.dt.tz_localize(TZ) if parsed.dt.tz is None else parsed.dt.tz_convert(TZ)
    for c in FIELDS:
        if c not in df: df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def read_yahoo(path):
    if not path.exists(): return pd.DataFrame(columns=["timestamp", *FIELDS])
    df = pd.read_csv(path); df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = df["timestamp"].dt.tz_localize(TZ) if df["timestamp"].dt.tz is None else df["timestamp"].dt.tz_convert(TZ)
    for c in FIELDS:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["timestamp"])


def bar_at(df, ts):
    if df.empty: return None
    rows = df.loc[df.timestamp == ts]
    if rows.empty: return None
    r = rows.iloc[0]; b = {k: float(r[k]) for k in FIELDS if k in r and pd.notna(r[k])}
    return b if len(b) == len(FIELDS) else None


def valid_bar(b):
    if not b: return False
    o,h,l,c,v = (b[k] for k in FIELDS)
    return all(pd.notna(x) and x > 0 for x in (o,h,l,c)) and v >= 0 and h >= max(o,l,c) and l <= min(o,h,c)


def matches(a,b):
    return a is not None and b is not None and all(abs(a[k]-b[k]) <= (1e-9 if k == "volume" else 1e-4) for k in FIELDS)


def window(day, start_hm, end_hm):
    return pd.Timestamp(f"{day} {start_hm}",tz=TZ), pd.Timestamp(f"{day} {end_hm}",tz=TZ)


def probe(directory, output, symbol="RELIANCE.NS", target_day="2026-10-15", sanity_day="2026-11-19", raw_fetcher=None, delay=0.5):
    fetch = raw_fetcher or fetch_raw_chart; directory=Path(directory); bare=symbol.removesuffix(".NS")
    san_start,san_end=window(sanity_day,"09:15","15:15"); tar_start,tar_end=window(target_day,"11:35","12:05")
    try:
        san_raw=fetch(bare,san_start,san_end,5); san_summary=summarize(san_raw); san=normalize(san_raw)
        health="RECENT_SANITY_DATA_RETURNED" if not san.empty else "RECENT_SANITY_EMPTY_OR_UNAVAILABLE"
    except Exception as e:
        san_summary={"status":"EXCEPTION","error":f"{type(e).__name__}: {e}"}; san=pd.DataFrame(columns=["timestamp",*FIELDS]); health="RECENT_SANITY_ERROR"
    yahoo=read_yahoo(directory/f"{symbol}_5m.csv")
    san_checks=[]
    for hm in ("09:15","12:00","15:10"):
        ts=pd.Timestamp(f"{sanity_day} {hm}",tz=TZ); nse=bar_at(san,ts); y=bar_at(yahoo,ts)
        san_checks.append({"timestamp":str(ts),"nse_chart":nse,"yahoo_5m":y,"match":matches(nse,y)})
    if delay: time.sleep(delay)
    try:
        tar_raw=fetch(bare,tar_start,tar_end,5); tar_summary=summarize(tar_raw); tar=normalize(tar_raw)
    except Exception as e:
        tar_summary={"status":"EXCEPTION","error":f"{type(e).__name__}: {e}"}; tar=pd.DataFrame(columns=["timestamp",*FIELDS])
    targets=[pd.Timestamp(f"{target_day} {hm}",tz=TZ) for hm in ("11:45","11:50")]
    neighbors=[pd.Timestamp(f"{target_day} {hm}",tz=TZ) for hm in ("11:40","11:55")]
    vals={str(t):bar_at(tar,t) for t in targets}; validity={str(t):valid_bar(vals[str(t)]) for t in targets}
    checks=[{"timestamp":str(t),"nse_chart":bar_at(tar,t),"yahoo_5m":bar_at(yahoo,t),"match":matches(bar_at(tar,t),bar_at(yahoo,t))} for t in neighbors]
    complete=all(vals[str(t)] is not None for t in targets); valid=all(validity.values()); neigh=all(x["match"] for x in checks)
    san_valid=not san.empty and sum(x["match"] for x in san_checks)>=1
    candidate=complete and valid and neigh and san_valid
    status="CANDIDATE_FOR_MANUAL_REVIEW" if candidate else "WITHHELD_SANITY_NOT_VALIDATED" if not tar.empty and not san_valid else "WITHHELD_VALIDATION_FAILED" if not tar.empty else "TARGET_EMPTY_OR_UNAVAILABLE"
    result={"provider":"NSE charting.nseindia.com/v1/charts/symbolHistoricalData via nsemine token/session helper",
        "authentication_required":False,"epoch_mode":"true UTC Unix seconds from timezone-aware IST timestamps; no extra +19800 seconds",
        "hypothesis":"The nsemine historical wrapper adds 19800 seconds after datetime.timestamp(); with runner TZ=Asia/Kolkata this may shift the query by +5h30m. This direct request isolates that conversion.",
        "symbol":symbol,"target_date":target_day,"sanity_date":sanity_day,"provider_health":health,
        "sanity_query":{"start_ist":str(san_start),"end_ist_exclusive":str(san_end),"raw_summary":san_summary,"rows_parsed":len(san),
            "first_timestamp":str(san.timestamp.min()) if not san.empty else None,"last_timestamp":str(san.timestamp.max()) if not san.empty else None,
            "neighbor_validation":{"compared":len(san_checks),"matches":sum(x["match"] for x in san_checks),"sanity_validated":san_valid,"checks":san_checks}},
        "target_query":{"start_ist":str(tar_start),"end_ist_exclusive":str(tar_end),"raw_summary":tar_summary,"rows_parsed":len(tar),
            "timestamps_ist":[str(x) for x in tar.timestamp],"target_bars":vals,"target_bar_validity":validity,"target_complete":complete,
            "neighbor_validation":{"compared":len(checks),"matches":sum(x["match"] for x in checks),"checks":checks},"candidate_status":status},
        "manual_review_candidates":int(candidate),"datasets_modified":False,"data_repaired":False,
        "interpretation":"Diagnostic only. Recent sanity must return bars and match at least one Yahoo sample; target candidates require both complete valid target bars and exact neighboring Yahoo OHLCV matches. No CSV is changed."}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--directory",default="data/research");p.add_argument("--symbol",default="RELIANCE.NS")
    p.add_argument("--target-date",default="2026-10-15");p.add_argument("--sanity-date",default="2026-11-19");p.add_argument("--output",default="reports/nse_epoch_probe.json");a=p.parse_args()
    print(json.dumps(probe(a.directory,a.output,a.symbol,a.target_date,a.sanity_date),indent=2,default=str))
if __name__=="__main__": main()
