"""Bounded no-auth probe for a public hosted NSE 1-minute candle MCP; no dataset mutation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import requests

ENDPOINT = "https://vjaiswal-nifty-mcp.hf.space/mcp"
TZ = "Asia/Kolkata"
FIELDS = ["open", "high", "low", "close", "volume"]


def _parse_response(response):
    ctype = response.headers.get("content-type", "")
    if "text/event-stream" in ctype:
        payloads = [line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")]
        for raw in reversed(payloads):
            if raw and raw != "[DONE]":
                try: return json.loads(raw)
                except json.JSONDecodeError: continue
        return None
    try: return response.json()
    except Exception: return None


def _rpc(session, endpoint, message, session_id=None, timeout=20):
    headers = {"Accept":"application/json, text/event-stream","Content-Type":"application/json","MCP-Protocol-Version":"2025-03-26"}
    if session_id: headers["MCP-Session-Id"] = session_id
    response = session.post(endpoint,headers=headers,json=message,timeout=timeout)
    response.raise_for_status()
    parsed = _parse_response(response)
    return parsed,response.headers.get("MCP-Session-Id",session_id),response.status_code,response.headers.get("content-type")


def _tool_schema(tools,name):
    return next((t.get("inputSchema",{}) for t in tools if t.get("name")==name),None)


def build_candle_args(schema,symbol="RELIANCE",day="2026-10-15"):
    props=schema.get("properties",{}) if isinstance(schema,dict) else {};args={}
    for key,prop in props.items():
        enum=prop.get("enum",[]);low=key.lower()
        if low in ("dataset","source","market"):
            args[key]="stocks" if "stocks" in enum else "stock" if "stock" in enum else enum[0] if enum else "stocks"
        elif low in ("symbol","ticker","instrument"):args[key]=symbol
        elif low in ("interval","timeframe","resolution","granularity"):
            args[key]="1minute" if "1minute" in enum else "1m" if "1m" in enum else enum[0] if enum else "1minute"
        elif low in ("start","start_date","from","from_date","date_from","date","trading_date"):args[key]=day
        elif low in ("end","end_date","to","to_date","date_to"):args[key]=day
        elif low in ("limit","max_rows","count"):
            maximum=prop.get("maximum");args[key]=min(500,int(maximum)) if maximum else 500
        elif low in ("desc","descending","reverse"):args[key]=False
    missing=[key for key in schema.get("required",[]) if key not in args] if isinstance(schema,dict) else []
    return args,missing



def build_latest_candle_args(schema,symbol="RELIANCE"):
    """Build a bounded latest-candle query using desc=True, without date filters."""
    args,missing=build_candle_args(schema,symbol,day="")
    required=set(schema.get("required",[])) if isinstance(schema,dict) else set()
    date_keys=[k for k in args if k.lower() in ("start","start_date","from","from_date","date_from","date","trading_date","end","end_date","to","to_date","date_to")]
    if any(k in required for k in date_keys):
        return args,["latest query requires optional date bounds"]
    for k in date_keys: args.pop(k,None)
    desc_key=next((k for k in schema.get("properties",{}) if k.lower() in ("desc","descending","reverse")),None)
    if desc_key: args[desc_key]=True
    else: missing.append("latest query requires descending flag")
    limit_key=next((k for k in schema.get("properties",{}) if k.lower() in ("limit","max_rows","count")),None)
    if limit_key:
        maximum=schema["properties"][limit_key].get("maximum")
        args[limit_key]=min(3,int(maximum)) if maximum else 3
    return args,missing


def build_list_symbols_args(schema,dataset="stocks"):
    """Build a conservative list_symbols query from its advertised schema."""
    args={}
    props=schema.get("properties",{}) if isinstance(schema,dict) else {}
    for key,prop in props.items():
        low=key.lower();enum=prop.get("enum",[])
        if low in ("dataset","source","market"):
            args[key]=dataset if dataset in enum else enum[0] if enum else dataset
        elif low in ("limit","max_rows","count"):
            maximum=prop.get("maximum");args[key]=min(200,int(maximum)) if maximum else 200
    missing=[k for k in schema.get("required",[]) if k not in args] if isinstance(schema,dict) else []
    return args,missing


def _symbol_records(payload):
    """Extract symbol metadata records from common MCP JSON payload shapes."""
    if isinstance(payload, list):
        return [item if isinstance(item, dict) else {"symbol": item} for item in payload]
    if isinstance(payload, dict):
        for key in ("symbols", "items", "data", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"symbol": item} for item in value]
        if any(key in payload for key in ("symbol", "ticker", "name")):
            return [payload]
    return []


def summarize_tool_result(result, target_symbol="RELIANCE", target_date="2026-10-15"):
    structured = result.get("structuredContent") if isinstance(result, dict) else None
    content = result.get("content", []) if isinstance(result, dict) else []
    raw_text_items = [x.get("text", "") for x in content if x.get("type") == "text"]
    text_items = [raw[:1500] for raw in raw_text_items]
    decoded = []
    for raw in raw_text_items:
        try:
            decoded.append(json.loads(raw))
        except Exception:
            continue

    # MCP servers may return one JSON object per text content item. Decode all of
    # them instead of treating only the first record as the entire symbol list.
    structured_records = _symbol_records(structured) if structured is not None else []
    # Some MCP implementations populate structuredContent with generic status
    # metadata while actual symbol records are delivered as separate text items.
    records = structured_records
    if not records:
        for payload in decoded:
            records.extend(_symbol_records(payload))
    names = []
    for record in records:
        value = record.get("symbol", record.get("ticker", record.get("name", "")))
        if value is not None and str(value).strip() and str(value).lower() != "none":
            names.append(str(value).strip())
    unique_symbols = list(dict.fromkeys(names))
    normalized_target = target_symbol.upper().replace(".NS", "")
    target_records = [
        record for record in records
        if str(record.get("symbol", record.get("ticker", record.get("name", "")))).upper().replace(".NS", "") == normalized_target
    ]

    row_values = [record.get("rows") for record in records if isinstance(record.get("rows"), (int, float))]
    starts = [str(record["start"])[:10] for record in records if record.get("start")]
    ends = [str(record["end"])[:10] for record in records if record.get("end")]
    target_coverage = []
    for record in target_records[:5]:
        start = str(record.get("start", ""))[:10] or None
        end = str(record.get("end", ""))[:10] or None
        target_coverage.append({
            "symbol": record.get("symbol", record.get("ticker", record.get("name"))),
            "rows": record.get("rows"),
            "start": start,
            "end": end,
            "target_date_within_reported_range": bool(start and end and start <= str(target_date)[:10] <= end),
        })
    return {
        "is_error": bool(result.get("isError", False)) if isinstance(result, dict) else None,
        "structured_content_type": type(structured).__name__ if structured is not None else None,
        "text_content_item_count": len(raw_text_items),
        "decoded_text_payload_count": len(decoded),
        "symbol_record_count": len(records),
        "symbol_count": len(unique_symbols),
        "contains_reliance": any(x.upper().replace(".NS", "") == "RELIANCE" for x in unique_symbols),
        "sample_symbols": unique_symbols[:10],
        "reported_rows_total": sum(row_values) if row_values else None,
        "reported_start_min": min(starts) if starts else None,
        "reported_end_max": max(ends) if ends else None,
        "target_symbol_metadata": target_coverage,
        "text_content": text_items[:2],
    }

def _extract_rows(result):
    if not isinstance(result,dict):return result if isinstance(result,list) else []
    structured=result.get("structuredContent")
    if isinstance(structured,dict):
        for key in ("candles","data","rows","ohlcv"):
            if isinstance(structured.get(key),list):return structured[key]
    for key in ("candles","data","rows","ohlcv"):
        if isinstance(result.get(key),list):return result[key]
    for item in result.get("content",[]):
        if item.get("type")=="text":
            try:
                decoded=json.loads(item.get("text",""))
                if isinstance(decoded,list):return decoded
                if isinstance(decoded,dict):
                    for key in ("candles","data","rows","ohlcv"):
                        if isinstance(decoded.get(key),list):return decoded[key]
            except Exception:pass
    return []


def normalize_rows(rows):
    if not isinstance(rows,list) or not rows:return pd.DataFrame(columns=["timestamp",*FIELDS])
    df=pd.DataFrame(rows);tc=next((c for c in ("timestamp","datetime","time","date") if c in df),None)
    if tc is None:return pd.DataFrame(columns=["timestamp",*FIELDS])
    t=df[tc]
    if pd.api.types.is_numeric_dtype(t):
        nums=pd.to_numeric(t,errors="coerce");unit="ms" if nums.abs().max(skipna=True)>=1e12 else "s"
        df["timestamp"]=pd.to_datetime(nums,unit=unit,utc=True,errors="coerce").dt.tz_convert(TZ)
    else:
        parsed=pd.to_datetime(t,errors="coerce")
        df["timestamp"]=parsed.dt.tz_localize(TZ) if parsed.dt.tz is None else parsed.dt.tz_convert(TZ)
    df["timestamp"]=df["timestamp"].dt.floor("min")
    for c in FIELDS:
        if c not in df:df[c]=pd.NA
        df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def read_yahoo(path):
    if not path.exists():return pd.DataFrame(columns=["timestamp",*FIELDS])
    df=pd.read_csv(path);df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    df["timestamp"]=df["timestamp"].dt.tz_localize(TZ) if df["timestamp"].dt.tz is None else df["timestamp"].dt.tz_convert(TZ)
    for c in FIELDS:
        if c in df:df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.dropna(subset=["timestamp"])


def bar_at(df,ts):
    rows=df.loc[df.timestamp==ts]
    if rows.empty:return None
    r=rows.iloc[0];b={c:float(r[c]) for c in FIELDS if c in r and pd.notna(r[c])}
    return b if len(b)==len(FIELDS) else None


def aggregate_five(df,start):
    stamps=[start+pd.Timedelta(minutes=i) for i in range(5)]
    part=df.loc[df.timestamp.isin(stamps)].sort_values("timestamp")
    if len(part)!=5 or list(part.timestamp)!=stamps or part[FIELDS].isna().any().any():return None
    o,c,h,l,v=part.open.iloc[0],part.close.iloc[-1],part.high.max(),part.low.min(),part.volume.sum()
    if min(o,c,h,l)<=0 or h<max(o,c,l) or l>min(o,c,h) or v<0:return None
    return {"open":float(o),"high":float(h),"low":float(l),"close":float(c),"volume":float(v),"one_minute_rows":5}


def matches(a,b):
    return a is not None and b is not None and all(abs(a[k]-b[k])<=(1e-9 if k=="volume" else 1e-4) for k in FIELDS)


def probe(directory,output,endpoint=ENDPOINT,session=None,day="2026-10-15"):
    session=session or requests.Session()
    report={"provider":"public hosted indian-stocks-mcp MCP server","endpoint":endpoint,"authentication_required":False,
        "cost":"free/no API key according to provider documentation","provenance_warning":"Hosted source is described as Kaggle-backed; exact source dataset and collection provenance are not independently verified.",
        "target_symbol":"RELIANCE","target_date":day,"datasets_modified":False,"data_repaired":False}
    sid=None
    try:
        init,sid,status,ctype=_rpc(session,endpoint,{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"intraday-stage8-probe","version":"0.1"}}})
        report["initialize"]={"http_status":status,"content_type":ctype,"protocol_version":(init or {}).get("result",{}).get("protocolVersion"),"session_created":bool(sid),"error":(init or {}).get("error")}
        if not init or init.get("error"):
            report["status"]="INITIALIZE_FAILED";report["error"]=(init or {}).get("error");return _write(report,output)
        headers={"Accept":"application/json, text/event-stream","Content-Type":"application/json","MCP-Protocol-Version":"2025-03-26"}
        if sid:headers["MCP-Session-Id"]=sid
        session.post(endpoint,headers=headers,json={"jsonrpc":"2.0","method":"notifications/initialized"},timeout=20)
        listed,sid,ls,lc=_rpc(session,endpoint,{"jsonrpc":"2.0","id":2,"method":"tools/list"},sid)
        tools=(listed or {}).get("result",{}).get("tools",[])
        report["tools_list"]={"http_status":ls,"content_type":lc,"tool_names":[t.get("name") for t in tools]}
        schemas={t.get("name"):_tool_schema(tools,t.get("name")) for t in tools}
        report["tool_schemas"]={name:{"required":schema.get("required",[]),"properties":{k:{"type":v.get("type"),"enum":v.get("enum"),"description":v.get("description")} for k,v in schema.get("properties",{}).items()}} for name,schema in schemas.items() if schema}
        if schemas.get("list_symbols") is not None:
            lsargs,lmissing=build_list_symbols_args(schemas["list_symbols"],"stocks")
            report["list_symbols_request"]={"args":lsargs,"missing_required":lmissing}
            if not lmissing:
                listed,sid,lss,lct=_rpc(session,endpoint,{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_symbols","arguments":lsargs}},sid,timeout=30)
                lres=(listed or {}).get("result",{})
                report["list_symbols_response"]={"http_status":lss,"content_type":lct,"jsonrpc_error":(listed or {}).get("error"),
                    **summarize_tool_result(lres, target_symbol="RELIANCE", target_date=day)}
        if any(t.get("name")=="dataset_status" for t in tools):
            ds,sid,dss,dct=_rpc(session,endpoint,{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"dataset_status","arguments":{}}},sid)
            dres=(ds or {}).get("result",{})
            report["dataset_status"]={"http_status":dss,"content_type":dct,"jsonrpc_error":(ds or {}).get("error",""),
                "is_error":dres.get("isError",False),"text":[x.get("text","")[:1000] for x in dres.get("content",[]) if x.get("type")=="text"][:2],
                "structured_content":json.dumps(dres.get("structuredContent"),default=str)[:1500] if dres.get("structuredContent") is not None else None}
        schema=schemas.get("get_candles")
        if schema is None:report["status"]="GET_CANDLES_TOOL_MISSING";return _write(report,output)
        args,missing=build_candle_args(schema,"RELIANCE",day)
        report["tool_schema"]={"properties":sorted(schema.get("properties",{})),"required":schema.get("required",[]),"args":args,"missing_required":missing}
        if missing:report["status"]="TOOL_SCHEMA_UNSUPPORTED";return _write(report,output)
        latest_args,latest_missing=build_latest_candle_args(schema,"RELIANCE")
        report["latest_candle_request"]={"args":latest_args,"missing_required":latest_missing}
        if not latest_missing:
            latest,sid,lcs,lct=_rpc(session,endpoint,{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"get_candles","arguments":latest_args}},sid,timeout=30)
            latest_result=(latest or {}).get("result",{});latest_rows=_extract_rows(latest_result);latest_bars=normalize_rows(latest_rows)
            report["latest_candle_response"]={"http_status":lcs,"content_type":lct,"jsonrpc_error":(latest or {}).get("error"),
                "is_error":latest_result.get("isError",False),"returned_rows":len(latest_rows),"normalized_rows":len(latest_bars),
                "timestamps_first_last":[str(latest_bars.timestamp.min()),str(latest_bars.timestamp.max())] if not latest_bars.empty else [],
                "row_keys":sorted(latest_rows[0].keys()) if latest_rows and isinstance(latest_rows[0],dict) else []}
        called,sid,cs,cc=_rpc(session,endpoint,{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"get_candles","arguments":args}},sid,timeout=30)
        result=(called or {}).get("result",{});rows=_extract_rows(result);bars=normalize_rows(rows)
        report["tool_call"]={"http_status":cs,"content_type":cc,"jsonrpc_error":(called or {}).get("error"),"is_error":result.get("isError",False),
            "returned_rows":len(rows),"normalized_rows":len(bars),"row_keys":sorted(rows[0].keys()) if rows and isinstance(rows[0],dict) else [],
            "timestamps_first_last":[str(bars.timestamp.min()),str(bars.timestamp.max())] if not bars.empty else [],
            "text_content":[x.get("text","")[:400] for x in result.get("content",[]) if x.get("type")=="text"][:2]}
        yahoo=read_yahoo(Path(directory)/"RELIANCE.NS_5m.csv")
        targets=[pd.Timestamp(f"{day} {hm}",tz=TZ) for hm in ("11:45","11:50")];neighbors=[pd.Timestamp(f"{day} {hm}",tz=TZ) for hm in ("11:40","11:55")]
        agg={str(t):aggregate_five(bars,t) for t in targets}
        checks=[{"timestamp":str(t),"source_5m":aggregate_five(bars,t),"yahoo_5m":bar_at(yahoo,t),"match":matches(aggregate_five(bars,t),bar_at(yahoo,t))} for t in neighbors]
        complete=all(agg.values());neighbor_ok=all(x["match"] for x in checks)
        report["target_validation"]={"target_5m_candidates":agg,"target_complete":complete,"neighbor_validation":{"compared":2,"matches":sum(x["match"] for x in checks),"checks":checks},
            "candidate_status":"CANDIDATE_FOR_MANUAL_REVIEW" if complete and neighbor_ok else "WITHHELD_VALIDATION_FAILED" if not bars.empty else "NO_ROWS_OR_UNAVAILABLE"}
        report["status"]="QUERY_COMPLETED" if called and not called.get("error") and not result.get("isError",False) else "QUERY_FAILED"
    except Exception as exc:report["status"]="EXCEPTION";report["error"]=f"{type(exc).__name__}: {exc}"
    return _write(report,output)


def _write(report,output):
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8");return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--directory",default="data/research");p.add_argument("--output",default="reports/public_stock_mcp_probe.json");p.add_argument("--date",default="2026-10-15");a=p.parse_args()
    print(json.dumps(probe(a.directory,a.output,day=a.date),indent=2,default=str))
if __name__=="__main__":main()
