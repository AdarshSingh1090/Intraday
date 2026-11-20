"""Research-only Yahoo versus independent Upstox 5-minute candle comparison."""
from __future__ import annotations
import argparse, gzip, json, os
from datetime import datetime, time, timedelta
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo
import pandas as pd

TIMEZONE = 'Asia/Kolkata'
UPSTOX_INSTRUMENTS_URL = 'https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz'
UPSTOX_HISTORICAL_URL = 'https://api.upstox.com/v3/historical-candle/{}/minutes/5/{}/{}'


def expected_session_grid(day, interval_minutes=5, timezone=TIMEZONE, session_start="09:15", session_end="15:30"):
    start = pd.Timestamp(datetime.combine(day, pd.to_datetime(session_start).time()), tz=timezone)
    end = pd.Timestamp(datetime.combine(day, pd.to_datetime(session_end).time()), tz=timezone)
    return pd.date_range(start=start, end=end, freq=f'{interval_minutes}min', inclusive='left')


def normalize_provider_frame(frame):
    cols=['timestamp','open','high','low','close','volume']
    if frame is None or frame.empty: return pd.DataFrame(columns=cols)
    data=frame.copy()
    if isinstance(data.columns,pd.MultiIndex): data.columns=data.columns.get_level_values(0)
    if 'timestamp' not in data.columns: data=data.reset_index()
    aliases={str(c).lower():c for c in data.columns}
    time_col=next((aliases[k] for k in ('timestamp','datetime','date') if k in aliases),None)
    if time_col is None: raise ValueError('Provider frame lacks timestamp column')
    rename={time_col:'timestamp'}
    for c in cols[1:]:
        if c not in aliases: raise ValueError(f'Provider frame lacks {c}')
        rename[aliases[c]]=c
    data=data.rename(columns=rename)[cols]
    ts=pd.to_datetime(data.timestamp,errors='coerce')
    data['timestamp']=ts.dt.tz_localize(TIMEZONE) if ts.dt.tz is None else ts.dt.tz_convert(TIMEZONE)
    return data.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)


def _as_of(as_of=None):
    now=pd.Timestamp(as_of or datetime.now(ZoneInfo(TIMEZONE)))
    return now.tz_localize(TIMEZONE) if now.tzinfo is None else now.tz_convert(TIMEZONE)


def summarize_provider_frame(frame,as_of=None,interval_minutes=5,session_start="09:15",session_end="15:30",post_session_end="16:00"):
    data=normalize_provider_frame(frame); now=_as_of(as_of); dates=sorted(set(data.timestamp.dt.date)); missing_by_date={}
    start_time=pd.to_datetime(session_start).time(); end_time=pd.to_datetime(session_end).time(); post_end=pd.to_datetime(post_session_end).time()
    post_freq={}; outside_freq={}; session_rows=0
    for day in dates:
        expected=expected_session_grid(day,interval_minutes,session_start=session_start,session_end=session_end)
        expected=expected[expected+pd.Timedelta(minutes=interval_minutes)<=now]
        day_data=data.loc[data.timestamp.dt.date==day,'timestamp']
        observed=set(day_data[(day_data.dt.time>=start_time)&(day_data.dt.time<end_time)])
        missing_by_date[str(day)]=[t.strftime('%H:%M') for t in expected if t not in observed]
        post=day_data[(day_data.dt.time>=end_time)&(day_data.dt.time<post_end)]
        outside=day_data[(day_data.dt.time<start_time)|(day_data.dt.time>=post_end)]
        session_rows+=len(observed)
        for t in post:
            k=t.strftime('%H:%M'); post_freq[k]=post_freq.get(k,0)+1
        for t in outside:
            k=t.strftime('%H:%M'); outside_freq[k]=outside_freq.get(k,0)+1
    times=[t for v in missing_by_date.values() for t in v]
    return {'rows':int(len(data)),'continuous_session_rows':session_rows,
      'post_session_observations_excluded':sum(post_freq.values()),
      'post_session_time_frequency':dict(sorted(post_freq.items())),
      'outside_market_window_observations':sum(outside_freq.values()),
      'outside_market_window_time_frequency':dict(sorted(outside_freq.items())),
      'duplicate_timestamps':int(data.timestamp.duplicated().sum()),
      'first_timestamp':str(data.timestamp.min()) if len(data) else None,'last_timestamp':str(data.timestamp.max()) if len(data) else None,
      'observed_dates':len(dates),'missing_completed_bars':len(times),
      'missing_time_frequency':{t:times.count(t) for t in sorted(set(times))},'missing_by_date':missing_by_date,
      'session':{'start':session_start,'end_exclusive':session_end,'post_session_window_end':post_session_end}}

def _union_missing(primary,alternate,as_of=None,interval_minutes=5,session_start="09:15",session_end="15:30"):
    a,b=normalize_provider_frame(primary),normalize_provider_frame(alternate); dates=sorted(set(a.timestamp.dt.date)|set(b.timestamp.dt.date)); now=_as_of(as_of)
    freq={'primary':{},'alternate':{}}; totals={'primary':0,'alternate':0}
    start_time=pd.to_datetime(session_start).time(); end_time=pd.to_datetime(session_end).time()
    for day in dates:
        exp=expected_session_grid(day,interval_minutes,session_start=session_start,session_end=session_end); exp=exp[exp+pd.Timedelta(minutes=interval_minutes)<=now]
        for name,frame in (('primary',a),('alternate',b)):
            day_data=frame.loc[frame.timestamp.dt.date==day,'timestamp']
            obs=set(day_data[(day_data.dt.time>=start_time)&(day_data.dt.time<end_time)])
            missing=[t for t in exp if t not in obs]; totals[name]+=len(missing)
            for t in missing:
                k=t.strftime('%H:%M'); freq[name][k]=freq[name].get(k,0)+1
    keys=sorted(set(freq['primary'])|set(freq['alternate']))
    return {'union_observed_dates':len(dates),'missing_completed_bars':totals,'missing_time_frequency':freq,
      'alternate_minus_primary_frequency':{k:freq['alternate'].get(k,0)-freq['primary'].get(k,0) for k in keys},
      'session':{'start':session_start,'end_exclusive':session_end}}

def _session_timestamp_comparison(primary, alternate, session_start="09:15", session_end="15:30"):
    a,b=normalize_provider_frame(primary),normalize_provider_frame(alternate)
    start_time=pd.to_datetime(session_start).time(); end_time=pd.to_datetime(session_end).time()
    ta=set(a.timestamp[(a.timestamp.dt.time>=start_time)&(a.timestamp.dt.time<end_time)])
    tb=set(b.timestamp[(b.timestamp.dt.time>=start_time)&(b.timestamp.dt.time<end_time)])
    return {'yahoo_only_timestamps':len(ta-tb),'upstox_only_timestamps':len(tb-ta),'matched_timestamps':len(ta&tb)}

def compare_frames(primary,alternate,as_of=None,interval_minutes=5,session_start="09:15",session_end="15:30",post_session_end="16:00"):
    a,b=normalize_provider_frame(primary),normalize_provider_frame(alternate); start_time=pd.to_datetime(session_start).time(); end_time=pd.to_datetime(session_end).time()
    ta=set(a.timestamp[(a.timestamp.dt.time>=start_time)&(a.timestamp.dt.time<end_time)])
    tb=set(b.timestamp[(b.timestamp.dt.time>=start_time)&(b.timestamp.dt.time<end_time)])
    return {'primary':summarize_provider_frame(a,as_of,interval_minutes,session_start,session_end,post_session_end),
      'alternate':summarize_provider_frame(b,as_of,interval_minutes,session_start,session_end,post_session_end),
      'timestamp_comparison':{'primary_only_timestamps':len(ta-tb),'alternate_only_timestamps':len(tb-ta),'matched_timestamps':len(ta&tb)},
      'raw_timestamp_comparison':{'primary_only_timestamps':len(set(a.timestamp)-set(b.timestamp)),
        'alternate_only_timestamps':len(set(b.timestamp)-set(a.timestamp)),'matched_timestamps':len(set(a.timestamp)&set(b.timestamp))},
      'missing_frequency_comparison':_union_missing(a,b,as_of,interval_minutes,session_start,session_end)}

def run_probe(symbols,start,end,output,session_start='09:15',session_end='15:30',post_session_end='16:00'):
    import yfinance as yf
    reports={}
    for symbol in symbols:
        try:
            history=yf.Ticker(symbol).history(start=start,end=end,interval='5m',auto_adjust=False,prepost=False)
            download=yf.download(symbol,start=start,end=end,interval='5m',auto_adjust=False,prepost=False,progress=False,group_by='column')
            reports[symbol]=compare_frames(history,download,session_start=session_start,session_end=session_end,post_session_end=post_session_end)
        except Exception as exc: reports[symbol]={'error':f'{type(exc).__name__}: {exc}'}
    result={'purpose':'compare yfinance interfaces; no data repair or strategy tuning','provider':'Yahoo Finance via yfinance',
      'yfinance_version':getattr(yf,'__version__','unknown'),'comparison_scope':'same yfinance package and Yahoo source; yf.download is an interface wrapper, not an independent provider',
      'independent_provider_comparison':False,'interpretation_warning':'Identical interface results do not validate source completeness; independent market data is required.',
      'start':start,'end_exclusive':end,'interval_minutes':5,'timezone':TIMEZONE,'session':{'start':session_start,'end_exclusive':session_end,'post_session_window_end':post_session_end},'reports':reports}
    Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(result,indent=2,default=str),encoding='utf-8'); return result


def _instrument_map(session,timeout=30):
    r=session.get(UPSTOX_INSTRUMENTS_URL,timeout=timeout); r.raise_for_status(); items=json.loads(gzip.decompress(r.content).decode())
    return {str(x.get('trading_symbol','')).upper():x['instrument_key'] for x in items if x.get('exchange')=='NSE' and x.get('segment')=='NSE_EQ' and x.get('instrument_type')=='EQ' and x.get('trading_symbol') and x.get('instrument_key')}


def _fetch_upstox(session,key,start,end_exclusive,token,timeout=30):
    cursor=pd.Timestamp(start).date(); last=pd.Timestamp(end_exclusive).date()-timedelta(days=1); frames=[]
    while cursor<=last:
        chunk_end=min(cursor+timedelta(days=27),last); url=UPSTOX_HISTORICAL_URL.format(quote(key,safe=''),chunk_end.isoformat(),cursor.isoformat())
        r=session.get(url,headers={'Accept':'application/json','Authorization':'Bearer '+token},timeout=timeout); r.raise_for_status()
        candles=r.json().get('data',{}).get('candles',[])
        if candles: frames.append(pd.DataFrame(candles,columns=['timestamp','open','high','low','close','volume','oi']).drop(columns=['oi']))
        cursor=chunk_end+timedelta(days=1)
    if not frames:return normalize_provider_frame(None)
    return normalize_provider_frame(pd.concat(frames,ignore_index=True).drop_duplicates(subset=['timestamp']))


def run_independent_probe(symbols,start,end,yahoo_dir,output,token=None,session=None,session_start='09:15',session_end='15:30',post_session_end='16:00'):
    import requests
    session=session or requests.Session(); reports={}
    result={'purpose':'independent-provider completeness comparison; no repair or tuning','provider':'Upstox historical candle V3',
      'independent_provider_comparison':True,'interval_minutes':5,'timezone':TIMEZONE,'start':start,'end_exclusive':end,
      'session':{'start':session_start,'end_exclusive':session_end,'post_session_window_end':post_session_end},
      'status':'SKIPPED_NO_TOKEN','skip_reason':'UPSTOX_ACCESS_TOKEN is not configured; Yahoo-only data cannot establish source completeness.','reports':reports}
    if not token:
        Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8'); return result
    result['status']='COMPLETED'; result.pop('skip_reason',None)
    try: instruments=_instrument_map(session)
    except Exception as exc:
        result['status']='FAILED_INSTRUMENT_MASTER'; result['error']=f'{type(exc).__name__}: {exc}'
        Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8'); return result
    for symbol in symbols:
        base=symbol.upper().removesuffix('.NS'); path=Path(yahoo_dir)/f'{symbol}_5m.csv'
        try:
            if base not in instruments: raise KeyError(f'Upstox NSE instrument not found for {base}')
            if not path.exists(): raise FileNotFoundError(f'Yahoo CSV missing: {path}')
            yahoo=normalize_provider_frame(pd.read_csv(path)); upstox=_fetch_upstox(session,instruments[base],start,end,token)
            reports[symbol]={'instrument_key':instruments[base],
              'yahoo':summarize_provider_frame(yahoo,session_start=session_start,session_end=session_end,post_session_end=post_session_end),
              'upstox':summarize_provider_frame(upstox,session_start=session_start,session_end=session_end,post_session_end=post_session_end),
              'timestamp_comparison':_session_timestamp_comparison(yahoo,upstox,session_start,session_end),
              'raw_timestamp_comparison':{'yahoo_only_timestamps':len(set(yahoo.timestamp)-set(upstox.timestamp)),
                'upstox_only_timestamps':len(set(upstox.timestamp)-set(yahoo.timestamp)),'matched_timestamps':len(set(yahoo.timestamp)&set(upstox.timestamp))},
              'missing_frequency_comparison':_union_missing(yahoo,upstox,session_start=session_start,session_end=session_end)}
        except Exception as exc: reports[symbol]={'error':f'{type(exc).__name__}: {exc}'}
    if any('error' in x for x in reports.values()):result['status']='COMPLETED_WITH_ERRORS'
    Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(result,indent=2,default=str),encoding='utf-8'); return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--symbols',default='RELIANCE.NS,TCS.NS,HDFCBANK.NS,ICICIBANK.NS,INFY.NS')
    p.add_argument('--start','--start-date',dest='start',required=True); p.add_argument('--end','--end-date',dest='end',required=True)
    p.add_argument('--output',default='reports/yahoo_provider_probe.json'); p.add_argument('--independent-provider',choices=['upstox'])
    p.add_argument('--yahoo-dir',default='data/research'); p.add_argument('--session-start',default='09:15')
    p.add_argument('--session-end',default='15:30'); p.add_argument('--post-session-end',default='16:00')
    a=p.parse_args(); syms=[s.strip() for s in a.symbols.split(',') if s.strip()]
    if a.independent_provider:
        r=run_independent_probe(syms,a.start,a.end,a.yahoo_dir,a.output,token=os.environ.get('UPSTOX_ACCESS_TOKEN'),
            session_start=a.session_start,session_end=a.session_end,post_session_end=a.post_session_end)
    else:r=run_probe(syms,a.start,a.end,a.output,a.session_start,a.session_end,a.post_session_end)
    print(json.dumps(r,indent=2,default=str))

if __name__=='__main__':main()
