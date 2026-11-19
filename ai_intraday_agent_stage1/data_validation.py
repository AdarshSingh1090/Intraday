import pandas as pd

def validate(df):
    issues=[]
    if df.empty: return df, [{"row":None,"issue":"empty dataset"}]
    required=["timestamp","open","high","low","close","volume"]
    for c in required:
        if c not in df.columns: issues.append({"row":None,"issue":f"missing column {c}"})
    if issues: return pd.DataFrame(), issues
    work=df.copy()
    for c in ["open","high","low","close","volume"]:
        work[c]=pd.to_numeric(work[c],errors="coerce")
    work["timestamp"]=pd.to_datetime(work["timestamp"],errors="coerce")
    bad=work[work[required].isna().any(axis=1)]
    for i in bad.index: issues.append({"row":int(i),"issue":"null/invalid field"})
    work=work.dropna(subset=required)
    bad_price=work[(work[["open","high","low","close"]]<=0).any(axis=1)]
    for i in bad_price.index: issues.append({"row":int(i),"issue":"non-positive price"})
    work=work[~work.index.isin(bad_price.index)]
    bad_vol=work[work.volume<0]
    for i in bad_vol.index: issues.append({"row":int(i),"issue":"negative volume"})
    work=work[~work.index.isin(bad_vol.index)]
    bad_ohlc=work[(work.high < work[["open","close","low"]].max(axis=1)) |
                   (work.low > work[["open","close","high"]].min(axis=1))]
    for i in bad_ohlc.index: issues.append({"row":int(i),"issue":"OHLC relationship invalid"})
    work=work[~work.index.isin(bad_ohlc.index)]
    dup=work[work.timestamp.duplicated(keep="last")]
    for i in dup.index: issues.append({"row":int(i),"issue":"duplicate timestamp"})
    work=work.drop_duplicates("timestamp",keep="last").sort_values("timestamp").reset_index(drop=True)
    return work,issues
