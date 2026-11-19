"""Simple, transparent comparison baselines for backtest periods."""
from __future__ import annotations
import math
import pandas as pd


def benchmark_summary(df, starting_equity=8900.0, slippage_bps=5.0,
                     fee_rate=0.0003, flat_fee=0.0):
    if df is None or df.empty:
        return {"status":"NO_DATA","benchmarks":{}}
    work=df.copy()
    work["timestamp"]=pd.to_datetime(work["timestamp"],errors="coerce")
    for col in ("open","high","low","close"):
        work[col]=pd.to_numeric(work[col],errors="coerce")
    work=work.dropna(subset=["timestamp","open","close"]).sort_values("timestamp")
    if work.empty or starting_equity<=0:
        return {"status":"NO_DATA","benchmarks":{}}
    slip=float(slippage_bps)/10000
    # Buy-and-hold: invest starting equity at first open and mark to last close.
    first=float(work.iloc[0].open)
    last=float(work.iloc[-1].close)
    entry=first*(1+slip)
    exit=last*(1-slip)
    shares=starting_equity/entry
    gross=(exit-entry)*shares
    fees=(entry*shares+exit*shares)*fee_rate+2*flat_fee
    net=gross-fees
    buy_hold={"net_pnl":round(net,2),"gross_pnl":round(gross,2),
              "fees":round(fees,2),"return_pct":round(net/starting_equity*100,4),
              "ending_equity":round(starting_equity+net,2),
              "start_timestamp":str(work.iloc[0].timestamp),"end_timestamp":str(work.iloc[-1].timestamp)}
    # Daily open-close baseline: deploy full equity each session, then compound.
    equity=float(starting_equity)
    daily=[]
    for day,group in work.groupby(work["timestamp"].dt.date,sort=True):
        group=group.sort_values("timestamp")
        open_px=float(group.iloc[0].open)*(1+slip)
        close_px=float(group.iloc[-1].close)*(1-slip)
        gross_day=(close_px-open_px)*(equity/open_px)
        fees_day=(equity+equity*(close_px/open_px))*fee_rate+2*flat_fee
        net_day=gross_day-fees_day
        equity+=net_day
        daily.append({"date":str(day),"net_pnl":round(net_day,2),"equity":round(equity,2)})
    intraday={"net_pnl":round(equity-starting_equity,2),
              "return_pct":round((equity/starting_equity-1)*100,4),
              "ending_equity":round(equity,2),"daily_results":daily}
    return {"status":"COMPLETED","benchmarks":{"no_trade":{"net_pnl":0.0,"return_pct":0.0,"ending_equity":round(starting_equity,2)},
                "buy_and_hold":buy_hold,"daily_open_close":intraday},
            "assumptions":"Fractional shares; same estimated slippage and fee assumptions; benchmarks are simplified and not investment recommendations."}
