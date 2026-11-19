import pandas as pd
from .base import MarketDataProvider
from .normalize import normalize_candles

class CSVProvider(MarketDataProvider):
    def __init__(self, path):
        self.path=path
        self.df=pd.read_csv(path)
        lower={c.lower():c for c in self.df.columns}
        aliases={"timestamp":["timestamp","time","datetime","date"],"open":["open"],"high":["high"],
                 "low":["low"],"close":["close"],"volume":["volume","vol"]}
        data={}
        for target,names in aliases.items():
            found=next((lower[n] for n in names if n in lower),None)
            if not found: raise ValueError(f"CSV missing required column: {target}")
            data[target]=self.df[found]
        self.df=pd.DataFrame(data)

    def get_intraday(self, instrument_key, interval_minutes):
        return normalize_candles(self.df.to_dict("records"))

    def get_history(self, instrument_key, unit, interval, from_date, to_date):
        df=normalize_candles(self.df.to_dict("records"))
        start=pd.Timestamp(from_date,tz="Asia/Kolkata")
        end=pd.Timestamp(to_date,tz="Asia/Kolkata")+pd.Timedelta(days=1)
        return df[(df.timestamp>=start)&(df.timestamp<end)].reset_index(drop=True)
