import sqlite3, os
from pathlib import Path
import pandas as pd

SCHEMA="""
CREATE TABLE IF NOT EXISTS candles(
 symbol TEXT NOT NULL, interval_minutes INTEGER NOT NULL, timestamp TEXT NOT NULL,
 open REAL, high REAL, low REAL, close REAL, volume REAL, source TEXT NOT NULL,
 ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(symbol, interval_minutes, timestamp)
);
CREATE TABLE IF NOT EXISTS quality_issues(
 id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, interval_minutes INTEGER,
 checked_at TEXT DEFAULT CURRENT_TIMESTAMP, row_index INTEGER, issue TEXT
);
"""
class CandleStore:
    def __init__(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.path=path
        with sqlite3.connect(path) as c: c.executescript(SCHEMA)
    def save(self,symbol,interval,df,source,issues=None):
        with sqlite3.connect(self.path) as c:
            rows=[]
            for r in df.itertuples(index=False):
                ts=pd.Timestamp(r.timestamp).isoformat()
                rows.append((symbol,interval,ts,float(r.open),float(r.high),float(r.low),float(r.close),float(r.volume),source))
            c.executemany("INSERT OR REPLACE INTO candles(symbol,interval_minutes,timestamp,open,high,low,close,volume,source) VALUES(?,?,?,?,?,?,?,?,?)",rows)
            if issues:
                c.executemany("INSERT INTO quality_issues(symbol,interval_minutes,row_index,issue) VALUES(?,?,?,?)",
                    [(symbol,interval,i.get("row"),i.get("issue")) for i in issues])
        return len(rows)
    def read(self,symbol=None,interval=None):
        q="SELECT * FROM candles WHERE 1=1"; args=[]
        if symbol: q+=" AND symbol=?"; args.append(symbol)
        if interval is not None: q+=" AND interval_minutes=?"; args.append(interval)
        q+=" ORDER BY timestamp"
        with sqlite3.connect(self.path) as c: return pd.read_sql_query(q,c,params=args)
    def count(self):
        with sqlite3.connect(self.path) as c: return c.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
