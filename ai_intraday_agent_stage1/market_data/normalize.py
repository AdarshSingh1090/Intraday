import pandas as pd

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

def normalize_candles(rows):
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[COLUMNS].sort_values("timestamp").reset_index(drop=True)

def upstox_candles(payload):
    candles = payload.get("data", {}).get("candles", [])
    rows=[]
    for c in candles:
        # Upstox candle format: timestamp, open, high, low, close, volume, optional OI
        if len(c) < 6: continue
        rows.append([c[0], c[1], c[2], c[3], c[4], c[5]])
    return normalize_candles(rows)
