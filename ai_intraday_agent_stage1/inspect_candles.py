from database.store import CandleStore

store = CandleStore("database/market_data.sqlite")

df = store.read(
    symbol="RELIANCE",
    interval=5
)

print("=" * 100)
print("LAST 10 RELIANCE CANDLES")
print("=" * 100)

columns = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume"
]

print(df[columns].tail(10).to_string(index=False))

print("\n" + "=" * 100)
print("OHLC VALIDATION")
print("=" * 100)

latest = df.iloc[-1]

print(f"Timestamp: {latest['timestamp']}")
print(f"Open: {latest['open']}")
print(f"High: {latest['high']}")
print(f"Low: {latest['low']}")
print(f"Close: {latest['close']}")
print(f"Volume: {latest['volume']}")

print("\nChecks:")

print(f"High >= Low: {latest['high'] >= latest['low']}")
print(f"High >= Open: {latest['high'] >= latest['open']}")
print(f"High >= Close: {latest['high'] >= latest['close']}")
print(f"Low <= Open: {latest['low'] <= latest['open']}")
print(f"Low <= Close: {latest['low'] <= latest['close']}")

print(
    f"Zero-range candle: "
    f"{latest['high'] == latest['low']}"
)

print(
    f"Zero-body candle: "
    f"{latest['open'] == latest['close']}"
)