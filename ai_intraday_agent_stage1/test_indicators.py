from database.store import CandleStore
from market_intelligence.indicators import calculate_indicators


# =========================================================
# LOAD EXISTING CANDLE DATA
# =========================================================

DB_PATH = "database/market_data.sqlite"

store = CandleStore(DB_PATH)

df = store.read(
    symbol="NIFTY50",
    interval=5
)


# =========================================================
# CHECK DATA
# =========================================================

if df.empty:
    print("No candle data found.")
    raise SystemExit


print("=" * 60)
print("RAW DATA")
print("=" * 60)

print(f"Rows loaded: {len(df)}")
print(f"Latest candle: {df['timestamp'].max()}")


# =========================================================
# CALCULATE INDICATORS
# =========================================================

result = calculate_indicators(df)


# =========================================================
# DISPLAY RESULTS
# =========================================================

columns = [
    "timestamp",
    "close",
    "ema_20",
    "ema_50",
    "rsi_14",
    "atr_14",
    "atr_percent",
    "return_1",
    "return_3",
    "return_5",
    "momentum_5_percent"
]


print("\n")
print("=" * 60)
print("CALCULATED INDICATORS")
print("=" * 60)

print(
    result[columns]
    .tail(20)
    .to_string(index=False)
)


# =========================================================
# CHECK INDICATOR AVAILABILITY
# =========================================================

print("\n")
print("=" * 60)
print("INDICATOR AVAILABILITY")
print("=" * 60)

indicator_columns = [
    "ema_20",
    "ema_50",
    "rsi_14",
    "atr_14"
]

for column in indicator_columns:

    valid_count = result[column].notna().sum()

    print(
        f"{column}: "
        f"{valid_count}/{len(result)} valid values"
    )


# =========================================================
# LATEST VALUES
# =========================================================

latest = result.iloc[-1]

print("\n")
print("=" * 60)
print("LATEST MARKET VALUES")
print("=" * 60)

for column in indicator_columns:

    print(
        f"{column}: {latest[column]}"
    )

print(
    f"Close: {latest['close']}"
)

print(
    f"ATR %: {latest['atr_percent']}"
)

print(
    f"5-candle return: "
    f"{latest['return_5']}%"
)
