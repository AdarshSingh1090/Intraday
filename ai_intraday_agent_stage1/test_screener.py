from database.store import CandleStore
from market_intelligence.indicators import calculate_indicators
from market_intelligence.screener import calculate_screening_metrics


# =========================================================
# LOAD DATA
# =========================================================

store = CandleStore("database/market_data.sqlite")

df = store.read(
    symbol="RELIANCE",
    interval=5
)

if df.empty:
    print("No data available.")
    raise SystemExit


# =========================================================
# CALCULATE INDICATORS
# =========================================================

df = calculate_indicators(df)


# =========================================================
# CALCULATE SCREENING METRICS
# =========================================================

result = calculate_screening_metrics(df)

if result["status"] != "success":
    print(result)
    raise SystemExit


metrics = result["metrics"]


# =========================================================
# DISPLAY
# =========================================================

print("=" * 75)
print("STOCK SCREENING METRICS")
print("=" * 75)

print(f"Symbol: RELIANCE")
print(f"Timestamp: {metrics['timestamp']}")

print("\nPRICE")
print("-" * 75)

print(f"Close: {metrics['close']}")

print("\nVOLUME & LIQUIDITY")
print("-" * 75)

print(f"Current volume: {metrics['volume']}")
print(f"20-candle average volume: {metrics['volume_sma_20']}")
print(f"Relative volume: {metrics['relative_volume']}")
print(f"Current traded value: {metrics['traded_value']}")
print(
    f"20-candle average traded value: "
    f"{metrics['average_traded_value_20']}"
)

print("\nCANDLE CHARACTERISTICS")
print("-" * 75)

print(f"Candle range %: {metrics['candle_range_percent']}")
print(f"Candle body %: {metrics['body_size_percent']}")

print("\nPRICE RETURNS")
print("-" * 75)

print(f"1-candle return: {metrics['return_1']}%")
print(f"3-candle return: {metrics['return_3']}%")
print(f"10-candle return: {metrics['return_10']}%")

print("\n" + "=" * 75)