import pandas as pd

from market_intelligence.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_atr,
    calculate_indicators
)


# =========================================================
# TEST 1: INSUFFICIENT DATA
# =========================================================

print("=" * 60)
print("TEST 1: INSUFFICIENT DATA")
print("=" * 60)

short_series = pd.Series(
    [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
)

ema_20 = calculate_ema(short_series, 20)

print(f"EMA 20: {ema_20.tolist()}")

assert ema_20.isna().all()

print("PASS: EMA correctly returns NaN with insufficient data")


# =========================================================
# TEST 2: CONSTANT PRICE
# =========================================================

print("\n")
print("=" * 60)
print("TEST 2: CONSTANT PRICE")
print("=" * 60)

prices = pd.Series([100.0] * 30)

rsi = calculate_rsi(prices, 14)

print(f"Latest RSI: {rsi.iloc[-1]}")

assert rsi.iloc[-1] == 50

print("PASS: RSI correctly identifies no price movement as neutral")


# =========================================================
# TEST 3: CONTINUOUS PRICE INCREASE
# =========================================================

print("\n")
print("=" * 60)
print("TEST 3: CONTINUOUS PRICE INCREASE")
print("=" * 60)

prices = pd.Series(
    [100 + i for i in range(30)]
)

rsi = calculate_rsi(prices, 14)

print(f"Latest RSI: {rsi.iloc[-1]}")

assert rsi.iloc[-1] > 90

print("PASS: RSI correctly identifies strong upward momentum")


# =========================================================
# TEST 4: CONTINUOUS PRICE DECREASE
# =========================================================

print("\n")
print("=" * 60)
print("TEST 4: CONTINUOUS PRICE DECREASE")
print("=" * 60)

prices = pd.Series(
    [100 - i for i in range(30)]
)

rsi = calculate_rsi(prices, 14)

print(f"Latest RSI: {rsi.iloc[-1]}")

assert rsi.iloc[-1] < 10

print("PASS: RSI correctly identifies strong downward momentum")


# =========================================================
# TEST 5: ATR ON CONSTANT PRICE
# =========================================================

print("\n")
print("=" * 60)
print("TEST 5: ATR ON CONSTANT PRICE")
print("=" * 60)

flat_df = pd.DataFrame({
    "high": [100.0] * 30,
    "low": [100.0] * 30,
    "close": [100.0] * 30
})

atr = calculate_atr(flat_df, 14)

print(f"Latest ATR: {atr.iloc[-1]}")

assert atr.iloc[-1] == 0

print("PASS: ATR correctly returns zero for no price movement")


# =========================================================
# TEST 6: INPUT DATA IS NOT MODIFIED
# =========================================================

print("\n")
print("=" * 60)
print("TEST 6: INPUT DATA IMMUTABILITY")
print("=" * 60)

original = pd.DataFrame({
    "timestamp": pd.date_range(
        "2026-01-01",
        periods=60,
        freq="5min"
    ),
    "open": [100.0] * 60,
    "high": [101.0] * 60,
    "low": [99.0] * 60,
    "close": [100.0] * 60,
    "volume": [1000] * 60
})

original_copy = original.copy(deep=True)

result = calculate_indicators(original)

assert original.equals(original_copy)

print("PASS: Original input DataFrame remains unchanged")


# =========================================================
# FINAL RESULT
# =========================================================

print("\n")
print("=" * 60)
print("ALL EDGE-CASE TESTS COMPLETED")
print("=" * 60)