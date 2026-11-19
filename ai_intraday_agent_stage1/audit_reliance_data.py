import pandas as pd
from database.store import CandleStore


store = CandleStore("database/market_data.sqlite")

df = store.read(
    symbol="RELIANCE",
    interval=5
)

if df.empty:
    print("No data found.")
    raise SystemExit


df = df.copy()

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# =========================================================
# ZERO-RANGE CANDLES
# =========================================================

df["range"] = df["high"] - df["low"]

df["body"] = abs(df["close"] - df["open"])

zero_range = df[df["range"] == 0]


# =========================================================
# LOW VOLUME CANDLES
# =========================================================

volume_median = df["volume"].median()

df["volume_ratio_vs_median"] = (
    df["volume"] / volume_median
)

low_volume = df[
    df["volume"] < volume_median * 0.10
]


# =========================================================
# MISSING TIMESTAMPS
# =========================================================

df["previous_timestamp"] = df["timestamp"].shift(1)

df["time_gap_minutes"] = (
    df["timestamp"] - df["previous_timestamp"]
).dt.total_seconds() / 60


# Only examine gaps within the same trading day
same_day = (
    df["timestamp"].dt.date
    == df["previous_timestamp"].dt.date
)

gaps = df[
    same_day
    & (df["time_gap_minutes"] > 5)
]


# =========================================================
# REPORT
# =========================================================

print("=" * 80)
print("RELIANCE DATA INTEGRITY AUDIT")
print("=" * 80)

print(f"Total candles: {len(df)}")
print(f"Median volume: {volume_median}")

print("\nZERO-RANGE CANDLES")
print("-" * 80)

print(f"Count: {len(zero_range)}")

if not zero_range.empty:
    print(
        zero_range[
            ["timestamp", "open", "high", "low", "close", "volume"]
        ].to_string(index=False)
    )


print("\nVERY LOW VOLUME CANDLES")
print("-" * 80)

print(f"Count: {len(low_volume)}")

if not low_volume.empty:
    print(
        low_volume[
            [
                "timestamp",
                "close",
                "volume",
                "volume_ratio_vs_median"
            ]
        ].to_string(index=False)
    )


print("\nMISSING CANDLE GAPS")
print("-" * 80)

print(f"Count: {len(gaps)}")

if not gaps.empty:
    print(
        gaps[
            [
                "previous_timestamp",
                "timestamp",
                "time_gap_minutes"
            ]
        ].to_string(index=False)
    )


print("\n" + "=" * 80)