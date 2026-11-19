from database.store import CandleStore
from market_intelligence.indicators import calculate_indicators
from market_intelligence.regime import classify_regime


# =========================================================
# LOAD DATA
# =========================================================

DB_PATH = "database/market_data.sqlite"

store = CandleStore(DB_PATH)

df = store.read(
    symbol="BANKNIFTY", 
    interval=5
)

if df.empty:
    print("No data found.")
    raise SystemExit


# =========================================================
# CALCULATE INDICATORS
# =========================================================

df = calculate_indicators(df)


# =========================================================
# CLASSIFY MARKET REGIME
# =========================================================

regime = classify_regime(df)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("=" * 70)
print("MARKET REGIME ANALYSIS")
print("=" * 70)

print(f"Status: {regime['status']}")

if regime["status"] != "success":
    print(regime)
    raise SystemExit


print(f"Timestamp: {regime['timestamp']}")

print("\n" + "-" * 70)
print("REGIME CLASSIFICATION")
print("-" * 70)

print(f"Trend:           {regime['trend']}")
print(f"Trend strength:  {regime['trend_strength']}")
print(f"Momentum:        {regime['momentum']}")
print(f"Volatility:      {regime['volatility']}")


# =========================================================
# INDICATORS
# =========================================================

print("\n" + "-" * 70)
print("INDICATORS")
print("-" * 70)

for key, value in regime["indicators"].items():

    print(f"{key}: {value}")


# =========================================================
# VOLATILITY BASELINE
# =========================================================

print("\n" + "-" * 70)
print("VOLATILITY BASELINE")
print("-" * 70)

print(regime["volatility_details"])


# =========================================================
# EXPLANATIONS
# =========================================================

print("\n" + "-" * 70)
print("EXPLANATIONS")
print("-" * 70)

for explanation in regime["explanations"]:

    print(f"- {explanation}")


# =========================================================
# HISTORICAL SNAPSHOT TEST
# =========================================================

print("\n" + "=" * 70)
print("RECENT REGIME HISTORY")
print("=" * 70)

print(
    f"{'Timestamp':<30}"
    f"{'Trend':<12}"
    f"{'Strength':<12}"
    f"{'Momentum':<20}"
    f"{'Volatility':<12}"
)

for i in range(10, 0, -1):

    snapshot = classify_regime(
        df.iloc[:-i]
    )

    if snapshot["status"] != "success":
        continue

    print(
        f"{str(snapshot['timestamp']):<30}"
        f"{snapshot['trend']:<12}"
        f"{snapshot['trend_strength']:<12}"
        f"{snapshot['momentum']:<20}"
        f"{snapshot['volatility']:<12}"
    )