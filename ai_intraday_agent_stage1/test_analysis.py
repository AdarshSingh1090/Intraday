from database.store import CandleStore
from market_intelligence.analysis import analyze_market


# =========================================================
# LOAD MARKET DATA
# =========================================================

store = CandleStore("database/market_data.sqlite")

df = store.read(
    symbol="NIFTY50",
    interval=5
)

if df.empty:
    print("No market data available.")
    raise SystemExit


# =========================================================
# UNIFIED ANALYSIS
# =========================================================

result = analyze_market(df)


# =========================================================
# DISPLAY
# =========================================================

print("=" * 70)
print("UNIFIED MARKET ANALYSIS")
print("=" * 70)

print(f"Status: {result['status']}")

if result["status"] != "success":
    print(result)
    raise SystemExit


regime = result["regime"]

print("\nMARKET REGIME")
print("-" * 70)

print(f"Trend:       {regime['trend']}")
print(f"Strength:    {regime['trend_strength']}")
print(f"Momentum:    {regime['momentum']}")
print(f"Volatility:  {regime['volatility']}")

print("\nLATEST INDICATORS")
print("-" * 70)

latest = result["indicators"].iloc[-1]

for column in [
    "close",
    "ema_20",
    "ema_50",
    "rsi_14",
    "atr_14",
    "atr_percent",
    "return_5",
    "momentum_5"
]:
    print(f"{column}: {latest[column]}")

print("\nEXPLANATION")
print("-" * 70)

for explanation in regime["explanations"]:
    print(f"- {explanation}")