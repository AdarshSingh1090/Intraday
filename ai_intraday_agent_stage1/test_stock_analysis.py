from database.store import CandleStore
from market_intelligence.analysis import analyze_market


store = CandleStore("database/market_data.sqlite")

symbols = ["RELIANCE"]

for symbol in symbols:

    print("\n" + "=" * 80)
    print(f"ANALYSIS: {symbol}")
    print("=" * 80)

    df = store.read(
        symbol=symbol,
        interval=5
    )

    if df.empty:
        print("No data available.")
        continue

    print(f"Candles available: {len(df)}")
    print(f"First candle: {df['timestamp'].iloc[0]}")
    print(f"Latest candle: {df['timestamp'].iloc[-1]}")

    result = analyze_market(df)

    if result["status"] != "success":
        print(f"Analysis failed: {result['status']}")
        continue

    regime = result["regime"]

    print("\nMARKET REGIME")
    print(f"Trend: {regime['trend']}")
    print(f"Trend strength: {regime['trend_strength']}")
    print(f"Momentum: {regime['momentum']}")
    print(f"Volatility: {regime['volatility']}")

    latest = result["indicators"].iloc[-1]

    print("\nINDICATORS")
    print(f"Close: {latest['close']}")
    print(f"RSI: {latest['rsi_14']}")
    print(f"ATR%: {latest['atr_percent']}")
    print(f"5-candle return: {latest['return_5']}%")
    print(f"5-candle momentum: {latest['momentum_5_percent']}%")

    print("\nEXPLANATION")

    for explanation in regime["explanations"]:
        print(f"- {explanation}")