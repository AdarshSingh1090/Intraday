from market_intelligence.indicators import calculate_indicators
from market_intelligence.regime import classify_regime


def analyze_market(df):

    if df is None or df.empty:
        return {
            "status": "no_data",
            "indicators": None,
            "regime": None
        }

    # Step 1: Calculate technical indicators
    indicator_df = calculate_indicators(df)

    # Step 2: Classify the market regime
    regime = classify_regime(indicator_df)

    # Step 3: Return consolidated analysis
    return {
        "status": regime["status"],
        "indicators": indicator_df,
        "regime": regime
    }