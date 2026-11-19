import pandas as pd


# =========================================================
# MARKET REGIME CLASSIFICATION
# =========================================================

def classify_trend(latest):
    """
    Classify the current trend using:

    - Price relative to EMA 20
    - EMA 20 relative to EMA 50
    - EMA 20 slope

    Returns:
        bullish
        bearish
        sideways
    """

    close = latest["close"]
    ema_20 = latest["ema_20"]
    ema_50 = latest["ema_50"]
    ema_20_slope = latest["ema_20_slope_percent"]

    if pd.isna(ema_20_slope):
        return "unknown"

    # -----------------------------------------------------
    # BULLISH
    # -----------------------------------------------------

    if (
        close > ema_20
        and ema_20 > ema_50
        and ema_20_slope > 0
    ):
        return "bullish"

    # -----------------------------------------------------
    # BEARISH
    # -----------------------------------------------------

    if (
        close < ema_20
        and ema_20 < ema_50
        and ema_20_slope < 0
    ):
        return "bearish"

    # -----------------------------------------------------
    # SIDEWAYS / MIXED CONDITIONS
    # -----------------------------------------------------

    return "sideways"


# =========================================================
# TREND STRENGTH
# =========================================================

def classify_trend_strength(latest):
    """
    Classify trend strength using EMA separation
    and EMA 20 slope.

    These thresholds are initial research thresholds.
    They must be validated through historical testing.
    """

    ema_spread = abs(
        latest["ema20_vs_ema50_percent"]
    )

    ema_slope = abs(
        latest["ema_20_slope_percent"]
    )

    if pd.isna(ema_spread) or pd.isna(ema_slope):
        return "unknown"

    # -----------------------------------------------------
    # STRONG TREND
    # -----------------------------------------------------

    if (
        ema_spread >= 0.15
        and ema_slope >= 0.03
    ):
        return "strong"

    # -----------------------------------------------------
    # MODERATE TREND
    # -----------------------------------------------------

    if (
        ema_spread >= 0.05
        and ema_slope >= 0.01
    ):
        return "moderate"

    # -----------------------------------------------------
    # WEAK TREND
    # -----------------------------------------------------

    return "weak"


# =========================================================
# MOMENTUM CLASSIFICATION
# =========================================================

def classify_momentum(rsi):
    """
    Classify momentum using RSI.

    This is a descriptive classification,
    not a buy/sell signal.
    """

    if pd.isna(rsi):
        return "unknown"

    if rsi >= 70:
        return "strong_bullish"

    if rsi >= 55:
        return "bullish"

    if rsi > 45:
        return "neutral"

    if rsi > 30:
        return "bearish"

    return "strong_bearish"


# =========================================================
# VOLATILITY CLASSIFICATION
# =========================================================

def classify_volatility(df, minimum_observations=30):
    """
    Classify volatility relative to historical ATR%.

    Uses percentile-based classification:

        <= 25th percentile: low
        25th - 75th percentile: normal
        >= 75th percentile: high

    The latest ATR is compared against prior observations.

    Requires sufficient historical data.
    """

    if "atr_percent" not in df.columns:
        return "unknown", None

    valid = df["atr_percent"].dropna()

    if len(valid) < minimum_observations:
        return "unknown", None

    latest_atr_percent = valid.iloc[-1]

    # Exclude the latest observation from the baseline.
    historical = valid.iloc[:-1]

    if len(historical) < minimum_observations:
        return "unknown", None

    low_threshold = historical.quantile(0.25)

    high_threshold = historical.quantile(0.75)

    # -----------------------------------------------------
    # LOW VOLATILITY
    # -----------------------------------------------------

    if latest_atr_percent <= low_threshold:

        return (
            "low",
            {
                "current_atr_percent": latest_atr_percent,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold
            }
        )

    # -----------------------------------------------------
    # HIGH VOLATILITY
    # -----------------------------------------------------

    if latest_atr_percent >= high_threshold:

        return (
            "high",
            {
                "current_atr_percent": latest_atr_percent,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold
            }
        )

    # -----------------------------------------------------
    # NORMAL VOLATILITY
    # -----------------------------------------------------

    return (
        "normal",
        {
            "current_atr_percent": latest_atr_percent,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold
        }
    )


# =========================================================
# EMA SLOPE
# =========================================================

def calculate_ema_slope(df, lookback=5):
    """
    Calculate EMA 20 slope over a specified lookback.

    Example:

        Current EMA20 = 100
        EMA20 five candles ago = 99

        Slope = 1%

    This measures the direction and rate of change
    of the EMA rather than individual candle noise.
    """

    if len(df) <= lookback:
        return pd.Series(
            [float("nan")] * len(df),
            index=df.index
        )

    ema_20 = df["ema_20"]

    previous_ema = ema_20.shift(lookback)

    slope = (
        (ema_20 - previous_ema)
        / previous_ema
    ) * 100

    return slope


# =========================================================
# PREPARE REGIME DATA
# =========================================================

def prepare_regime_data(df):
    """
    Add regime-specific supporting calculations.
    """

    result = df.copy()

    result["ema_20_slope_percent"] = (
        calculate_ema_slope(
            result,
            lookback=5
        )
    )

    return result


# =========================================================
# COMPLETE REGIME CLASSIFICATION
# =========================================================

def classify_regime(df):
    """
    Generate an explainable market regime snapshot.

    Returns a dictionary containing:

    - Trend
    - Trend strength
    - Momentum
    - Volatility
    - Supporting indicator values
    - Explanations
    """

    if df.empty:
        return {
            "status": "insufficient_data"
        }

    result = prepare_regime_data(df)

    latest = result.iloc[-1]

    required_indicators = [
        "ema_20",
        "ema_50",
        "rsi_14",
        "atr_14",
        "ema_20_slope_percent"
    ]

    missing = [
        column
        for column in required_indicators
        if pd.isna(latest[column])
    ]

    if missing:

        return {
            "status": "insufficient_data",
            "missing_indicators": missing
        }

    # -----------------------------------------------------
    # CLASSIFICATIONS
    # -----------------------------------------------------

    trend = classify_trend(latest)

    trend_strength = classify_trend_strength(latest)

    momentum = classify_momentum(
        latest["rsi_14"]
    )

    volatility, volatility_details = (
        classify_volatility(result)
    )

    # -----------------------------------------------------
    # EXPLANATIONS
    # -----------------------------------------------------

    explanations = []

    if trend == "bullish":

        explanations.append(
            "Price is above EMA20, EMA20 is above EMA50, "
            "and EMA20 is rising."
        )

    elif trend == "bearish":

        explanations.append(
            "Price is below EMA20, EMA20 is below EMA50, "
            "and EMA20 is falling."
        )

    elif trend == "sideways":

        explanations.append(
            "Trend conditions are mixed and do not "
            "satisfy the bullish or bearish rules."
        )

    if momentum == "strong_bullish":

        explanations.append(
            "RSI indicates strong bullish momentum "
            "and may represent an extended move."
        )

    elif momentum == "strong_bearish":

        explanations.append(
            "RSI indicates strong bearish momentum."
        )

    elif momentum == "neutral":

        explanations.append(
            "RSI momentum is relatively neutral."
        )

    if volatility == "high":

        explanations.append(
            "ATR percentage is in the upper range "
            "of the historical baseline."
        )

    elif volatility == "low":

        explanations.append(
            "ATR percentage is in the lower range "
            "of the historical baseline."
        )

    elif volatility == "normal":

        explanations.append(
            "ATR percentage is within the normal "
            "historical range."
        )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    return {
        "status": "success",

        "timestamp": latest["timestamp"],

        "trend": trend,

        "trend_strength": trend_strength,

        "momentum": momentum,

        "volatility": volatility,

        "indicators": {
            "close": latest["close"],
            "ema_20": latest["ema_20"],
            "ema_50": latest["ema_50"],
            "ema_20_slope_percent":
                latest["ema_20_slope_percent"],
            "ema20_vs_ema50_percent":
                latest["ema20_vs_ema50_percent"],
            "rsi_14": latest["rsi_14"],
            "atr_14": latest["atr_14"],
            "atr_percent": latest["atr_percent"],
            "return_5": latest["return_5"]
        },

        "volatility_details": volatility_details,

        "explanations": explanations
    }