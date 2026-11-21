import pandas as pd


# =========================================================
# INDICATOR CALCULATIONS
# =========================================================

def calculate_ema(series, period):
    """
    Calculate Exponential Moving Average.

    Uses adjust=False for recursive EMA calculation.
    Returns NaN until sufficient observations are available.
    """

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


def calculate_rsi(series, period=14):
    """
    Calculate RSI using Wilder's smoothing method.

    RSI:
        0   -> extremely weak momentum
        50  -> neutral momentum
        100 -> extremely strong momentum
    """

    delta = series.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    # Handle situations where there are no losses.
    rsi = rsi.where(avg_loss != 0, 100)

    # If both gains and losses are zero, RSI is neutral.
    rsi = rsi.where(
        ~((avg_gain == 0) & (avg_loss == 0)),
        50
    )

    return rsi


def calculate_atr(df, period=14):
    """
    Calculate Average True Range using Wilder's smoothing.

    True Range is the maximum of:

    1. High - Low
    2. abs(High - Previous Close)
    3. abs(Low - Previous Close)
    """

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr_1 = high - low

    tr_2 = (high - previous_close).abs()

    tr_3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr_1, tr_2, tr_3],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    return atr


def calculate_indicators(df):
    """
    Calculate all initial Stage 2 indicators.

    Required columns:
        timestamp
        open
        high
        low
        close
        volume

    Returns:
        A new DataFrame containing original candles
        and calculated indicators.
    """

    if df.empty:
        return df.copy()

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    result = df.copy()

    # -----------------------------------------------------
    # SORT CHRONOLOGICALLY
    # -----------------------------------------------------

    result["timestamp"] = pd.to_datetime(
        result["timestamp"]
    )

    result = result.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # ENSURE NUMERIC DATA
    # -----------------------------------------------------

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    # -----------------------------------------------------
    # MOVING AVERAGES
    # -----------------------------------------------------

    result["ema_20"] = calculate_ema(
        result["close"],
        20
    )

    result["ema_50"] = calculate_ema(
        result["close"],
        50
    )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    result["rsi_14"] = calculate_rsi(
        result["close"],
        14
    )

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    result["atr_14"] = calculate_atr(
        result,
        14
    )

    # -----------------------------------------------------
    # NORMALIZED ATR
    # -----------------------------------------------------

    result["atr_percent"] = (
        result["atr_14"] / result["close"]
    ) * 100

    # -----------------------------------------------------
    # RETURNS
    # -----------------------------------------------------

    result["return_1"] = (
        result["close"].pct_change(1) * 100
    )

    result["return_3"] = (
        result["close"].pct_change(3) * 100
    )

    result["return_5"] = (
        result["close"].pct_change(5) * 100
    )

    # -----------------------------------------------------
    # EMA 20 SLOPE (5-CANDLE LOOKBACK)
    # -----------------------------------------------------

    result["ema_20_slope_percent"] = (
        (result["ema_20"] - result["ema_20"].shift(5))
        / result["ema_20"].shift(5)
    ) * 100

    # -----------------------------------------------------
    # EMA DISTANCE
    # -----------------------------------------------------

    result["close_vs_ema20_percent"] = (
        (result["close"] - result["ema_20"])
        / result["ema_20"]
    ) * 100

    result["ema20_vs_ema50_percent"] = (
        (result["ema_20"] - result["ema_50"])
        / result["ema_50"]
    ) * 100

    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

    result["momentum_5"] = (
        result["close"] - result["close"].shift(5)
    )

    result["momentum_5_percent"] = (
        result["close"].pct_change(5) * 100
    )

    return result