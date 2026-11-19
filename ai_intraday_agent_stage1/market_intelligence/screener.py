import pandas as pd


def calculate_screening_metrics(df):

    if df is None or df.empty:
        return {
            "status": "no_data",
            "metrics": None
        }

    result = df.copy()

    result = result.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------------------------------
    # BASIC PRICE METRICS
    # -----------------------------------------------------

    result["candle_range"] = result["high"] - result["low"]

    result["candle_range_percent"] = (
        result["candle_range"] / result["close"]
    ) * 100

    result["body_size"] = abs(
        result["close"] - result["open"]
    )

    result["body_size_percent"] = (
        result["body_size"] / result["close"]
    ) * 100

    # -----------------------------------------------------
    # VOLUME METRICS
    # -----------------------------------------------------

    result["volume_sma_20"] = (
        result["volume"]
        .rolling(window=20, min_periods=20)
        .mean()
    )

    result["relative_volume"] = (
        result["volume"] / result["volume_sma_20"]
    )

    # -----------------------------------------------------
    # TRADED VALUE / LIQUIDITY PROXY
    # -----------------------------------------------------

    result["traded_value"] = (
        result["close"] * result["volume"]
    )

    result["average_traded_value_20"] = (
        result["traded_value"]
        .rolling(window=20, min_periods=20)
        .mean()
    )

    # -----------------------------------------------------
    # SHORT-TERM PRICE MOVEMENT
    # -----------------------------------------------------

    result["return_1"] = (
        result["close"].pct_change() * 100
    )

    result["return_3"] = (
        result["close"].pct_change(3) * 100
    )

    result["return_10"] = (
        result["close"].pct_change(10) * 100
    )

    # -----------------------------------------------------
    # LATEST METRICS
    # -----------------------------------------------------

    latest = result.iloc[-1]

    metrics = {
        "timestamp": latest["timestamp"],
        "close": latest["close"],
        "volume": latest["volume"],
        "volume_sma_20": latest["volume_sma_20"],
        "relative_volume": latest["relative_volume"],
        "traded_value": latest["traded_value"],
        "average_traded_value_20": latest[
            "average_traded_value_20"
        ],
        "candle_range_percent": latest[
            "candle_range_percent"
        ],
        "body_size_percent": latest[
            "body_size_percent"
        ],
        "return_1": latest["return_1"],
        "return_3": latest["return_3"],
        "return_10": latest["return_10"],
    }

    return {
        "status": "success",
        "metrics": metrics,
        "data": result
    }