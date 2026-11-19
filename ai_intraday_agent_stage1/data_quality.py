from datetime import time
import pandas as pd


# =========================================================
# MARKET SESSION SETTINGS (INDIA)
# =========================================================

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


# =========================================================
# DATA QUALITY AUDIT
# =========================================================

def audit_candles(
    df,
    interval_minutes=5,
    freshness_hours=24
):
    """
    Audit OHLCV candles.

    Checks:
    - Null values
    - Invalid timestamps
    - Duplicate timestamps
    - Invalid OHLC relationships
    - Zero/negative volume
    - Missing candles INSIDE observed data range
    - Future timestamps
    - Data freshness

    Important:
    Missing candles are checked only between the first
    and last observed candle of each day.

    Therefore, partial datasets such as DEMO data are not
    incorrectly penalized for missing the rest of the day.
    """

    report = {}

    if df is None or df.empty:
        return {
            "status": "FAILED",
            "message": "Dataset is empty."
        }

    df = df.copy()

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    # -----------------------------------------------------
    # 1. CHECK REQUIRED COLUMNS
    # -----------------------------------------------------

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        return {
            "status": "FAILED",
            "message": (
                "Missing required columns: "
                + ", ".join(missing_columns)
            )
        }

    # -----------------------------------------------------
    # 2. CONVERT TIMESTAMPS
    # -----------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    invalid_timestamps = int(
        df["timestamp"].isna().sum()
    )

    report["invalid_timestamps"] = (
        invalid_timestamps
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    if df.empty:
        return {
            "status": "FAILED",
            "message": "No valid timestamps found."
        }

    # -----------------------------------------------------
    # 3. CONVERT NUMERIC COLUMNS
    # -----------------------------------------------------

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # -----------------------------------------------------
    # 4. NULL VALUES
    # -----------------------------------------------------

    null_values = (
        df[required_columns]
        .isna()
        .sum()
        .to_dict()
    )

    report["null_values"] = {
        key: int(value)
        for key, value in null_values.items()
    }

    rows_with_nulls = int(
        df[required_columns]
        .isna()
        .any(axis=1)
        .sum()
    )

    report["rows_with_nulls"] = (
        rows_with_nulls
    )

    # -----------------------------------------------------
    # 5. DUPLICATE TIMESTAMPS
    # -----------------------------------------------------

    duplicate_count = int(
        df["timestamp"]
        .duplicated()
        .sum()
    )

    report["duplicate_timestamps"] = (
        duplicate_count
    )

    # -----------------------------------------------------
    # 6. OHLC VALIDATION
    # -----------------------------------------------------

    valid_ohlc = (
        (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    )

    valid_ohlc = valid_ohlc.fillna(False)

    invalid_ohlc_count = int(
        (~valid_ohlc).sum()
    )

    report["invalid_ohlc_rows"] = (
        invalid_ohlc_count
    )

    # -----------------------------------------------------
    # 7. VOLUME VALIDATION
    # -----------------------------------------------------

    zero_volume_count = int(
        (df["volume"] == 0).sum()
    )

    negative_volume_count = int(
        (df["volume"] < 0).sum()
    )

    report["zero_volume_candles"] = (
        zero_volume_count
    )

    report["negative_volume_candles"] = (
        negative_volume_count
    )

    # -----------------------------------------------------
    # 8. FUTURE TIMESTAMP CHECK
    # -----------------------------------------------------

    latest_timestamp = df["timestamp"].max()

    if latest_timestamp.tz is not None:

        now = pd.Timestamp.now(
            tz=latest_timestamp.tz
        )

    else:

        now = pd.Timestamp.now()

    future_tolerance_minutes = 5

    future_cutoff = (
        now
        + pd.Timedelta(
            minutes=future_tolerance_minutes
        )
    )

    future_rows = df[
        df["timestamp"] > future_cutoff
    ]

    future_timestamp_count = int(
        len(future_rows)
    )

    report["future_timestamps"] = (
        future_timestamp_count
    )

    report["future_timestamp_examples"] = [
        str(timestamp)
        for timestamp in future_rows[
            "timestamp"
        ].head(5)
    ]

    # -----------------------------------------------------
    # 9. DATA FRESHNESS
    # -----------------------------------------------------

    age = now - latest_timestamp

    age_hours = (
        age.total_seconds() / 3600
    )

    report["latest_timestamp"] = str(
        latest_timestamp
    )

    report["data_age_hours"] = round(
        age_hours,
        2
    )

    report["freshness_threshold_hours"] = (
        freshness_hours
    )

    report["is_stale"] = (
        age_hours > freshness_hours
    )

    # -----------------------------------------------------
    # 10. MISSING CANDLES
    # -----------------------------------------------------
    #
    # Only inspect the range between the first and last
    # observed candle for each day.
    #
    # This prevents partial/demo datasets from being
    # penalized for candles that were never expected
    # to be included in the dataset.
    # -----------------------------------------------------

    df["trading_date"] = (
        df["timestamp"].dt.date
    )

    missing_candles = {}

    total_missing = 0

    for trading_date, day_df in df.groupby(
        "trading_date"
    ):

        timestamps = (
            day_df["timestamp"]
            .drop_duplicates()
            .sort_values()
        )

        if len(timestamps) < 2:

            missing_candles[
                str(trading_date)
            ] = {
                "observed_count": len(timestamps),
                "missing_count": 0,
                "missing_timestamps": [],
                "coverage_start": str(
                    timestamps.min()
                ),
                "coverage_end": str(
                    timestamps.max()
                )
            }

            continue

        first_timestamp = timestamps.min()

        last_timestamp = timestamps.max()

        # Expected timestamps only within observed range.
        expected = pd.date_range(
            start=first_timestamp,
            end=last_timestamp,
            freq=f"{interval_minutes}min"
        )

        actual = pd.DatetimeIndex(
            timestamps
        )

        missing = expected.difference(
            actual
        )

        missing_count = len(missing)

        total_missing += missing_count

        missing_candles[
            str(trading_date)
        ] = {
            "observed_count": len(actual),
                "missing_count": missing_count,
            "missing_timestamps": [
                str(timestamp)
                for timestamp in missing
            ],
            "coverage_start": str(
                first_timestamp
            ),
            "coverage_end": str(
                last_timestamp
            )
        }

    report["missing_candles_by_day"] = (
        missing_candles
    )

    report["total_missing_candles"] = (
        total_missing
    )

    # -----------------------------------------------------
    # 11. DATASET STATISTICS
    # -----------------------------------------------------

    total_rows = len(df)

    report["total_rows"] = total_rows

    valid_rows = (
        df[required_columns]
        .notna()
        .all(axis=1)
        .sum()
    )

    report["valid_rows"] = int(
        valid_rows
    )

    if total_rows > 0:

        valid_percentage = (
            valid_rows / total_rows
        ) * 100

    else:

        valid_percentage = 0

    report["valid_row_percentage"] = round(
        valid_percentage,
        2
    )

    # -----------------------------------------------------
    # 12. QUALITY SCORE
    # -----------------------------------------------------
    #
    # This is a transparent heuristic:
    # percentage of rows without nulls or invalid OHLC.
    #
    # It is NOT a trading performance score.
    # -----------------------------------------------------

    bad_rows = (
        df[required_columns]
        .isna()
        .any(axis=1)
        | (~valid_ohlc)
    )

    bad_rows_count = int(
        bad_rows.sum()
    )

    if total_rows > 0:

        quality_score = (
            100
            * (
                total_rows - bad_rows_count
            )
            / total_rows
        )

    else:

        quality_score = 0

    report["quality_score"] = round(
        max(0, quality_score),
        2
    )

    # -----------------------------------------------------
    # 13. STATUS
    # -----------------------------------------------------

    if (
        invalid_timestamps > 0
        or invalid_ohlc_count > 0
        or negative_volume_count > 0
        or future_timestamp_count > 0
    ):

        status = "FAILED"

    elif (
        duplicate_count > 0
        or total_missing > 0
        or zero_volume_count > 0
        or report["is_stale"]
        or rows_with_nulls > 0
    ):

        status = "WARNING"

    else:

        status = "PASSED"

    report["status"] = status

    return report


# =========================================================
# REPORT PRINTER
# =========================================================

def print_report(report):

    print("\n" + "=" * 55)
    print("           MARKET DATA QUALITY REPORT")
    print("=" * 55)

    if report.get("message"):

        print(report["message"])

        return

    print(
        f"\nStatus: {report['status']}"
    )

    print(
        f"Quality Score: "
        f"{report['quality_score']}/100"
    )

    print(
        f"Valid Rows: "
        f"{report['valid_rows']}/"
        f"{report['total_rows']}"
    )

    print(
        f"Valid Row Percentage: "
        f"{report['valid_row_percentage']}%"
    )

    # -----------------------------------------------------
    # BASIC VALIDATION
    # -----------------------------------------------------

    print("\n--- BASIC VALIDATION ---")

    print(
        f"Duplicate timestamps: "
        f"{report['duplicate_timestamps']}"
    )

    print(
        f"Invalid timestamps: "
        f"{report['invalid_timestamps']}"
    )

    print(
        f"Invalid OHLC rows: "
        f"{report['invalid_ohlc_rows']}"
    )

    print(
        f"Rows containing nulls: "
        f"{report['rows_with_nulls']}"
    )

    print(
        f"Zero-volume candles: "
        f"{report['zero_volume_candles']}"
    )

    print(
        f"Negative-volume candles: "
        f"{report['negative_volume_candles']}"
    )

    print(
        f"Future timestamps: "
        f"{report['future_timestamps']}"
    )

    # -----------------------------------------------------
    # MISSING CANDLES
    # -----------------------------------------------------

    print("\n--- CANDLE GAPS ---")

    print(
        f"Missing candles within observed ranges: "
        f"{report['total_missing_candles']}"
    )

    # -----------------------------------------------------
    # FRESHNESS
    # -----------------------------------------------------

    print("\n--- FRESHNESS ---")

    print(
        f"Latest candle: "
        f"{report['latest_timestamp']}"
    )

    print(
        f"Data age: "
        f"{report['data_age_hours']} hours"
    )

    print(
        f"Stale: "
        f"{report['is_stale']}"
    )

    # -----------------------------------------------------
    # DAILY SUMMARY
    # -----------------------------------------------------

    print("\n--- DAILY COVERAGE ---")

    for date, data in report[
        "missing_candles_by_day"
    ].items():

        print(
            f"{date}: "
            f"{data['observed_count']} observed candles | "
            f"Missing within range: "
            f"{data['missing_count']}"
        )

        print(
            f"  Coverage: "
            f"{data['coverage_start']} "
            f"-> {data['coverage_end']}"
        )

        if data["missing_timestamps"]:

            print(
                "  Missing timestamps:"
            )

            for timestamp in data[
                "missing_timestamps"
            ]:

                print(
                    f"    {timestamp}"
                )

    # -----------------------------------------------------
    # FUTURE TIMESTAMP DETAILS
    # -----------------------------------------------------

    if report[
        "future_timestamp_examples"
    ]:

        print(
            "\n--- FUTURE TIMESTAMP EXAMPLES ---"
        )

        for timestamp in report[
            "future_timestamp_examples"
        ]:

            print(
                f"  {timestamp}"
            )

    print("\n" + "=" * 55)