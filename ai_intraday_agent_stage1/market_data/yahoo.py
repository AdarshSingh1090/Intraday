import yfinance as yf
import pandas as pd

from .base import MarketDataProvider
from .normalize import normalize_candles


class YahooProvider(MarketDataProvider):

    def get_intraday(self, instrument_key: str, interval_minutes: int) -> pd.DataFrame:
        ticker = yf.Ticker(instrument_key)

        interval_map = {
            1: "1m",
            2: "2m",
            5: "5m",
            15: "15m",
            30: "30m",
            60: "60m",
        }

        interval = interval_map.get(interval_minutes)

        if interval is None:
            raise ValueError(
                f"Yahoo Finance does not support this intraday interval: "
                f"{interval_minutes} minutes"
            )

        df = ticker.history(
            period="5d",
            interval=interval,
            auto_adjust=False
        )

        if df.empty:
            raise ValueError(
                f"No intraday data returned for {instrument_key}"
            )

        df = df.reset_index()

        timestamp_column = "Datetime" if "Datetime" in df.columns else "Date"

        rows = []

        for _, row in df.iterrows():
            rows.append([
                row[timestamp_column],
                row["Open"],
                row["High"],
                row["Low"],
                row["Close"],
                row["Volume"]
            ])

        return normalize_candles(rows)

    def get_history(
        self,
        instrument_key: str,
        unit: str,
        interval: int,
        from_date: str,
        to_date: str
    ) -> pd.DataFrame:

        ticker = yf.Ticker(instrument_key)

        interval_map = {
            "daily": "1d",
            "weekly": "1wk",
            "monthly": "1mo",
        }

        yahoo_interval = interval_map.get(unit.lower())

        if yahoo_interval is None:
            raise ValueError(
                "Supported history units: daily, weekly, monthly"
            )

        df = ticker.history(
            start=from_date,
            end=to_date,
            interval=yahoo_interval,
            auto_adjust=False
        )

        if df.empty:
            raise ValueError(
                f"No historical data returned for {instrument_key}"
            )

        df = df.reset_index()

        timestamp_column = "Date"

        rows = []

        for _, row in df.iterrows():
            rows.append([
                row[timestamp_column],
                row["Open"],
                row["High"],
                row["Low"],
                row["Close"],
                row["Volume"]
            ])

        return normalize_candles(rows)