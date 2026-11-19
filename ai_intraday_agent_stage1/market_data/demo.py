import pandas as pd
import numpy as np
from datetime import time

from .base import MarketDataProvider


class DemoProvider(MarketDataProvider):
    """
    Synthetic candles for infrastructure testing only.
    Generated during a previous weekday's Indian market session.
    NOT suitable for trading or strategy evaluation.
    """

    def __init__(self, seed=7):
        self.seed = seed

    def _make(self, n=120):
        rng = np.random.default_rng(self.seed)

        # Use the most recent weekday strictly before today.
        today = pd.Timestamp.now(tz="Asia/Kolkata").date()
        session_date = pd.Timestamp(today) - pd.Timedelta(days=1)

        while session_date.weekday() >= 5:
            session_date -= pd.Timedelta(days=1)

        # Start at 09:15 IST and generate 120 one-minute candles.
        start = pd.Timestamp(
            f"{session_date.date()} 09:15",
            tz="Asia/Kolkata"
        )

        ts = pd.date_range(
            start=start,
            periods=n,
            freq="min"
        )

        price = 100 + np.cumsum(rng.normal(0, 0.12, n))

        open_ = np.r_[price[0], price[:-1]]

        high = (
            np.maximum(open_, price)
            + rng.uniform(0, 0.08, n)
        )

        low = (
            np.minimum(open_, price)
            - rng.uniform(0, 0.08, n)
        )

        volume = rng.integers(100, 5000, n)

        return pd.DataFrame({
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": price,
            "volume": volume
        })

    def get_intraday(self, instrument_key, interval_minutes):
        df = self._make()

        if interval_minutes > 1:
            df = (
                df.set_index("timestamp")
                .resample(f"{interval_minutes}min")
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                })
                .dropna()
                .reset_index()
            )

        return df

    def get_history(
        self,
        instrument_key,
        unit,
        interval,
        from_date,
        to_date
    ):
        return self._make()