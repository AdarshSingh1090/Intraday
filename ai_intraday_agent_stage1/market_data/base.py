from abc import ABC, abstractmethod
import pandas as pd

class MarketDataProvider(ABC):
    @abstractmethod
    def get_intraday(self, instrument_key: str, interval_minutes: int) -> pd.DataFrame:
        """Return normalized OHLCV dataframe."""

    @abstractmethod
    def get_history(self, instrument_key: str, unit: str, interval: int,
                    from_date: str, to_date: str) -> pd.DataFrame:
        """Return normalized historical OHLCV dataframe."""
