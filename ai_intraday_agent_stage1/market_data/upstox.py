import os
import requests
from .base import MarketDataProvider
from .normalize import upstox_candles

BASE = "https://api.upstox.com"

class UpstoxProvider(MarketDataProvider):
    def __init__(self, token=None):
        self.token = token or os.getenv("UPSTOX_ACCESS_TOKEN")
        if not self.token:
            raise RuntimeError("UPSTOX_ACCESS_TOKEN missing. Set it in .env.")
        self.session=requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        })

    def _get(self, url):
        r=self.session.get(url,timeout=30)
        if r.status_code in (401,403):
            raise RuntimeError(f"Upstox authentication/permission error ({r.status_code}). Refresh/check token.")
        if not r.ok:
            raise RuntimeError(f"Upstox API error {r.status_code}: {r.text[:500]}")
        return r.json()

    def get_intraday(self, instrument_key, interval_minutes):
        url=f"{BASE}/v3/historical-candle/intraday/{instrument_key}/minutes/{interval_minutes}"
        return upstox_candles(self._get(url))

    def get_history(self, instrument_key, unit, interval, from_date, to_date):
        url=f"{BASE}/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"
        return upstox_candles(self._get(url))
