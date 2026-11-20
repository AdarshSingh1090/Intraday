"""Multi-symbol historical data acquisition and audit for Stage 8 research.

This module is research-only: it downloads and stores candles, but does not
place orders or generate trade signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Callable

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
YAHOO_INTERVALS = {1: "1m", 2: "2m", 5: "5m", 15: "15m", 30: "30m", 60: "60m"}


@dataclass(frozen=True)
class ResearchFetchConfig:
    interval_minutes: int = 5
    start_date: str = ""
    end_date: str = ""
    market_timezone: str = "Asia/Kolkata"


def normalize_yahoo_frame(frame: pd.DataFrame, timezone: str = "Asia/Kolkata") -> pd.DataFrame:
    """Normalize a yfinance OHLCV frame to the project's canonical columns."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "timestamp" not in data.columns:
        data = data.reset_index()
    aliases = {str(col).lower(): col for col in data.columns}
    time_col = next((aliases[k] for k in ("timestamp", "datetime", "date") if k in aliases), None)
    if time_col is None:
        raise ValueError("Historical provider data is missing a timestamp/date column.")

    rename = {}
    for col in ("open", "high", "low", "close", "volume"):
        if col not in aliases:
            raise ValueError(f"Historical provider data is missing {col}.")
        rename[aliases[col]] = col
    rename[time_col] = "timestamp"
    data = data.rename(columns=rename)[OHLCV_COLUMNS]
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    if data["timestamp"].dt.tz is None:
        data["timestamp"] = data["timestamp"].dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    else:
        data["timestamp"] = data["timestamp"].dt.tz_convert(timezone)
    for col in OHLCV_COLUMNS[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.sort_values("timestamp").reset_index(drop=True)


def audit_research_candles(frame: pd.DataFrame) -> dict:
    """Strict, transparent data audit; malformed data is not silently repaired."""
    issues = []
    if frame is None or frame.empty:
        return {"eligible": False, "rows": 0, "issues": ["empty dataset"], "coverage": {}}
    missing = set(OHLCV_COLUMNS) - set(frame.columns)
    if missing:
        return {"eligible": False, "rows": len(frame), "issues": [f"missing columns: {sorted(missing)}"], "coverage": {}}

    data = frame.copy()
    ts = pd.to_datetime(data["timestamp"], errors="coerce")
    if ts.isna().any():
        issues.append(f"invalid timestamps: {int(ts.isna().sum())}")
    if ts.duplicated().any():
        issues.append(f"duplicate timestamps: {int(ts.duplicated().sum())}")
    if not ts.is_monotonic_increasing:
        issues.append("timestamps are not sorted ascending")
    numeric = data[OHLCV_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        issues.append(f"non-numeric/null OHLCV values: {int(numeric.isna().sum().sum())}")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        issues.append("non-positive prices")
    if (numeric["volume"] < 0).any():
        issues.append("negative volume")
    if not numeric.empty:
        inconsistent = (
            numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)
        ) | (numeric["low"] > numeric[["open", "close"]].min(axis=1))
        if inconsistent.any():
            issues.append(f"inconsistent OHLC rows: {int(inconsistent.sum())}")

    coverage = {}
    if ts.notna().any():
        dates = ts.dt.date
        coverage = {
            "first_timestamp": str(ts.min()),
            "last_timestamp": str(ts.max()),
            "trading_dates": int(dates.nunique()),
        }
    return {"eligible": not issues, "rows": len(data), "issues": issues, "coverage": coverage}


def _default_yahoo_fetcher(ticker: str, interval: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker(ticker).history(
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        prepost=False,
    )


def fetch_multi_symbol_research_data(
    symbols: list[str],
    config: ResearchFetchConfig,
    output_dir: str | Path,
    fetcher: Callable[[str, str, str, str], pd.DataFrame] | None = None,
) -> dict:
    """Fetch each symbol independently, save successful datasets and a manifest.

    Symbols should be Yahoo tickers (e.g. RELIANCE.NS). Failures are recorded
    per symbol and do not prevent the remaining symbols from being processed.
    """
    if not symbols or any(not str(s).strip() for s in symbols):
        raise ValueError("At least one non-empty symbol is required.")
    if config.interval_minutes not in YAHOO_INTERVALS:
        raise ValueError(f"Unsupported Yahoo intraday interval: {config.interval_minutes} minutes.")
    if not config.start_date or not config.end_date:
        raise ValueError("start_date and end_date are required.")
    if pd.Timestamp(config.start_date) >= pd.Timestamp(config.end_date):
        raise ValueError("start_date must be earlier than end_date.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fetcher = fetcher or _default_yahoo_fetcher
    interval = YAHOO_INTERVALS[config.interval_minutes]
    results = {}

    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        try:
            raw = fetcher(symbol, interval, config.start_date, config.end_date)
            candles = normalize_yahoo_frame(raw, config.market_timezone)
            audit = audit_research_candles(candles)
            file_path = output / f"{safe_symbol}_{config.interval_minutes}m.csv"
            if not candles.empty:
                candles.to_csv(file_path, index=False)
            results[symbol] = {
                "status": "SAVED" if audit["eligible"] else "REJECTED",
                "file": str(file_path) if not candles.empty else None,
                "audit": audit,
            }
        except Exception as exc:
            results[symbol] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

    manifest = {
        "stage": 8,
        "purpose": "multi-symbol historical research data",
        "interval_minutes": config.interval_minutes,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "market_timezone": config.market_timezone,
        "symbols_requested": [str(s).strip().upper() for s in symbols],
        "results": results,
        "successful_symbols": [s for s, r in results.items() if r.get("status") == "SAVED"],
        "failed_or_rejected_symbols": [s for s, r in results.items() if r.get("status") != "SAVED"],
        "paper_only": True,
        "execution": "none",
        "data_warning": "Provider availability, intraday retention, corporate actions, missing bars and data quality must be reviewed before evaluation.",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    manifest["manifest_file"] = str(manifest_path)
    return manifest
