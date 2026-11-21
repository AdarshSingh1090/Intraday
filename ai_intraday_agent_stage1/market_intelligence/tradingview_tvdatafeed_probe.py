"""No-login TradingView 5-minute candle diagnostic via unofficial tvdatafeed.

Bounded to one symbol and 5,000 bars. Manual review only; never repairs CSVs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TARGET_TIMES = ("11:45", "11:50")
NEIGHBOR_TIMES = ("11:40", "11:55")
PRICE_TOLERANCE = 0.01


def fetch_tv_hist(symbol, exchange="NSE", interval="5", n_bars=5000):
    from tvDatafeed import TvDatafeed, Interval
    tv = TvDatafeed()  # no username/password; unauthenticated public mode
    try:
        return tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=Interval.in_5_minute,
            n_bars=n_bars,
            extended_session=False,
        )
    finally:
        ws = getattr(tv, "ws", None)
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


def _normalize_frame(frame):
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = frame.copy()
    if "timestamp" not in out.columns:
        out = out.reset_index()
    timestamp_col = next((c for c in out.columns if str(c).lower() in {"timestamp", "datetime", "date", "time"}), None)
    if timestamp_col is None and "index" in out.columns:
        timestamp_col = "index"  # reset_index() on an unnamed DatetimeIndex
    if timestamp_col is None:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out["timestamp"] = pd.to_datetime(out[timestamp_col], errors="coerce")
    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert("Asia/Kolkata")
    for field in ("open", "high", "low", "close", "volume"):
        if field not in out:
            out[field] = pd.NA
        out[field] = pd.to_numeric(out[field], errors="coerce")
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _bar_at(frame, timestamp):
    rows = frame.loc[frame["timestamp"] == timestamp]
    if rows.empty:
        return None
    row = rows.iloc[0]
    values = {field: float(row[field]) for field in ("open", "high", "low", "close", "volume") if pd.notna(row[field])}
    return values if len(values) == 5 else None


def _matches(source, yahoo):
    if source is None or yahoo is None:
        return False
    return (
        all(abs(float(source[field]) - float(yahoo[field])) <= PRICE_TOLERANCE
            for field in ("open", "high", "low", "close"))
        and int(float(source["volume"])) == int(float(yahoo["volume"]))
    )


def _read_yahoo_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = pd.read_csv(path)
    ts_col = next((c for c in frame.columns if str(c).lower() in {"timestamp", "datetime", "date", "time"}), None)
    if ts_col is None:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame[ts_col], errors="coerce")
    if frame["timestamp"].dt.tz is None:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert("Asia/Kolkata")
    return frame.dropna(subset=["timestamp"])


def probe(directory, symbol="RELIANCE.NS", output="reports/tradingview_tvdatafeed_probe.json",
          n_bars=5000, fetcher=None, target_date="2026-10-15"):
    fetcher = fetcher or fetch_tv_hist
    base_symbol = symbol.removesuffix(".NS")
    report = {
        "provider": "TradingView public chart feed via unofficial tvdatafeed",
        "authentication_required": False,
        "login_used": False,
        "n_bars_requested": n_bars,
        "symbol": symbol,
        "exchange": "NSE",
        "interval": "5m",
        "target_date": target_date,
        "target_times_ist": [f"{target_date} {t}:00+05:30" for t in TARGET_TIMES],
        "neighbor_times_ist": [f"{target_date} {t}:00+05:30" for t in NEIGHBOR_TIMES],
        "data_repaired": False,
        "csv_modified": False,
        "manual_review_candidate": False,
    }
    try:
        raw = fetcher(base_symbol, "NSE", "5", n_bars)
        frame = _normalize_frame(raw)
        report["status"] = "DATA_RETURNED" if not frame.empty else "EMPTY_OR_UNAVAILABLE"
        report["rows_returned"] = int(len(frame))
        report["earliest_ist"] = frame["timestamp"].min().isoformat() if not frame.empty else None
        report["latest_ist"] = frame["timestamp"].max().isoformat() if not frame.empty else None
        targets = [pd.Timestamp(f"{target_date} {t}:00", tz="Asia/Kolkata") for t in TARGET_TIMES]
        neighbors = [pd.Timestamp(f"{target_date} {t}:00", tz="Asia/Kolkata") for t in NEIGHBOR_TIMES]
        report["target_bars"] = {str(ts): _bar_at(frame, ts) for ts in targets}
        report["target_complete"] = all(bar is not None for bar in report["target_bars"].values())
        yahoo = _read_yahoo_csv(Path(directory) / f"{symbol}_5m.csv")
        checks = []
        for ts in neighbors:
            source_bar = _bar_at(frame, ts)
            yahoo_bar = _bar_at(yahoo, ts)
            checks.append({"timestamp_ist": ts.isoformat(), "tradingview_bar": source_bar,
                           "yahoo_bar": yahoo_bar, "match": _matches(source_bar, yahoo_bar)})
        matches = sum(check["match"] for check in checks)
        report["neighbor_validation"] = {
            "checks": checks, "matches": matches, "required_matches": 2,
            "status": "BOTH_NEIGHBORS_MATCH" if matches == 2 else "NEIGHBOR_VALIDATION_FAILED",
        }
        report["candidate_status"] = (
            "CANDIDATES_FOR_MANUAL_REVIEW" if report["target_complete"] and matches == 2
            else "WITHHELD_NEIGHBOR_MISMATCH" if report["target_complete"]
            else "NO_COMPLETE_TARGET_BARS"
        )
        report["manual_review_candidate"] = report["candidate_status"] == "CANDIDATES_FOR_MANUAL_REVIEW"
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["candidate_status"] = "INCONCLUSIVE_PROVIDER_ERROR"
        report["target_bars"] = {f"{target_date} {t}:00+05:30": None for t in TARGET_TIMES}
        report["neighbor_validation"] = {"status": "NOT_RUN", "checks": [], "matches": 0}
    report["interpretation"] = (
        "No-login TradingView diagnostic only. Target candles are manual-review candidates only if both "
        "target bars exist and both neighboring 5m bars match Yahoo within 0.01 price and exact volume. "
        "TradingView feed provenance and no-login availability are not guaranteed; no CSV is changed."
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="data/research")
    parser.add_argument("--symbol", default="RELIANCE.NS")
    parser.add_argument("--target-date", default="2026-10-15")
    parser.add_argument("--n-bars", type=int, default=5000)
    parser.add_argument("--output", default="reports/tradingview_tvdatafeed_probe.json")
    args = parser.parse_args()
    result = probe(args.directory, args.symbol, args.output, args.n_bars, target_date=args.target_date)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
