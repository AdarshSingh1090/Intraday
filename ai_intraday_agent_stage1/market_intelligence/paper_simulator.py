"""Deterministic paper-trade simulation. No broker or order execution."""
from dataclasses import dataclass
import math
import sqlite3
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    slippage_bps_per_side: float = 5.0
    fee_rate_per_side: float = 0.0003
    flat_fee_per_side: float = 0.0
    market_timezone: str = "Asia/Kolkata"


def _valid_num(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def simulate_candidate(candidate, forward_candles, config=None):
    """Simulate a candidate from the first forward candle's open.

    If stop and target both occur in one candle, stop is assumed first (conservative).
    If a stop is gapped through, exit at the open (worse than stop). Targets fill at
    target price. If neither exit is hit, close at final candle close.
    """
    config = config or SimulationConfig()
    required = ("entry_reference_price", "stop_loss", "target", "quantity")
    if not candidate or any(k not in candidate or not _valid_num(candidate[k]) for k in required) or candidate.get("direction") not in ("LONG", "SHORT"):
        return {"status": "REJECTED", "reason": "Candidate is missing required numeric fields.", "paper_only": True}
    if not isinstance(forward_candles, pd.DataFrame) or forward_candles.empty:
        return {"status": "REJECTED", "reason": "No forward candles supplied.", "paper_only": True}
    bars = forward_candles.copy()
    required_cols = {"timestamp", "open", "high", "low", "close"}
    if not required_cols.issubset(bars.columns):
        return {"status": "REJECTED", "reason": "Forward candles missing required OHLC columns.", "paper_only": True}
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
    try:
        if bars["timestamp"].dt.tz is None:
            bars["timestamp"] = bars["timestamp"].dt.tz_localize(config.market_timezone)
        else:
            bars["timestamp"] = bars["timestamp"].dt.tz_convert(config.market_timezone)
    except (TypeError, ValueError):
        return {"status": "REJECTED", "reason": "Forward timestamps could not be normalized to market timezone.", "paper_only": True}
    for col in ("open", "high", "low", "close"):
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    if bars[["timestamp", "open", "high", "low", "close"]].isna().any().any():
        return {"status": "REJECTED", "reason": "Forward candles contain invalid timestamps or numeric OHLC values.", "paper_only": True}
    if bars["timestamp"].duplicated().any():
        return {"status": "REJECTED", "reason": "Forward candles contain duplicate timestamps.", "paper_only": True}
    if (bars[["open", "high", "low", "close"]] <= 0).any().any():
        return {"status": "REJECTED", "reason": "Forward candles contain nonpositive prices.", "paper_only": True}
    invalid_ohlc = ((bars["high"] < bars[["open", "close", "low"]].max(axis=1)) |
                    (bars["low"] > bars[["open", "close"]].min(axis=1))
                    )
    if invalid_ohlc.any():
        return {"status": "REJECTED", "reason": "Forward candles contain inconsistent OHLC relationships.", "paper_only": True}
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    signal_ts = candidate.get("signal_timestamp")
    if signal_ts:
        try:
            signal_time = pd.to_datetime(signal_ts)
            if signal_time.tzinfo is None:
                signal_time = signal_time.tz_localize(config.market_timezone)
            else:
                signal_time = signal_time.tz_convert(config.market_timezone)
            first_forward_time = pd.to_datetime(bars.iloc[0]["timestamp"])
            if first_forward_time <= signal_time:
                return {"status": "REJECTED", "reason": "Forward candles must begin strictly after the signal timestamp.", "paper_only": True}
        except (TypeError, ValueError):
            return {"status": "REJECTED", "reason": "Signal/forward candle timestamps could not be compared.", "paper_only": True}
    direction = candidate["direction"]
    raw_qty = float(candidate["quantity"])
    if raw_qty <= 0 or not raw_qty.is_integer():
        return {"status": "REJECTED", "reason": "Quantity must be a positive whole number.", "paper_only": True}
    qty = int(raw_qty)
    ref = float(candidate["entry_reference_price"])
    stop = float(candidate["stop_loss"])
    target = float(candidate["target"])
    if qty <= 0 or ref <= 0 or direction not in ("LONG", "SHORT"):
        return {"status": "REJECTED", "reason": "Invalid direction, quantity, or reference price.", "paper_only": True}
    if (direction == "LONG" and not stop < ref < target) or (direction == "SHORT" and not target < ref < stop):
        return {"status": "REJECTED", "reason": "Stop/target geometry invalid for direction.", "paper_only": True}
    if not all(_valid_num(v) for v in (config.slippage_bps_per_side, config.fee_rate_per_side, config.flat_fee_per_side)) or min(config.slippage_bps_per_side, config.fee_rate_per_side, config.flat_fee_per_side) < 0:
        return {"status": "REJECTED", "reason": "Simulation cost configuration must be nonnegative.", "paper_only": True}

    slip = config.slippage_bps_per_side / 10000.0
    first = bars.iloc[0]
    entry_base = float(first["open"])
    entry_fill = entry_base * (1 + slip if direction == "LONG" else 1 - slip)
    # Avoid entering after a signal has already been invalidated or reached its target.
    if (direction == "LONG" and (entry_base <= stop or entry_base >= target)) or (direction == "SHORT" and (entry_base >= stop or entry_base <= target)):
        return {"status": "REJECTED", "reason": "Forward opening price crossed stop or target before simulated entry; setup is no longer valid.", "paper_only": True}
    risk_budget = candidate.get("risk_budget")
    stop_fill = stop * (1 - slip if direction == "LONG" else 1 + slip)
    estimated_stop_loss = abs(entry_fill - stop_fill) * qty
    estimated_stop_fees = (entry_fill * qty + stop_fill * qty) * config.fee_rate_per_side + 2 * config.flat_fee_per_side
    estimated_stop_risk = estimated_stop_loss + estimated_stop_fees
    if _valid_num(risk_budget) and estimated_stop_risk > float(risk_budget):
        return {"status": "REJECTED", "reason": "Opening gap, slippage and estimated stop costs make risk exceed approved risk budget.", "paper_only": True}
    max_capital = candidate.get("max_capital")
    if _valid_num(max_capital) and entry_fill * qty > float(max_capital):
        return {"status": "REJECTED", "reason": "Opening gap/slippage makes actual entry capital exceed approved capital limit.", "paper_only": True}
    # Opening-gap geometry is checked on the raw open; slippage and fees are
    # separately included in the risk-budget check and P&L cost model.
    actual_risk_per_share = abs(entry_base - stop)
    actual_reward_per_share = abs(target - entry_base)
    actual_reward_risk = actual_reward_per_share / actual_risk_per_share if actual_risk_per_share > 0 else 0.0
    minimum_reward_risk = float(candidate.get("minimum_reward_risk_ratio", 2.0))
    if actual_reward_risk < minimum_reward_risk:
        return {"status": "REJECTED", "reason": "Opening gap/slippage reduces actual reward-risk below minimum.", "paper_only": True}
    exit_price = None
    exit_reason = None
    exit_ts = None
    for _, bar in bars.iterrows():
        o, h, l, cl = map(float, (bar.open, bar.high, bar.low, bar.close))
        ts = str(bar.timestamp)
        if direction == "LONG":
            if o <= stop:
                exit_price, exit_reason = o * (1 - slip), "STOP_GAP"
            elif l <= stop and h >= target:
                exit_price, exit_reason = stop * (1 - slip), "STOP (BOTH HIT; CONSERVATIVE)"
            elif l <= stop:
                exit_price, exit_reason = stop * (1 - slip), "STOP"
            elif h >= target:
                exit_price, exit_reason = target * (1 - slip), "TARGET"
        else:
            if o >= stop:
                exit_price, exit_reason = o * (1 + slip), "STOP_GAP"
            elif h >= stop and l <= target:
                exit_price, exit_reason = stop * (1 + slip), "STOP (BOTH HIT; CONSERVATIVE)"
            elif h >= stop:
                exit_price, exit_reason = stop * (1 + slip), "STOP"
            elif l <= target:
                exit_price, exit_reason = target * (1 + slip), "TARGET"
        if exit_price is not None:
            exit_ts = ts
            break
    if exit_price is None:
        last = bars.iloc[-1]
        base_exit = float(last.close)
        exit_price = base_exit * (1 - slip if direction == "LONG" else 1 + slip)
        exit_reason = "END_OF_DATA"
        exit_ts = str(last.timestamp)

    gross = (exit_price - entry_fill) * qty if direction == "LONG" else (entry_fill - exit_price) * qty
    entry_notional = entry_fill * qty
    exit_notional = exit_price * qty
    fees = entry_notional * config.fee_rate_per_side + exit_notional * config.fee_rate_per_side + 2 * config.flat_fee_per_side
    net = gross - fees
    entry_dt = pd.to_datetime(first.timestamp)
    if entry_dt.tzinfo is not None:
        entry_dt = entry_dt.tz_convert("Asia/Kolkata")
    entry_clock = entry_dt.time()
    if entry_clock < pd.Timestamp("10:00").time():
        time_bucket = "OPEN_0915_1000"
    elif entry_clock < pd.Timestamp("12:00").time():
        time_bucket = "MORNING_1000_1200"
    elif entry_clock < pd.Timestamp("14:00").time():
        time_bucket = "MIDDAY_1200_1400"
    else:
        time_bucket = "CLOSE_1400_1530"
    return {
        "status": "SIMULATED",
        "symbol": candidate.get("symbol"),
        "direction": direction,
        "trend": candidate.get("trend"),
        "trend_strength": candidate.get("trend_strength"),
        "volatility": candidate.get("volatility"),
        "score": candidate.get("score"),
        "confidence": candidate.get("confidence"),
        "rsi": candidate.get("rsi"),
        "relative_volume": candidate.get("relative_volume"),
        "entry_time_bucket": time_bucket,
        "entry_month": entry_dt.strftime("%Y-%m"),
        "entry_weekday": entry_dt.strftime("%A"),
        "quantity": qty,
        "signal_timestamp": candidate.get("signal_timestamp"),
        "entry_timestamp": str(first.timestamp),
        "entry_reference_price": ref,
        "entry_fill_price": round(entry_fill, 4),
        "stop_loss": stop,
        "target": target,
        "exit_timestamp": exit_ts,
        "exit_fill_price": round(exit_price, 4),
        "exit_reason": exit_reason,
        "gross_pnl": round(gross, 2),
        "estimated_fees": round(fees, 2),
        "net_pnl": round(net, 2),
        "return_on_entry_capital_pct": round(net / entry_notional * 100, 4) if entry_notional else 0.0,
        "estimated_stop_risk": round(estimated_stop_risk, 2),
        "r_multiple_net": round(net / estimated_stop_risk, 4) if estimated_stop_risk > 0 else None,
        "entry_notional": round(entry_notional, 2),
        "exit_notional": round(exit_notional, 2),
        "paper_only": True,
        "execution": "none",
        "cost_model_note": "Estimated fees and slippage only; not actual exchange/broker charges.",
    }


class PaperTradeStore:
    """Append-only persistence for simulation results."""
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT, direction TEXT, quantity INTEGER,
                entry_timestamp TEXT, entry_fill_price REAL,
                exit_timestamp TEXT, exit_fill_price REAL,
                exit_reason TEXT, gross_pnl REAL, estimated_fees REAL,
                net_pnl REAL, result_json TEXT NOT NULL
            )""")

    def save(self, result):
        if result.get("status") != "SIMULATED":
            raise ValueError("Only completed simulation results can be persisted.")
        import json
        with sqlite3.connect(self.path) as con:
            cur = con.execute("""INSERT INTO paper_trades
                (symbol,direction,quantity,entry_timestamp,entry_fill_price,
                 exit_timestamp,exit_fill_price,exit_reason,gross_pnl,estimated_fees,net_pnl,result_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (result.get("symbol"),result.get("direction"),result.get("quantity"),
                 result.get("entry_timestamp"),result.get("entry_fill_price"),
                 result.get("exit_timestamp"),result.get("exit_fill_price"),result.get("exit_reason"),
                 result.get("gross_pnl"),result.get("estimated_fees"),result.get("net_pnl"),
                 json.dumps(result,sort_keys=True)))
            return cur.lastrowid

    def read_all(self):
        with sqlite3.connect(self.path) as con:
            return pd.read_sql_query("SELECT * FROM paper_trades ORDER BY id", con)
