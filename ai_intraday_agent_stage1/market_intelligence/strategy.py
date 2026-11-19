"""Rule-based intraday setup generator. Paper-only; does not execute trades."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StrategyConfig:
    minimum_score: int = 5
    minimum_relative_volume: float = 1.2
    atr_stop_multiplier: float = 1.2
    reward_risk_ratio: float = 2.0
    long_rsi_min: float = 52.0
    long_rsi_max: float = 68.0
    short_rsi_min: float = 32.0
    short_rsi_max: float = 48.0
    max_ema20_distance_atr: float = 1.0
    minimum_trend_strength: str = "moderate"
    allow_short: bool = True


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _no_trade(status, reasons, diagnostics=None):
    return {
        "status": status,
        "signal": "NO_TRADE",
        "candidate": None,
        "score": 0,
        "score_max": 6,
        "confidence": 0.0,
        "reasons": reasons,
        "diagnostics": diagnostics or {},
        "paper_only": True,
        "execution": "none",
    }


def generate_setup(evaluation, config=None):
    """Generate a scored candidate only when evaluation passed the data gate."""
    config = config or StrategyConfig()
    if not evaluation or not evaluation.get("trade_eligible"):
        reasons = (evaluation or {}).get("blocking_reasons") or ["Market data is not trade eligible."]
        return _no_trade("BLOCKED", list(reasons))

    technical = evaluation.get("technical_analysis") or {}
    regime = technical.get("regime") or {}
    indicator_df = technical.get("indicators")
    screening = (evaluation.get("screening") or {}).get("metrics") or {}

    if indicator_df is None or getattr(indicator_df, "empty", True):
        return _no_trade("INSUFFICIENT_DATA", ["Technical indicators are unavailable."])

    latest = indicator_df.iloc[-1]
    needed = ["close", "ema_20", "ema_50", "rsi_14", "atr_14", "atr_percent", "ema_20_slope_percent"]
    missing = [key for key in needed if key not in latest or not _finite(latest[key])]
    if missing:
        return _no_trade("INSUFFICIENT_DATA", ["Missing/non-finite indicators: " + ", ".join(missing)])

    close = float(latest["close"])
    ema20 = float(latest["ema_20"])
    ema50 = float(latest["ema_50"])
    rsi = float(latest["rsi_14"])
    atr = float(latest["atr_14"])
    slope = float(latest["ema_20_slope_percent"])
    relative_volume = screening.get("relative_volume")
    if not _finite(relative_volume):
        return _no_trade("INSUFFICIENT_DATA", ["Relative volume is unavailable."])

    if atr <= 0 or close <= 0:
        return _no_trade("INVALID_INDICATORS", ["ATR and close must be positive."])

    trend = regime.get("trend", "unknown")
    strength = regime.get("trend_strength", "unknown")
    momentum = regime.get("momentum", "unknown")
    volatility = regime.get("volatility", "unknown")

    strength_rank = {"weak": 1, "moderate": 2, "strong": 3}
    required_rank = strength_rank.get(config.minimum_trend_strength, 2)
    current_rank = strength_rank.get(strength, 0)

    direction = None
    if trend == "bullish" and current_rank >= required_rank:
        direction = "LONG"
    elif config.allow_short and trend == "bearish" and current_rank >= required_rank:
        direction = "SHORT"
    if direction is None:
        return _no_trade("NO_SETUP", ["Trend is not aligned with an allowed strategy direction."],
                        {"trend": trend, "strength": strength, "momentum": momentum})

    checks = []
    reasons = []
    if direction == "LONG":
        checks.append((close > ema20 > ema50 and slope > 0, "Price/EMA alignment bullish"))
        checks.append((config.long_rsi_min <= rsi <= config.long_rsi_max, f"RSI within long range {config.long_rsi_min}-{config.long_rsi_max}"))
        checks.append((momentum in ("bullish", "strong_bullish"), "Momentum confirms bullish direction"))
        checks.append((float(relative_volume) >= config.minimum_relative_volume, f"Relative volume >= {config.minimum_relative_volume}"))
        checks.append((close - ema20 <= config.max_ema20_distance_atr * atr, "Price not excessively extended from EMA20"))
        checks.append((float(latest["close"]) > float(latest["open"]), "Latest candle is bullish"))
    else:
        checks.append((close < ema20 < ema50 and slope < 0, "Price/EMA alignment bearish"))
        checks.append((config.short_rsi_min <= rsi <= config.short_rsi_max, f"RSI within short range {config.short_rsi_min}-{config.short_rsi_max}"))
        checks.append((momentum in ("bearish", "strong_bearish"), "Momentum confirms bearish direction"))
        checks.append((float(relative_volume) >= config.minimum_relative_volume, f"Relative volume >= {config.minimum_relative_volume}"))
        checks.append((ema20 - close <= config.max_ema20_distance_atr * atr, "Price not excessively extended from EMA20"))
        checks.append((float(latest["close"]) < float(latest["open"]), "Latest candle is bearish"))

    score = sum(bool(passed) for passed, _ in checks)
    reasons = [description for passed, description in checks if passed]
    failed = [description for passed, description in checks if not passed]
    confidence = round(score / len(checks) * 100, 1)

    if score < config.minimum_score:
        return _no_trade("NO_SETUP", ["Setup score below threshold.", *failed],
                        {"direction": direction, "score": score, "confidence": confidence,\n                         "passed_checks": reasons, "failed_checks": failed})

    entry_reference = close
    stop_distance = atr * config.atr_stop_multiplier
    if direction == "LONG":
        stop = entry_reference - stop_distance
        target = entry_reference + stop_distance * config.reward_risk_ratio
    else:
        stop = entry_reference + stop_distance
        target = entry_reference - stop_distance * config.reward_risk_ratio

    risk_per_share = abs(entry_reference - stop)
    reward_per_share = abs(target - entry_reference)
    actual_rr = reward_per_share / risk_per_share if risk_per_share else 0.0
    if actual_rr < config.reward_risk_ratio:
        return _no_trade("RISK_REWARD_REJECTED", ["Calculated reward/risk is below configured minimum."])

    candidate = {
        "direction": direction,
        "symbol": evaluation.get("symbol"),
        "signal_timestamp": str(latest["timestamp"]),
        "entry_reference_price": round(entry_reference, 4),
        "stop_loss": round(stop, 4),
        "target": round(target, 4),
        "risk_per_share": round(risk_per_share, 4),
        "reward_per_share": round(reward_per_share, 4),
        "reward_risk_ratio": round(actual_rr, 2),
        "atr": round(atr, 4),
        "atr_stop_multiplier": config.atr_stop_multiplier,
        "relative_volume": round(float(relative_volume), 3),
        "rsi": round(rsi, 2),
        "trend": trend,
        "trend_strength": strength,
        "volatility": volatility,
        "score": score,
        "score_max": len(checks),
        "confidence": confidence,
        "entry_note": "Close is a reference only; no simulated fill is assumed.",
    }
    return {
        "status": "CANDIDATE",
        "signal": direction,
        "candidate": candidate,
        "score": score,
        "score_max": len(checks),
        "confidence": confidence,
        "reasons": reasons,
        "diagnostics": {"failed_checks": failed},
        "paper_only": True,
        "execution": "none",
    }
