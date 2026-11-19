"""Capital and per-trade risk validation for paper candidates only."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RiskConfig:
    budget: float = 8900.0
    risk_per_trade_pct: float = 0.5
    max_capital_per_trade_pct: float = 35.0
    max_portfolio_exposure_pct: float = 70.0
    max_daily_loss_pct: float = 2.0
    max_trades_per_day: int = 3\n    minimum_reward_risk_ratio: float = 2.0


def _number(value):
    try:
        n = float(value)
        return math.isfinite(n)
    except (TypeError, ValueError):
        return False


def assess_candidate(candidate, config=None, open_exposure=0.0, daily_pnl=0.0, trades_today=0):
    """Validate and size a candidate. No order or fill is created."""
    config = config or RiskConfig()
    reasons = []
    if not candidate:
        return {"status": "REJECTED", "approved": False, "quantity": 0,
                "reasons": ["No candidate supplied."], "paper_only": True, "execution": "none"}

    required_numeric = ("entry_reference_price", "stop_loss", "target")
    missing = [k for k in required_numeric if k not in candidate or not _number(candidate[k])]
    if not isinstance(candidate.get("direction"), str) or not candidate.get("direction"):
        missing.append("direction")
    if missing:
        return {"status": "REJECTED", "approved": False, "quantity": 0,
                "reasons": ["Missing/invalid candidate fields: " + ", ".join(missing)],
                "paper_only": True, "execution": "none"}

    direction = candidate["direction"]
    entry = float(candidate["entry_reference_price"])
    stop = float(candidate["stop_loss"])
    target = float(candidate["target"])
    risk_per_share = abs(entry - stop)
    if entry <= 0 or risk_per_share <= 0:
        reasons.append("Entry and risk per share must be positive.")
    if direction == "LONG" and not (stop < entry < target):
        reasons.append("Long candidate requires stop < entry < target.")
    elif direction == "SHORT" and not (target < entry < stop):
        reasons.append("Short candidate requires target < entry < stop.")
    elif direction not in ("LONG", "SHORT"):
        reasons.append("Unsupported direction.")

    if config.budget <= 0:
        reasons.append("Budget must be positive.")
    if not (0 < config.risk_per_trade_pct <= 100):
        reasons.append("Risk-per-trade percentage must be between 0 and 100.")
    if not (0 < config.max_capital_per_trade_pct <= 100):
        reasons.append("Capital-per-trade percentage must be between 0 and 100.")
    if open_exposure < 0:
        reasons.append("Open exposure cannot be negative.")
    if trades_today >= config.max_trades_per_day:
        reasons.append("Maximum trades per day reached.")
    daily_loss_limit = config.budget * config.max_daily_loss_pct / 100
    if daily_pnl <= -daily_loss_limit:
        reasons.append("Daily loss limit reached; paper trading halted for the day.")

    if reasons:
        return {"status": "REJECTED", "approved": False, "quantity": 0,
                "reasons": reasons, "paper_only": True, "execution": "none"}

    risk_budget = config.budget * config.risk_per_trade_pct / 100
    capital_limit = config.budget * config.max_capital_per_trade_pct / 100
    exposure_limit = config.budget * config.max_portfolio_exposure_pct / 100
    remaining_exposure = max(0.0, exposure_limit - open_exposure)
    allowed_capital = min(capital_limit, remaining_exposure)
    quantity_by_risk = math.floor(risk_budget / risk_per_share)
    quantity_by_capital = math.floor(allowed_capital / entry)
    quantity = min(quantity_by_risk, quantity_by_capital)

    if quantity <= 0:
        reasons.append("Risk/capital limits permit zero shares.")
        return {"status": "REJECTED", "approved": False, "quantity": 0,
                "reasons": reasons, "risk_budget": round(risk_budget, 2),
                "allowed_capital": round(allowed_capital, 2),
                "paper_only": True, "execution": "none"}

    capital_required = quantity * entry
    actual_risk = quantity * risk_per_share
    reward_per_share = abs(target - entry)
    rr = reward_per_share / risk_per_share
    return {
        "status": "APPROVED_FOR_PAPER_SIMULATION",
        "approved": True,
        "symbol": candidate.get("symbol"),
        "direction": direction,
        "quantity": quantity,
        "entry_reference_price": entry,
        "stop_loss": stop,
        "target": target,
        "capital_required": round(capital_required, 2),
        "capital_pct_of_budget": round(capital_required / config.budget * 100, 2),
        "risk_per_share": round(risk_per_share, 4),
        "total_risk": round(actual_risk, 2),
        "risk_pct_of_budget": round(actual_risk / config.budget * 100, 3),
        "risk_budget": round(risk_budget, 2),
        "reward_risk_ratio": round(rr, 2),
        "remaining_exposure_after": round(open_exposure + capital_required, 2),
        "daily_loss_limit": round(daily_loss_limit, 2),
        "reasons": ["Position size is within configured risk and capital limits."],
        "paper_only": True,
        "execution": "none",
    }
