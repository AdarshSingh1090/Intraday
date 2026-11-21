"""Statistical and risk evaluation for paper backtests.

Metrics are descriptive. Short samples and intraday-only histories are not
sufficient to establish predictive edge.
"""
from __future__ import annotations

import math
import random
from statistics import mean, median, pstdev


def _finite(value):
    try:
        number = float(value)
        return math.isfinite(number)
    except (TypeError, ValueError):
        return False


def _safe_float(value, default=0.0):
    return float(value) if _finite(value) else default


def _percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_mean_ci(values, iterations=5000, confidence=0.95, seed=42):
    """Percentile bootstrap CI for mean trade P&L; deterministic for reproducibility."""
    if len(values) < 2 or iterations <= 0:
        return None
    rng = random.Random(seed)
    samples = []
    n = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(mean(sample))
    alpha = (1 - confidence) / 2
    return [round(_percentile(samples, alpha), 4), round(_percentile(samples, 1-alpha), 4)]


def evaluate_performance(trades, starting_equity=8900.0, bootstrap_iterations=5000):
    """Compute trade, equity, daily, and uncertainty metrics from closed trades."""
    if not _finite(starting_equity) or float(starting_equity) <= 0:
        raise ValueError("starting_equity must be a finite positive number")
    trades = sorted(list(trades or []), key=lambda t: str(t.get("entry_timestamp") or t.get("exit_timestamp") or ""))
    pnl = [_safe_float(t.get("net_pnl")) for t in trades]
    gross_pnl = [_safe_float(t.get("gross_pnl")) for t in trades]
    fees = [_safe_float(t.get("estimated_fees")) for t in trades]
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]
    breakeven = [x for x in pnl if x == 0]
    gross_profit = sum(wins)
    gross_loss_abs = abs(sum(losses))
    pf = gross_profit / gross_loss_abs if gross_loss_abs else (None if gross_profit == 0 else None)

    equity = float(starting_equity)
    peak = equity
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    equity_curve = [{"trade": 0, "equity": round(equity, 2), "timestamp": None}]
    for index, trade in enumerate(trades, start=1):
        equity += pnl[index-1]
        peak = max(peak, equity)
        dd = peak - equity
        dd_pct = dd / peak * 100 if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)
        max_drawdown_pct = max(max_drawdown_pct, dd_pct)
        equity_curve.append({"trade": index, "equity": round(equity, 2),
                            "timestamp": trade.get("exit_timestamp")})

    daily = {}
    for trade, value in zip(trades, pnl):
        stamp = str(trade.get("entry_timestamp") or trade.get("exit_timestamp") or "")
        day = stamp[:10] if stamp else "UNKNOWN"
        daily[day] = daily.get(day, 0.0) + value
    daily = dict(sorted(daily.items()))
    daily_values = list(daily.values())
    daily_returns = []
    running_equity = float(starting_equity)
    for day_pnl in daily_values:
        daily_returns.append(day_pnl / running_equity if running_equity else 0.0)
        running_equity += day_pnl

    sharpe = None
    if len(daily_returns) >= 2 and pstdev(daily_returns) > 0:
        sharpe = mean(daily_returns) / pstdev(daily_returns) * math.sqrt(252)
    downside = [min(0.0, r) for r in daily_returns]
    downside_dev = math.sqrt(sum(x*x for x in downside) / len(downside)) if downside else 0.0
    sortino = mean(daily_returns) / downside_dev * math.sqrt(252) if daily_returns and downside_dev > 0 else None

    r_values = []
    for trade in trades:
        r = trade.get("r_multiple_net")
        if _finite(r):
            r_values.append(float(r))
    days = len([d for d in daily if d != "UNKNOWN"])
    total_return_pct = (equity / starting_equity - 1) * 100
    avg_daily_return = mean(daily_returns) if daily_returns else 0.0
    calmar = (avg_daily_return * 252 / (max_drawdown_pct / 100)) if max_drawdown_pct > 0 else None

    return {
        "trades": len(trades),
        "wins": len(wins), "losses": len(losses), "breakeven": len(breakeven),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss_abs, 2),
        "profit_factor": round(pf, 4) if pf is not None else None,
        "gross_pnl": round(sum(gross_pnl), 2),
        "estimated_fees": round(sum(fees), 2),
        "net_pnl": round(sum(pnl), 2),
        "expectancy_per_trade": round(mean(pnl), 4) if pnl else 0.0,
        "median_trade_pnl": round(median(pnl), 4) if pnl else 0.0,
        "average_win": round(mean(wins), 4) if wins else 0.0,
        "average_loss": round(mean(losses), 4) if losses else 0.0,
        "win_loss_ratio": round(mean(wins) / abs(mean(losses)), 4) if wins and losses and mean(losses) != 0 else None,
        "best_trade": round(max(pnl), 2) if pnl else 0.0,
        "worst_trade": round(min(pnl), 2) if pnl else 0.0,
        "total_return_pct": round(total_return_pct, 4),
        "ending_equity": round(equity, 2),
        "starting_equity": round(float(starting_equity), 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "daily_sharpe_annualized": round(sharpe, 4) if sharpe is not None else None,
        "daily_sortino_annualized": round(sortino, 4) if sortino is not None else None,
        "calmar_ratio_approx": round(calmar, 4) if calmar is not None else None,
        "trading_days": days,
        "trades_per_day": round(len(trades) / max(1, days), 3),
        "average_r_multiple_net": round(mean(r_values), 4) if r_values else None,
        "median_r_multiple_net": round(median(r_values), 4) if r_values else None,
        "expectancy_bootstrap_95_ci": _bootstrap_mean_ci(pnl, bootstrap_iterations),
        "daily_net_pnl": {day: round(value, 2) for day, value in daily.items()},
        "equity_curve": equity_curve,
        "warnings": [
            "Sharpe/Sortino annualization assumes 252 trading days and is unstable on short samples.",
            "Bootstrap interval is trade-level and does not account for serial dependence or regime clustering.",
            "Metrics are historical simulation estimates, not forecasts."
        ],
    }

def segment_performance(trades, group_by=("symbol", "direction", "exit_reason"),
                       starting_equity=8900.0, bootstrap_iterations=1000):
    """Summarize performance by grounded trade attributes; omit missing groups."""
    output = {}
    for field in group_by:
        groups = {}
        for trade in trades or []:
            value = trade.get(field)
            if value is None:
                continue
            groups.setdefault(str(value), []).append(trade)
        output[field] = {
            label: evaluate_performance(group, starting_equity=starting_equity,
                                       bootstrap_iterations=bootstrap_iterations)
            for label, group in groups.items()
        }
    return output
