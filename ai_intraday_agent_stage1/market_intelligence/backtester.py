"""Historical walk-forward backtesting. Research only; no broker execution."""
from dataclasses import dataclass
import math
import pandas as pd

from market_intelligence.performance import evaluate_performance
from market_intelligence.evaluation import evaluate_market_data
from market_intelligence.strategy import generate_setup, StrategyConfig
from market_intelligence.risk import assess_candidate, RiskConfig
from market_intelligence.paper_simulator import simulate_candidate, SimulationConfig
from market_intelligence.quality_gate import QualityConfig


@dataclass(frozen=True)
class BacktestConfig:
    interval_minutes: int = 5
    warmup_bars: int = 60
    budget: float = 8900.0
    max_trades_per_day: int = 3
    daily_loss_limit_pct: float = 2.0
    risk_per_trade_pct: float = 0.5
    max_capital_per_trade_pct: float = 35.0
    max_portfolio_exposure_pct: float = 70.0
    slippage_bps_per_side: float = 5.0
    fee_rate_per_side: float = 0.0003
    flat_fee_per_side: float = 0.0
    long_only: bool = False


def _empty_metrics():
    return {
        "trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0,
        "gross_pnl": 0.0, "estimated_fees": 0.0, "net_pnl": 0.0,
        "profit_factor": 0.0, "average_win": 0.0, "average_loss": 0.0,
        "max_drawdown": 0.0, "trades_per_day": 0.0,
        "expectancy_per_trade": 0.0, "median_trade_pnl": 0.0,
        "max_drawdown_pct": 0.0, "total_return_pct": 0.0,
        "ending_equity": 0.0, "average_r_multiple_net": None,
        "expectancy_bootstrap_95_ci": None,
    }


def calculate_metrics(trades, starting_equity=8900.0):
    if not trades:
        return _empty_metrics()
    pnl = [float(t.get("net_pnl", 0.0)) for t in trades]
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    equity = peak = 0.0
    max_dd = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    days = {str(t.get("entry_timestamp", ""))[:10] for t in trades}
    advanced = evaluate_performance(trades, starting_equity=starting_equity)
    return {
        "trades": len(pnl), "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(pnl) * 100, 2),
        "gross_pnl": round(sum(float(t.get("gross_pnl", 0)) for t in trades), 2),
        "estimated_fees": round(sum(float(t.get("estimated_fees", 0)) for t in trades), 2),
        "net_pnl": round(sum(pnl), 2),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else (float("inf") if gross_profit else 0.0),
        "average_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "average_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "max_drawdown": round(max_dd, 2),
        "trades_per_day": round(len(pnl) / max(1, len(days)), 2),
        "expectancy_per_trade": advanced["expectancy_per_trade"],
        "median_trade_pnl": advanced["median_trade_pnl"],
        "max_drawdown_pct": advanced["max_drawdown_pct"],
        "total_return_pct": advanced["total_return_pct"],
        "ending_equity": advanced["ending_equity"],
        "average_r_multiple_net": advanced["average_r_multiple_net"],
        "daily_sharpe_annualized": advanced["daily_sharpe_annualized"],
        "daily_sortino_annualized": advanced["daily_sortino_annualized"],
        "calmar_ratio_approx": advanced["calmar_ratio_approx"],
        "expectancy_bootstrap_95_ci": advanced["expectancy_bootstrap_95_ci"],
        "daily_net_pnl": advanced["daily_net_pnl"],
        "equity_curve": advanced["equity_curve"],
        "warnings": advanced["warnings"],
    }


def backtest(df, symbol="UNKNOWN", config=None, strategy_config=None):
    """Walk forward: signal uses bars through t; fill/simulation uses t+1 onward."""
    config = config or BacktestConfig()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"status": "REJECTED", "reason": "No historical candles provided.", "trades": [], "metrics": _empty_metrics(), "paper_only": True}
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return {"status": "REJECTED", "reason": "Historical data missing required OHLCV columns.", "trades": [], "metrics": _empty_metrics(), "paper_only": True}
    bars = df.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    if bars[required].isna().any().any():
        return {"status": "REJECTED", "reason": "Historical candles contain invalid timestamps or missing/non-numeric OHLCV values.", "trades": [], "metrics": _empty_metrics(), "paper_only": True}
    if bars["timestamp"].duplicated().any():
        return {"status": "REJECTED", "reason": "Historical candles contain duplicate timestamps.", "trades": [], "metrics": _empty_metrics(), "paper_only": True}
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    o, h, l, cl = (bars[col] for col in ("open", "high", "low", "close"))
    invalid_ohlc = (h < bars[["open", "close", "low"]].max(axis=1)) | (l > bars[["open", "close"]].min(axis=1))
    if invalid_ohlc.any():
        return {"status": "REJECTED", "reason": "Historical candles contain inconsistent OHLC relationships.", "trades": [], "metrics": _empty_metrics(), "paper_only": True}
    if len(bars) <= config.warmup_bars + 1:
        return {"status": "REJECTED", "reason": "Insufficient candles for warm-up and forward simulation.", "trades": [], "metrics": _empty_metrics(), "paper_only": True}
    if (bars[["open", "high", "low", "close"]] <= 0).any().any() or (bars["volume"] < 0).any():
        return {"status": "REJECTED", "reason": "Historical candles contain invalid prices or negative volume.", "trades": [], "metrics": _empty_metrics(), "paper_only": True}

    trades, skipped = [], []
    daily_pnl, daily_count = {}, {}
    i = config.warmup_bars
    sim_config = SimulationConfig(config.slippage_bps_per_side, config.fee_rate_per_side, config.flat_fee_per_side)
    strat_config = strategy_config or StrategyConfig(allow_short=not config.long_only)
    while i < len(bars) - 1:
        signal_bar = bars.iloc[i]
        day = signal_bar["timestamp"].date().isoformat()
        pnl_today = daily_pnl.get(day, 0.0)
        count_today = daily_count.get(day, 0)
        # Prevent crossing session boundaries in forward simulations.
        forward = bars.iloc[i+1:].copy()
        forward = forward[forward["timestamp"].dt.date == signal_bar["timestamp"].date()]
        if forward.empty:
            i += 1
            continue
        # Historical freshness is anchored to the signal candle, not wall-clock now.
        window = bars.iloc[:i+1].copy()
        quality_cfg = QualityConfig(freshness_minutes=10**9, recent_window=min(20, len(window)), minimum_recent_quality=0.0)
        evaluation = evaluate_market_data(window, interval_minutes=config.interval_minutes, now=signal_bar["timestamp"], quality_config=quality_cfg, symbol=symbol)
        setup = generate_setup(evaluation, strat_config)
        if setup.get("status") != "CANDIDATE":
            i += 1
            continue
        risk_cfg = RiskConfig(
            budget=config.budget,
            risk_per_trade_pct=config.risk_per_trade_pct,
            max_capital_per_trade_pct=config.max_capital_per_trade_pct,
            max_portfolio_exposure_pct=config.max_portfolio_exposure_pct,
            max_daily_loss_pct=config.daily_loss_limit_pct,
            max_trades_per_day=config.max_trades_per_day,
        )
        risk = assess_candidate(setup["candidate"], risk_cfg, open_exposure=0, daily_pnl=pnl_today, trades_today=count_today)
        if not risk.get("approved"):
            skipped.append({"timestamp": str(signal_bar["timestamp"]), "reason": risk.get("reasons", [])})
            i += 1
            continue
        approved = dict(setup["candidate"])
        approved.update({k: risk[k] for k in ("quantity", "risk_budget") if k in risk})
        approved["max_capital"] = config.budget * config.max_capital_per_trade_pct / 100
        approved["approved"] = True
        sim = simulate_candidate(approved, forward, sim_config)
        if sim.get("status") != "SIMULATED":
            skipped.append({"timestamp": str(signal_bar["timestamp"]), "reason": sim.get("reason", "simulation rejected")})
            i += 1
            continue
        trades.append(sim)
        daily_pnl[day] = pnl_today + float(sim["net_pnl"])
        daily_count[day] = count_today + 1
        # Resume after the exit candle; never allow overlapping positions.
        exit_ts = pd.to_datetime(sim["exit_timestamp"], errors="coerce")
        exit_idx = bars.index[bars["timestamp"] == exit_ts]
        i = max(i + 1, int(exit_idx[0]) + 1) if len(exit_idx) else i + 1

    return {
        "status": "COMPLETED",
        "symbol": symbol,
        "config": config.__dict__,
        "trades": trades,
        "skipped": skipped,
        "metrics": calculate_metrics(trades, starting_equity=config.budget),
        "paper_only": True,
        "execution": "none",
        "disclaimer": "Historical simulation only. Results are not predictive; estimated costs and candle-based fills may differ from real execution.",
    }
