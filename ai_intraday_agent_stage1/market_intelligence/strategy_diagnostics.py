"""Historical strategy filter diagnostics. Research-only; never selects a configuration."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import pandas as pd

from market_intelligence.indicators import calculate_indicators
from market_intelligence.screener import calculate_screening_metrics
from market_intelligence.strategy import StrategyConfig

FILTERS = ["trend_strength", "ema_alignment", "rsi", "momentum", "relative_volume", "ema_extension", "candle_confirmation"]


def _rank_strength(value):
    return {"weak": 1, "moderate": 2, "strong": 3}.get(value, 0)


def _trend_strength(row):
    spread = abs(row.get("ema20_vs_ema50_percent", float("nan")))
    slope = abs(row.get("ema_20_slope_percent", float("nan")))
    if pd.isna(spread) or pd.isna(slope):
        return "unknown"
    if spread >= 0.15 and slope >= 0.03:
        return "strong"
    if spread >= 0.05 and slope >= 0.01:
        return "moderate"
    return "weak"


def _direction_for_row(row, allow_short=True):
    close, e20, e50, slope = row["close"], row["ema_20"], row["ema_50"], row["ema_20_slope_percent"]
    if close > e20 > e50 and slope > 0:
        return "LONG"
    if allow_short and close < e20 < e50 and slope < 0:
        return "SHORT"
    return None


def _filter_flags(row, direction, config):
    close, e20, e50 = float(row["close"]), float(row["ema_20"]), float(row["ema_50"])
    slope, rsi, atr = float(row["ema_20_slope_percent"]), float(row["rsi_14"]), float(row["atr_14"])
    relvol = row.get("relative_volume")
    strength = _trend_strength(row)
    flags = {
        "trend_strength": _rank_strength(strength) >= _rank_strength(config.minimum_trend_strength),
        "ema_alignment": (close > e20 > e50 and slope > 0) if direction == "LONG" else (close < e20 < e50 and slope < 0),
        "rsi": (config.long_rsi_min <= rsi <= config.long_rsi_max) if direction == "LONG" else (config.short_rsi_min <= rsi <= config.short_rsi_max),
        "momentum": (rsi >= 55) if direction == "LONG" else (rsi <= 45),
        "relative_volume": pd.notna(relvol) and float(relvol) >= config.minimum_relative_volume,
        "ema_extension": (close-e20 <= config.max_ema20_distance_atr*atr) if direction == "LONG" else (e20-close <= config.max_ema20_distance_atr*atr),
        "candle_confirmation": float(row["close"]) > float(row["open"]) if direction == "LONG" else float(row["close"]) < float(row["open"]),
    }
    return flags


def _scenario_masks():
    masks = {"all_filters_pass": tuple(FILTERS)}
    for name in FILTERS:
        masks[f"remove_{name}"] = tuple(f for f in FILTERS if f != name)
    for left, right in combinations(FILTERS, 2):
        masks[f"remove_{left}_and_{right}"] = tuple(f for f in FILTERS if f not in (left, right))
    masks["remove_hard_filters_rsi_and_relative_volume"] = tuple(f for f in FILTERS if f not in ("rsi", "relative_volume"))
    masks["remove_trend_gates_strength_and_alignment"] = tuple(f for f in FILTERS if f not in ("trend_strength", "ema_alignment"))
    masks["remove_all_filters"] = ()
    return masks


def diagnose_symbol(frame, symbol, config=None):
    """Count directional observations and filter pass/fail; scenarios are descriptive only."""
    config = config or StrategyConfig()
    if frame is None or frame.empty:
        raise ValueError(f"{symbol}: dataset is empty")
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    if data["timestamp"].isna().any():
        raise ValueError(f"{symbol}: invalid timestamps")
    data = data.sort_values("timestamp").reset_index(drop=True)
    indicators = calculate_indicators(data)
    screening = calculate_screening_metrics(indicators)
    enriched = screening.get("data", indicators).copy()
    enriched["trend_strength"] = enriched.apply(_trend_strength, axis=1)

    rows = []
    valid_indicator_rows = 0
    for _, row in enriched.iterrows():
        if not all(pd.notna(row.get(c)) for c in ("ema_20", "ema_50", "rsi_14", "atr_14", "ema_20_slope_percent")):
            continue
        valid_indicator_rows += 1
        direction = _direction_for_row(row, allow_short=config.allow_short)
        if direction is None:
            continue
        flags = _filter_flags(row, direction, config)
        flags.update({"direction": direction, "timestamp": row["timestamp"], "symbol": symbol})
        flags["score"] = sum(bool(flags[f]) for f in FILTERS)
        flags["baseline_strategy_pass"] = bool(flags["rsi"] and flags["relative_volume"] and flags["score"] >= config.minimum_score)
        rows.append(flags)

    observations = pd.DataFrame(rows)
    if observations.empty:
        summary = {name: {"tested": 0, "passed": 0, "failed": 0, "pass_rate_pct": None} for name in FILTERS}
        scenarios = {name: {"required_filters": list(req), "passing_observations": 0, "pass_rate_pct": None} for name, req in _scenario_masks().items()}
        scenarios["strategy_baseline_hard_filters_plus_minimum_score"] = {"required_filters": ["rsi", "relative_volume", f"score>={config.minimum_score}"], "passing_observations": 0, "pass_rate_pct": None}
        return {"symbol": symbol, "rows": len(data), "valid_indicator_rows": valid_indicator_rows, "directional_observations": 0, "filter_summary": summary, "scenarios": scenarios, "diagnostic_rows": []}

    total = len(observations)
    summary = {}
    for name in FILTERS:
        passed = int(observations[name].sum())
        summary[name] = {"tested": total, "passed": passed, "failed": total-passed, "pass_rate_pct": round(100*passed/total, 2)}
    scenarios = {}
    for name, required_filters in _scenario_masks().items():
        mask = observations[list(required_filters)].all(axis=1) if required_filters else pd.Series(True, index=observations.index)
        count = int(mask.sum())
        scenarios[name] = {"required_filters": list(required_filters), "passing_observations": count, "pass_rate_pct": round(100*count/total, 2)}
    baseline_count = int(observations["baseline_strategy_pass"].sum())
    scenarios["strategy_baseline_hard_filters_plus_minimum_score"] = {
        "required_filters": ["rsi", "relative_volume", f"score>={config.minimum_score}"],
        "passing_observations": baseline_count, "pass_rate_pct": round(100*baseline_count/total, 2),
    }
    bottlenecks = sorted(FILTERS, key=lambda f: (summary[f]["pass_rate_pct"], f))
    direction_counts = observations["direction"].value_counts().to_dict()
    return {
        "symbol": symbol, "rows": len(data), "valid_indicator_rows": valid_indicator_rows,
        "directional_observations": total, "direction_counts": direction_counts,
        "filter_summary": summary, "bottleneck_filters_lowest_pass_rate_first": bottlenecks,
        "scenarios": scenarios, "diagnostic_rows": observations.to_dict(orient="records"),
        "notes": ["Counts are diagnostics, not trades or profitability estimates.", "Filter-removal scenarios are descriptive; no configuration is selected.", "Indicator calculations use current and prior data only; no forward returns are evaluated."],
    }


def diagnose_directory(directory, config=None):
    """Diagnose all symbol CSVs in a directory and aggregate filter counts."""
    path = Path(directory)
    files = sorted(path.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV datasets found in {path}")
    results = {}
    for file in files:
        symbol = file.stem.rsplit("_", 1)[0]
        results[symbol] = diagnose_symbol(pd.read_csv(file), symbol, config)
    aggregate = {}
    for name in FILTERS:
        tested = sum(r["filter_summary"][name]["tested"] for r in results.values())
        passed = sum(r["filter_summary"][name]["passed"] for r in results.values())
        aggregate[name] = {"tested": tested, "passed": passed, "failed": tested-passed, "pass_rate_pct": round(100*passed/tested, 2) if tested else None}
    ordered = sorted(FILTERS, key=lambda f: (aggregate[f]["pass_rate_pct"] if aggregate[f]["pass_rate_pct"] is not None else 101, f))
    return {
        "report_type": "strategy_filter_diagnostic", "symbols": results,
        "aggregate": {"directional_observations": sum(r["directional_observations"] for r in results.values()), "filter_summary": aggregate, "bottleneck_filters_lowest_pass_rate_first": ordered},
        "paper_only": True, "execution": "none",
        "warning": "This report does not estimate returns, select a strategy, or recommend relaxing any rule.",
    }
