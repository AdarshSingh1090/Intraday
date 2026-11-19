"""Chronological data splitting and parameter sensitivity utilities.

These helpers avoid random row splits for time-series experiments. They do not
automatically select a winning configuration.
"""
from __future__ import annotations
import pandas as pd


def chronological_split(df, timestamp_col="timestamp", train_fraction=0.60, validation_fraction=0.20):
    if df is None or df.empty or timestamp_col not in df.columns:
        raise ValueError("A non-empty dataset with timestamp column is required.")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Train/validation fractions must be positive and sum to less than one.")
    work = df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    if work[timestamp_col].isna().any():
        raise ValueError("Invalid timestamps cannot be split.")
    work = work.sort_values(timestamp_col).reset_index(drop=True)
    dates = sorted(work[timestamp_col].dt.date.unique())
    if len(dates) < 3:
        raise ValueError("At least three distinct trading dates are required.")
    train_end = max(1, int(len(dates) * train_fraction))
    val_end = max(train_end + 1, int(len(dates) * (train_fraction + validation_fraction)))
    val_end = min(val_end, len(dates) - 1)
    train_dates, val_dates, test_dates = dates[:train_end], dates[train_end:val_end], dates[val_end:]
    return {
        "train": work[work[timestamp_col].dt.date.isin(train_dates)].copy(),
        "validation": work[work[timestamp_col].dt.date.isin(val_dates)].copy(),
        "test": work[work[timestamp_col].dt.date.isin(test_dates)].copy(),
        # Warm-up context includes only dates through each evaluation period's end.
        "validation_with_warmup": work[work[timestamp_col].dt.date <= val_dates[-1]].copy(),
        "test_with_warmup": work[work[timestamp_col].dt.date <= test_dates[-1]].copy(),
        "date_ranges": {
            "train": [str(train_dates[0]), str(train_dates[-1])],
            "validation": [str(val_dates[0]), str(val_dates[-1])],
            "test": [str(test_dates[0]), str(test_dates[-1])],
        },
        "method": "chronological_by_trading_date",
    }


def walk_forward_splits(df, timestamp_col="timestamp", min_train_days=20,
                        validation_days=5, step_days=5, expanding=True):
    """Generate date-aligned walk-forward folds with no row-level randomization."""
    if df is None or df.empty or timestamp_col not in df.columns:
        raise ValueError("A non-empty dataset with timestamp column is required.")
    if min_train_days < 1 or validation_days < 1 or step_days < 1:
        raise ValueError("Fold sizes must be positive.")
    work = df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    if work[timestamp_col].isna().any():
        raise ValueError("Invalid timestamps cannot be split.")
    work = work.sort_values(timestamp_col).reset_index(drop=True)
    dates = sorted(work[timestamp_col].dt.date.unique())
    folds = []
    validation_start = min_train_days
    fold_number = 1
    while validation_start + validation_days <= len(dates):
        train_start = 0 if expanding else validation_start - min_train_days
        train_dates = dates[train_start:validation_start]
        val_dates = dates[validation_start:validation_start + validation_days]
        if len(train_dates) < min_train_days:
            break
        folds.append({
            "fold": fold_number,
            "train": work[work[timestamp_col].dt.date.isin(train_dates)].copy(),
            "validation": work[work[timestamp_col].dt.date.isin(val_dates)].copy(),
            "train_dates": [str(train_dates[0]), str(train_dates[-1])],
            "validation_dates": [str(val_dates[0]), str(val_dates[-1])],
            "validation_with_warmup": work[work[timestamp_col].dt.date <= val_dates[-1]].copy(),
            "expanding": expanding,
        })
        validation_start += step_days
        fold_number += 1
    return folds


def run_parameter_sensitivity(data, configurations, backtest_fn, metric_keys=None):
    """Run supplied configs and return comparable rows; does not rank/select a winner."""
    rows = []
    for index, config in enumerate(configurations, start=1):
        result = backtest_fn(data, config=config)
        metrics = result.get("metrics", {})
        row = {"experiment_id": index, "status": result.get("status"),
               "config": getattr(config, "__dict__", str(config))}
        keys = metric_keys or ("trades", "win_rate_pct", "profit_factor", "net_pnl",
                               "expectancy_per_trade", "max_drawdown_pct", "total_return_pct")
        for key in keys:
            row[key] = metrics.get(key)
        rows.append(row)
    return pd.DataFrame(rows)

def run_walk_forward_backtests(df, backtest_fn, config_factory, symbol="UNKNOWN",
                               timestamp_col="timestamp", min_train_days=20,
                               validation_days=5, step_days=5, expanding=True):
    """Run each fold on validation dates while retaining prior history for warm-up."""
    folds = walk_forward_splits(df, timestamp_col, min_train_days,
                                validation_days, step_days, expanding)
    results=[]
    for fold in folds:
        start_date, end_date = fold["validation_dates"]
        config=config_factory(start_date, end_date)
        result=backtest_fn(fold["validation_with_warmup"], symbol=symbol, config=config)
        results.append({
            "fold": fold["fold"], "train_dates": fold["train_dates"],
            "validation_dates": fold["validation_dates"],
            "status": result.get("status"), "metrics": result.get("metrics", {}),
            "trades": result.get("trades", []), "result": result,
        })
    return {"status":"COMPLETED" if results else "NO_FOLDS",
            "symbol":symbol,"folds":results,
            "method":"walk_forward_validation_only; each fold retains prior history for indicator warm-up"}


def run_multi_symbol_backtests(datasets, backtest_fn, config_factory):
    """Run symbols separately and aggregate raw trades descriptively.

    The aggregate is not a portfolio simulator: it does not coordinate capital,
    concurrent exposure, correlation, or cross-symbol risk.
    """
    symbol_results={}
    combined_trades=[]
    for symbol, data in (datasets or {}).items():
        result=backtest_fn(data, symbol=symbol, config=config_factory(symbol))
        symbol_results[symbol]=result
        combined_trades.extend(result.get("trades", []))
    return {"status":"COMPLETED" if symbol_results else "NO_DATA",
            "symbols":symbol_results,"combined_trades":combined_trades,
            "combined_trade_metrics":evaluate_performance(combined_trades,starting_equity=8900.0,bootstrap_iterations=0),
            "warning":"Combined trade metrics are descriptive only and are not portfolio-level results."}
