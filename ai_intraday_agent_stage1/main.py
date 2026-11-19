import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
import pandas as pd

from data_quality import audit_candles, print_report
from market_data.upstox import UpstoxProvider
from market_data.csv_provider import CSVProvider
from market_data.demo import DemoProvider
from market_data.yahoo import YahooProvider

from data_validation import validate
from database.store import CandleStore
from market_intelligence.quality_gate import assess_candles, print_quality_report
from market_intelligence.evaluation import evaluate_market_data
from market_intelligence.strategy import generate_setup, StrategyConfig
from market_intelligence.risk import assess_candidate, RiskConfig
from market_intelligence.paper_simulator import simulate_candidate, SimulationConfig, PaperTradeStore
import json
from market_intelligence.backtester import backtest, BacktestConfig


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")

CONFIG = yaml.safe_load(
    (ROOT / "config.yaml").read_text(encoding="utf-8")
)

DB = ROOT / CONFIG["data"]["database"]


# =========================================================
# UTILITIES
# =========================================================

def symbol_config(symbol):
    for item in CONFIG["instruments"]["watchlist"]:
        if item["symbol"].upper() == symbol.upper():
            return item

    raise SystemExit(
        f"Unknown symbol {symbol}. "
        "Edit config.yaml/watchlist first."
    )


def pipeline(symbol, interval, df, source):
    clean, issues = validate(df)

    store = CandleStore(str(DB))

    saved = store.save(
        symbol,
        interval,
        clean,
        source,
        issues
    )

    print(
        f"Source: {source} | "
        f"rows received: {len(df)} | "
        f"accepted: {len(clean)} | "
        f"issues: {len(issues)} | "
        f"stored: {saved}"
    )

    if issues:
        print("\nQuality issues (first 10):")

        for issue in issues[:10]:
            print(
                f"  row {issue['row']}: "
                f"{issue['issue']}"
            )

    if not clean.empty:
        print(
            f"Range: {clean.timestamp.min()} "
            f"-> {clean.timestamp.max()}"
        )

    return clean


# =========================================================
# DEMO DATA
# =========================================================

def cmd_demo(args):
    df = DemoProvider().get_intraday(
        args.symbol,
        args.interval
    )

    pipeline(
        args.symbol,
        args.interval,
        df,
        "synthetic_demo"
    )


# =========================================================
# UPSTOX
# =========================================================

def cmd_fetch(args):
    inst = symbol_config(args.symbol)["instrument_key"]

    provider = UpstoxProvider()

    df = provider.get_intraday(
        inst,
        args.interval
    )

    pipeline(
        args.symbol,
        args.interval,
        df,
        "upstox_v3_intraday"
    )


def cmd_history(args):
    inst = symbol_config(args.symbol)["instrument_key"]

    provider = UpstoxProvider()

    df = provider.get_history(
        inst,
        "minutes",
        args.interval,
        args.from_date,
        args.to_date
    )

    pipeline(
        args.symbol,
        args.interval,
        df,
        "upstox_v3_history"
    )


# =========================================================
# YAHOO FINANCE
# =========================================================

def yahoo_ticker(symbol):
    """
    Convert Indian market symbols into Yahoo Finance tickers.
    """

    symbol = symbol.upper()

    ticker_mapping = {
        "NIFTY50": "^NSEI",
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN",
    }

    if symbol in ticker_mapping:
        return ticker_mapping[symbol]

    return f"{symbol}.NS"


def cmd_fetch_yahoo(args):
    ticker = yahoo_ticker(args.symbol)

    print(
        f"Fetching Yahoo Finance data for {args.symbol}"
    )

    print(f"Yahoo ticker: {ticker}")

    print(
        f"Interval: {args.interval} minute(s)"
    )

    provider = YahooProvider()

    df = provider.get_intraday(
        ticker,
        args.interval
    )

    pipeline(
        args.symbol,
        args.interval,
        df,
        "yahoo_finance"
    )


def cmd_yahoo_history(args):
    ticker = yahoo_ticker(args.symbol)

    print(
        f"Fetching historical data for {args.symbol}"
    )

    print(f"Yahoo ticker: {ticker}")

    provider = YahooProvider()

    df = provider.get_history(
        ticker,
        args.unit,
        args.interval,
        args.from_date,
        args.to_date
    )

    pipeline(
        args.symbol,
        args.interval,
        df,
        "yahoo_finance_history"
    )


# =========================================================
# CSV IMPORT
# =========================================================

def cmd_import(args):
    provider = CSVProvider(args.file)

    df = provider.get_intraday(
        args.symbol,
        args.interval
    )

    pipeline(
        args.symbol,
        args.interval,
        df,
        f"csv:{Path(args.file).name}"
    )


# =========================================================
# DATA QUALITY MONITOR
# =========================================================

def cmd_quality(args):
    store = CandleStore(str(DB))

    df = store.read(
        args.symbol,
        args.interval
    )

    if df.empty:
        print(
            f"No data found for {args.symbol} "
            f"at {args.interval}-minute interval."
        )

        return

    report = audit_candles(
        df,
        interval_minutes=args.interval,
        freshness_hours=args.freshness_hours
    )

    print_report(report)


# =========================================================
# NEW QUALITY GATE
# =========================================================

def cmd_quality_gate(args):
    store = CandleStore(str(DB))
    df = store.read(args.symbol, args.interval)
    if df.empty:
        print(f"No data found for {args.symbol} at {args.interval}-minute interval.")
        return
    report = assess_candles(
        df,
        interval_minutes=args.interval,
        config=None
    )
    print_quality_report(report)
    if args.require_eligible and not report["trade_eligible"]:
        raise SystemExit(2)


# =========================================================
# INTEGRATED EVALUATION
# =========================================================

def cmd_evaluate(args):
    store = CandleStore(str(DB))
    df = store.read(args.symbol, args.interval)
    if df.empty:
        print(f"No data found for {args.symbol} at {args.interval}-minute interval.")
        return
    result = evaluate_market_data(df, interval_minutes=args.interval, symbol=args.symbol)
    print("\n" + "=" * 58)
    print("       DATA-GATED MARKET EVALUATION")
    print("=" * 58)
    print(f"Symbol: {args.symbol}")
    print(f"Interval: {args.interval} minute(s)")
    print(f"Status: {result['status']}")
    print(f"Trade eligible for strategy: {'YES' if result['trade_eligible'] else 'NO'}")
    print(f"Quality score: {result['quality']['quality_score']}/100")
    if result['blocking_reasons']:
        print("\nBlocking reasons:")
        for reason in result['blocking_reasons']:
            print(f"- {reason}")
    if result.get('technical_analysis') and result['technical_analysis'].get('regime'):
        regime = result['technical_analysis']['regime']
        print("\nMarket regime:")
        for key in ('trend', 'trend_strength', 'momentum', 'volatility'):
            if key in regime:
                print(f"{key}: {regime[key]}")
    if result.get('screening') and result['screening'].get('metrics'):
        print("\nScreening metrics:")
        for key, value in result['screening']['metrics'].items():
            print(f"{key}: {value}")
    print("\\n" + result.get('warning', ''))
    print("=" * 58)
    if args.require_eligible and not result['trade_eligible']:
        raise SystemExit(2)


# =========================================================
# STRATEGY EVALUATION (PAPER CANDIDATE ONLY)
# =========================================================

def cmd_strategy(args):
    store = CandleStore(str(DB))
    df = store.read(args.symbol, args.interval)
    if df.empty:
        print(f"No data found for {args.symbol} at {args.interval}-minute interval.")
        return
    evaluation = evaluate_market_data(df, interval_minutes=args.interval, symbol=args.symbol)
    setup = generate_setup(evaluation, StrategyConfig(allow_short=not args.long_only))
    print("\n" + "=" * 58)
    print("        PAPER STRATEGY CANDIDATE")
    print("=" * 58)
    print(f"Symbol: {args.symbol}")
    print(f"Interval: {args.interval} minute(s)")
    print(f"Status: {setup['status']}")
    print(f"Signal: {setup['signal']}")
    print(f"Score: {setup['score']}/{setup['score_max']}")
    print(f"Confidence (rule score): {setup['confidence']}%")
    for reason in setup['reasons']:
        print(f"+ {reason}")
    if setup['candidate']:
        print("\nCandidate (NOT an order):")
        for key, value in setup['candidate'].items():
            print(f"{key}: {value}")
    if setup['diagnostics'].get('failed_checks'):
        print("\nFailed checks:")
        for reason in setup['diagnostics']['failed_checks']:
            print(f"- {reason}")
    if setup['status'] == 'BLOCKED':
        print("\nBlocking reasons:")
        for reason in setup['reasons']:
            print(f"- {reason}")
    print("\nPaper-only: YES | Execution: NONE")
    print("=" * 58)
    if args.require_candidate and setup['status'] != 'CANDIDATE':
        raise SystemExit(2)


# =========================================================
# RISK CHECK (PAPER POSITION SIZE ONLY)
# =========================================================

def cmd_risk(args):
    store = CandleStore(str(DB))
    df = store.read(args.symbol, args.interval)
    if df.empty:
        print(f"No data found for {args.symbol} at {args.interval}-minute interval.")
        return
    evaluation = evaluate_market_data(df, interval_minutes=args.interval, symbol=args.symbol)
    setup = generate_setup(evaluation, StrategyConfig(allow_short=not args.long_only))
    risk = assess_candidate(
        setup.get("candidate"),
        RiskConfig(budget=args.budget),
        open_exposure=args.open_exposure,
        daily_pnl=args.daily_pnl,
        trades_today=args.trades_today,
    )
    print("\n" + "=" * 58)
    print("       PAPER RISK & POSITION SIZING")
    print("=" * 58)
    print(f"Symbol: {args.symbol}")
    print(f"Strategy status: {setup['status']}")
    print(f"Strategy signal: {setup['signal']}")
    print(f"Risk status: {risk['status']}")
    for key in ("quantity", "capital_required", "capital_pct_of_budget",
                "total_risk", "risk_pct_of_budget", "risk_budget",
                "reward_risk_ratio", "daily_loss_limit"):
        if key in risk:
            print(f"{key}: {risk[key]}")
    for reason in risk.get("reasons", []):
        print(f"- {reason}")
    print("Paper-only: YES | Execution: NONE")
    print("=" * 58)
    if args.require_approved and not risk["approved"]:
        raise SystemExit(2)


# =========================================================
# PAPER SIMULATION (NO EXECUTION)
# =========================================================

def cmd_paper_sim(args):
    candidate_data = json.loads(Path(args.candidate_json).read_text(encoding="utf-8"))
    if candidate_data.get("approved") is not True:
        raise SystemExit("Refusing simulation: candidate JSON must contain approved=true from risk assessment.")
    if candidate_data.get("quantity", 0) <= 0:
        raise SystemExit("Refusing simulation: approved quantity must be positive.")
    forward = pd.read_csv(args.forward_csv)
    result = simulate_candidate(
        candidate_data,
        forward,
        SimulationConfig(
            slippage_bps_per_side=args.slippage_bps,
            fee_rate_per_side=args.fee_rate,
            flat_fee_per_side=args.flat_fee,
        ),
    )
    print("\n" + "=" * 58)
    print("            PAPER TRADE SIMULATION")
    print("=" * 58)
    for key in ("status", "symbol", "direction", "quantity", "entry_timestamp",
                "entry_reference_price", "entry_fill_price", "stop_loss", "target",
                "exit_timestamp", "exit_fill_price", "exit_reason", "gross_pnl",
                "estimated_fees", "net_pnl", "return_on_entry_capital_pct"):
        if key in result:
            print(f"{key}: {result[key]}")
    if result.get("status") == "SIMULATED" and not args.no_save:
        trade_id = PaperTradeStore(str(DB)).save(result)
        print(f"Saved paper trade record: #{trade_id}")
    if result.get("cost_model_note"):
        print(result["cost_model_note"])
    print("PAPER ONLY | NO BROKER ORDERS")
    print("=" * 58)


def cmd_paper_trades(args):
    trades = PaperTradeStore(str(DB)).read_all()
    if trades.empty:
        print("No paper simulation records yet.")
        return
    print(trades[["id", "symbol", "direction", "quantity", "entry_fill_price",
                  "exit_fill_price", "exit_reason", "net_pnl"]].to_string(index=False))
    print(f"Total simulated net P&L: {trades['net_pnl'].sum():.2f}")


# =========================================================
# DATABASE STATUS
# =========================================================

def cmd_status(args):
    store = CandleStore(str(DB))

    df = store.read()

    print(f"Database: {DB}")
    print(f"Stored candles: {len(df)}")

    if len(df):
        print(
            df.groupby(
                ["symbol", "interval_minutes"]
            ).size().to_string()
        )

        print(
            f"Latest timestamp: {df.timestamp.max()}"
        )


# =========================================================
# EXPORT
# =========================================================

def cmd_export(args):
    store = CandleStore(str(DB))

    df = store.read(
        args.symbol,
        args.interval
    )

    out = ROOT / CONFIG["data"]["processed_dir"]

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    name = (
        f"{args.symbol or 'all'}_"
        f"{args.interval or 'all'}_candles.csv"
    )

    path = out / name

    df.to_csv(
        path,
        index=False
    )

    print(
        f"Exported {len(df)} rows -> {path}"
    )


# =========================================================
# SYSTEM DIAGNOSTICS
# =========================================================

def cmd_doctor(args):
    print("Stage 1 Data Infrastructure Doctor")

    print(
        f"Python: {sys.version.split()[0]}"
    )

    print(f"Database path: {DB}")

    print(
        "Upstox token configured: "
        f"{'YES' if os.getenv('UPSTOX_ACCESS_TOKEN') else 'NO'}"
    )

    print("Yahoo Finance available: YES")
    print("Demo mode available: YES")
    print("Paper-only: YES (no order API code)")

    print(
        "Note: Yahoo Finance free data may be delayed, "
        "limited, or incomplete."
    )


# =========================================================
# COMMAND LINE PARSER
# =========================================================

def build_parser():

    p = argparse.ArgumentParser(
        description=(
            "AI Intraday Agent Stage 1 "
            "data tools (paper-only)"
        )
    )

    # IMPORTANT:
    # Create the subparser object before registering commands.
    sub = p.add_subparsers(
        dest="cmd",
        required=True
    )

    # -----------------------------------------------------
    # DOCTOR
    # -----------------------------------------------------

    d = sub.add_parser("doctor")

    d.set_defaults(
        func=cmd_doctor
    )

    # -----------------------------------------------------
    # DEMO
    # -----------------------------------------------------

    demo = sub.add_parser("demo")

    demo.add_argument(
        "--symbol",
        default="DEMO"
    )

    demo.add_argument(
        "--interval",
        type=int,
        default=1
    )

    demo.set_defaults(
        func=cmd_demo
    )

    # -----------------------------------------------------
    # UPSTOX INTRADAY
    # -----------------------------------------------------

    f = sub.add_parser("fetch-intraday")

    f.add_argument(
        "--symbol",
        required=True
    )

    f.add_argument(
        "--interval",
        type=int,
        choices=range(1, 301),
        default=1
    )

    f.set_defaults(
        func=cmd_fetch
    )

    # -----------------------------------------------------
    # UPSTOX HISTORY
    # -----------------------------------------------------

    h = sub.add_parser("fetch-history")

    h.add_argument(
        "--symbol",
        required=True
    )

    h.add_argument(
        "--interval",
        type=int,
        default=1
    )

    h.add_argument(
        "--from-date",
        required=True
    )

    h.add_argument(
        "--to-date",
        required=True
    )

    h.set_defaults(
        func=cmd_history
    )

    # -----------------------------------------------------
    # YAHOO INTRADAY
    # -----------------------------------------------------

    yh = sub.add_parser("fetch-yahoo")

    yh.add_argument(
        "--symbol",
        required=True
    )

    yh.add_argument(
        "--interval",
        type=int,
        choices=[1, 2, 5, 15, 30, 60],
        default=5
    )

    yh.set_defaults(
        func=cmd_fetch_yahoo
    )

    # -----------------------------------------------------
    # YAHOO HISTORICAL
    # -----------------------------------------------------

    yhist = sub.add_parser("fetch-yahoo-history")

    yhist.add_argument(
        "--symbol",
        required=True
    )

    yhist.add_argument(
        "--unit",
        choices=["daily", "weekly", "monthly"],
        default="daily"
    )

    yhist.add_argument(
        "--interval",
        type=int,
        default=1
    )

    yhist.add_argument(
        "--from-date",
        required=True
    )

    yhist.add_argument(
        "--to-date",
        required=True
    )

    yhist.set_defaults(
        func=cmd_yahoo_history
    )

    # -----------------------------------------------------
    # CSV IMPORT
    # -----------------------------------------------------

    i = sub.add_parser("import-csv")

    i.add_argument(
        "--symbol",
        required=True
    )

    i.add_argument(
        "--interval",
        type=int,
        default=1
    )

    i.add_argument(
        "--file",
        required=True
    )

    i.set_defaults(
        func=cmd_import
    )

    # -----------------------------------------------------
    # DATA QUALITY
    # -----------------------------------------------------

    q = sub.add_parser("quality")

    q.add_argument(
        "--symbol",
        required=True
    )

    q.add_argument(
        "--interval",
        type=int,
        required=True
    )

    q.add_argument(
        "--freshness-hours",
        type=float,
        default=24
    )

    q.set_defaults(
        func=cmd_quality
    )

    # -----------------------------------------------------
    # QUALITY GATE
    # -----------------------------------------------------

    qg = sub.add_parser("quality-gate")
    qg.add_argument("--symbol", required=True)
    qg.add_argument("--interval", type=int, required=True)
    qg.add_argument("--require-eligible", action="store_true",
                    help="Exit with code 2 when data is not trade eligible.")
    qg.set_defaults(func=cmd_quality_gate)

    # -----------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------

    ev = sub.add_parser("evaluate")
    ev.add_argument("--symbol", required=True)
    ev.add_argument("--interval", type=int, required=True)
    ev.add_argument("--require-eligible", action="store_true",
                    help="Exit with code 2 when quality gate blocks.")
    ev.set_defaults(func=cmd_evaluate)

    # -----------------------------------------------------
    # STRATEGY
    # -----------------------------------------------------

    st = sub.add_parser("strategy")
    st.add_argument("--symbol", required=True)
    st.add_argument("--interval", type=int, required=True)
    st.add_argument("--long-only", action="store_true",
                    help="Disable short setup candidates.")
    st.add_argument("--require-candidate", action="store_true",
                    help="Exit with code 2 when no candidate is produced.")
    st.set_defaults(func=cmd_strategy)

    # -----------------------------------------------------
    # RISK
    # -----------------------------------------------------

    rk = sub.add_parser("risk")
    rk.add_argument("--symbol", required=True)
    rk.add_argument("--interval", type=int, required=True)
    rk.add_argument("--budget", type=float, default=8900.0)
    rk.add_argument("--open-exposure", type=float, default=0.0,
                    help="Current simulated open-position capital; not yet persisted.")
    rk.add_argument("--daily-pnl", type=float, default=0.0,
                    help="Current simulated daily P&L; not yet persisted.")
    rk.add_argument("--trades-today", type=int, default=0,
                    help="Current simulated trade count; not yet persisted.")
    rk.add_argument("--long-only", action="store_true")
    rk.add_argument("--require-approved", action="store_true")
    rk.set_defaults(func=cmd_risk)

    # -----------------------------------------------------
    # PAPER SIMULATOR
    # -----------------------------------------------------

    ps = sub.add_parser("paper-sim")
    ps.add_argument("--candidate-json", required=True,
                    help="JSON containing approved risk assessment fields.")
    ps.add_argument("--forward-csv", required=True,
                    help="Forward candles with timestamp, open, high, low, close, volume.")
    ps.add_argument("--slippage-bps", type=float, default=5.0)
    ps.add_argument("--fee-rate", type=float, default=0.0003,
                    help="Estimated fee rate per side.")
    ps.add_argument("--flat-fee", type=float, default=0.0)
    ps.add_argument("--no-save", action="store_true")
    ps.set_defaults(func=cmd_paper_sim)

    pt = sub.add_parser("paper-trades")
    pt.set_defaults(func=cmd_paper_trades)

    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    s = sub.add_parser("status")

    s.set_defaults(
        func=cmd_status
    )

    # -----------------------------------------------------
    # EXPORT
    # -----------------------------------------------------

    e = sub.add_parser("export")

    e.add_argument(
        "--symbol"
    )

    e.add_argument(
        "--interval",
        type=int
    )

    e.set_defaults(
        func=cmd_export
    )

    return p


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    args = build_parser().parse_args()

    try:
        args.func(args)

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr
        )

        sys.exit(1)