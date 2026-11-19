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
    result = evaluate_market_data(df, interval_minutes=args.interval)
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