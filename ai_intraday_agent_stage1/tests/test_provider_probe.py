import sys
from types import SimpleNamespace

import pandas as pd

from market_intelligence.provider_probe import compare_frames, expected_session_grid, run_probe, summarize_provider_frame


def frame(times):
    return pd.DataFrame({
        "Datetime": times,
        "Open": [100.0] * len(times),
        "High": [101.0] * len(times),
        "Low": [99.0] * len(times),
        "Close": [100.5] * len(times),
        "Volume": [1000] * len(times),
    })


def test_expected_grid_is_75_five_minute_bars():
    grid = expected_session_grid(pd.Timestamp("2026-11-19").date())
    assert len(grid) == 75
    assert grid[0].strftime("%H:%M") == "09:15"
    assert grid[-1].strftime("%H:%M") == "15:25"


def test_missing_completed_bar_is_reported_without_repair():
    grid = expected_session_grid(pd.Timestamp("2026-11-19").date())
    times = grid.delete(39)  # 12:30 candle absent
    report = summarize_provider_frame(frame(times), as_of="2026-11-20 16:00:00+05:30")
    assert report["missing_completed_bars"] == 1
    assert report["missing_time_frequency"] == {"12:30": 1}


def test_provider_comparison_reports_timestamp_disagreement():
    grid = expected_session_grid(pd.Timestamp("2026-11-19").date())
    a, b = frame(grid.delete(39)), frame(grid)
    report = compare_frames(a, b, as_of="2026-11-20 16:00:00+05:30")
    assert report["timestamp_comparison"]["primary_only_timestamps"] == 0
    assert report["timestamp_comparison"]["alternate_only_timestamps"] == 1
    assert report["alternate"]["missing_completed_bars"] == 0


def test_probe_report_discloses_same_provider_scope(monkeypatch, tmp_path):
    grid = expected_session_grid(pd.Timestamp("2026-11-19").date())
    fake_yfinance = SimpleNamespace(
        __version__="test-version",
        Ticker=lambda symbol: SimpleNamespace(history=lambda **kwargs: frame(grid)),
        download=lambda *args, **kwargs: frame(grid),
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)

    output = tmp_path / "probe.json"
    report = run_probe(["TEST.NS"], "2026-11-19", "2026-11-20", output)

    assert report["yfinance_version"] == "test-version"
    assert report["independent_provider_comparison"] is False
    assert "not an independent provider" in report["comparison_scope"]
    assert "independent market-data provider" in report["interpretation_warning"]
    assert output.exists()
