"""Conservative NSE regular-session calendar for whole-session data-gap audits.

Calendar coverage is intentionally versioned. Unknown years block eligibility
instead of assuming weekday trading; special sessions are explicitly excluded
from regular-session strategy research.
"""
from __future__ import annotations

from datetime import date, timedelta
import pandas as pd

TIMEZONE = "Asia/Kolkata"
CALENDAR_SOURCE = "NSE official 2026 equity holiday calendar and special-session circulars"
SOURCE_URLS = [
    "https://nseindia.com/resources/exchange-communication-holidays/",
    "https://nsearchives.nseindia.com/content/circulars/CMTR71775.pdf",
    "https://nsearchives.nseindia.com/content/circulars/CMTR72260.pdf",
    "https://nsearchives.nseindia.com/content/circulars/CMTR72349.pdf",
    "https://archives.nseindia.com/content/circulars/FAOP76940.pdf",
]

# Updated from the NSE calendar/circulars cited above. This is deliberately
# finite and reviewable; it must be updated when the next year's official
# calendar and ad-hoc/special sessions are published.
CALENDARS = {
    2026: {
        "holidays": {
            "2026-01-15": "Municipal Corporation Election - Maharashtra",
            "2026-01-26": "Republic Day",
            "2026-03-03": "Holi",
            "2026-03-26": "Shri Ram Navami",
            "2026-03-31": "Shri Mahavir Jayanti",
            "2026-04-03": "Good Friday",
            "2026-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
            "2026-05-01": "Maharashtra Day",
            "2026-05-28": "Bakri Id",
            "2026-06-26": "Muharram",
            "2026-09-14": "Ganesh Chaturthi",
            "2026-10-02": "Mahatma Gandhi Jayanti",
            "2026-10-20": "Dussehra",
            "2026-11-10": "Diwali-Balipratipada",
            "2026-11-24": "Prakash Gurpurb Sri Guru Nanak Dev",
            "2026-12-25": "Christmas",
        },
        "special_sessions": {
            "2026-02-01": {
                "name": "Union Budget special live trading",
                "start": "09:15",
                "end": "15:30",
                "regular_strategy_included": False,
            },
            "2026-11-08": {
                "name": "Diwali Muhurat trading",
                "start": "14:20",
                "end": "15:40",
                "regular_strategy_included": False,
            },
        },
    }
}


def _date(value) -> date:
    return pd.Timestamp(value).date()


def audit_whole_sessions(
    observed_dates_by_symbol: dict[str, set[date]],
    start_date=None,
    end_date_exclusive=None,
    as_of=None,
    session_end: str = "15:15",
) -> dict:
    """Detect absent regular trading sessions within a requested/observed window.

    If no requested range is provided, use the union of observed dates so this
    audit can still catch gaps between files. A manifest's end date is
    interpreted as exclusive. Current-session whole-day absence is deferred
    until the regular session has ended.
    """
    as_of_ts = pd.Timestamp.now(tz=TIMEZONE) if as_of is None else pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize(TIMEZONE)
    else:
        as_of_ts = as_of_ts.tz_convert(TIMEZONE)

    all_observed = sorted({d for values in observed_dates_by_symbol.values() for d in values})
    if not observed_dates_by_symbol:
        return {
            "status": "NO_SYMBOLS", "eligible": False,
            "issues": ["no symbol datasets available for whole-session calendar audit"],
            "per_symbol": {}, "expected_sessions": [], "missing_sessions_union": [],
            "calendar_source": CALENDAR_SOURCE, "source_urls": SOURCE_URLS,
        }
    if start_date is None:
        if not all_observed:
            return {
                "status": "NO_DATE_RANGE", "eligible": False, "issues": ["no observed dates or requested date range"],
                "per_symbol": {}, "expected_sessions": [], "missing_sessions": [],
                "calendar_source": CALENDAR_SOURCE, "source_urls": SOURCE_URLS,
            }
        start = all_observed[0]
    else:
        start = _date(start_date)
    if end_date_exclusive is None:
        if not all_observed:
            return {
                "status": "NO_DATE_RANGE", "eligible": False, "issues": ["no observed dates or requested date range"],
                "per_symbol": {}, "expected_sessions": [], "missing_sessions": [],
                "calendar_source": CALENDAR_SOURCE, "source_urls": SOURCE_URLS,
            }
        end_exclusive = all_observed[-1] + timedelta(days=1)
    else:
        end_exclusive = _date(end_date_exclusive)
    if start >= end_exclusive:
        return {
            "status": "INVALID_RANGE", "eligible": False,
            "issues": [f"invalid calendar range: {start} >= {end_exclusive}"],
            "per_symbol": {}, "expected_sessions": [], "missing_sessions": [],
            "calendar_source": CALENDAR_SOURCE, "source_urls": SOURCE_URLS,
        }

    # Never audit requested future sessions as if their data should exist.
    effective_end = min(end_exclusive, as_of_ts.date() + timedelta(days=1))
    if start >= effective_end:
        return {
            "status": "NO_AUDITABLE_DATES", "eligible": False,
            "date_range": {
                "requested_start_inclusive": start.isoformat(),
                "requested_end_exclusive": end_exclusive.isoformat(),
                "audited_end_exclusive": effective_end.isoformat(),
            },
            "issues": ["requested range contains no dates at or before the audit timestamp"],
            "per_symbol": {}, "expected_sessions": [], "missing_sessions_union": [],
            "calendar_source": CALENDAR_SOURCE, "source_urls": SOURCE_URLS,
        }
    supported_years, unsupported_years = [], []
    expected, holidays, special_sessions = set(), {}, {}
    for year in range(start.year, effective_end.year + 1):
        config = CALENDARS.get(year)
        if config is None:
            unsupported_years.append(year)
            continue
        supported_years.append(year)
        year_start = max(start, date(year, 1, 1))
        year_end = min(effective_end, date(year + 1, 1, 1))
        if year_start >= year_end:
            continue
        weekdays = pd.date_range(year_start, year_end - timedelta(days=1), freq="B")
        holiday_dates = set(config["holidays"])
        special_dates = set(config["special_sessions"])
        for day_ts in weekdays:
            day = day_ts.date().isoformat()
            if day in holiday_dates or day in special_dates:
                continue
            expected.add(day_ts.date())
        for day, name in config["holidays"].items():
            if year_start <= _date(day) < year_end:
                holidays[day] = name
        for day, info in config["special_sessions"].items():
            if year_start <= _date(day) < year_end:
                special_sessions[day] = info

    current_date = as_of_ts.date()
    current_session_pending = (
        current_date in expected
        and as_of_ts.time() < pd.to_datetime(session_end).time()
    )
    pending_dates = {current_date} if current_session_pending else set()
    expected_list = sorted(d.isoformat() for d in expected)

    per_symbol, issues, missing_union, unexpected_union = {}, [], set(), set()
    for symbol, observed in sorted(observed_dates_by_symbol.items()):
        observed_in_range = {d for d in observed if start <= d < effective_end}
        outside_requested = sorted(d.isoformat() for d in observed if d < start or d >= effective_end)
        missing = expected - observed_in_range - pending_dates
        special_observed = sorted(
            d.isoformat() for d in observed_in_range if d.isoformat() in special_sessions
        )
        unexpected = sorted(
            d.isoformat() for d in observed_in_range
            if d not in expected and d.isoformat() not in special_sessions
        )
        missing_union.update(missing)
        unexpected_union.update(map(_date, unexpected))
        symbol_issues = []
        if not observed_in_range:
            symbol_issues.append("no observed sessions inside the audited date range")
            issues.append(f"{symbol}: no observed sessions inside the audited date range")
        if outside_requested:
            symbol_issues.append(f"observed dates outside the audited date range: {outside_requested}")
            issues.append(f"{symbol}: observed dates outside the audited date range: {outside_requested}")
        if missing:
            missing_text = sorted(d.isoformat() for d in missing)
            symbol_issues.append(f"missing full trading sessions: {missing_text}")
            issues.append(f"{symbol}: missing full trading sessions: {missing_text}")
        if unexpected:
            symbol_issues.append(f"observed dates are not regular trading sessions: {unexpected}")
            issues.append(f"{symbol}: observed dates are not regular trading sessions: {unexpected}")
        if special_observed:
            symbol_issues.append(f"special-session data is outside regular-session research scope: {special_observed}")
            issues.append(f"{symbol}: special-session data is outside regular-session research scope: {special_observed}")
        per_symbol[symbol] = {
            "eligible": not symbol_issues and not unsupported_years,
            "expected_sessions": len(expected),
            "observed_sessions_in_range": len(observed_in_range),
            "missing_sessions": sorted(d.isoformat() for d in missing),
            "unexpected_dates": unexpected,
            "outside_requested_range": outside_requested,
            "special_session_dates_observed": special_observed,
            "pending_current_session": sorted(d.isoformat() for d in pending_dates),
            "issues": symbol_issues,
        }

    if unsupported_years:
        issue = f"official session calendar unavailable for years: {unsupported_years}"
        issues.insert(0, issue)

    return {
        "status": "PASS" if not issues else "BLOCKED",
        "eligible": not issues and bool(observed_dates_by_symbol),
        "date_range": {
            "requested_start_inclusive": start.isoformat(),
            "requested_end_exclusive": end_exclusive.isoformat(),
            "audited_end_exclusive": effective_end.isoformat(),
        },
        "supported_years": supported_years,
        "unsupported_years": unsupported_years,
        "expected_sessions": expected_list,
        "expected_session_count": len(expected),
        "missing_sessions_union": sorted(d.isoformat() for d in missing_union),
        "unexpected_dates_union": sorted(d.isoformat() for d in unexpected_union),
        "holidays_excluded": holidays,
        "special_sessions_excluded": special_sessions,
        "pending_current_session": sorted(d.isoformat() for d in pending_dates),
        "per_symbol": per_symbol,
        "issues": issues,
        "calendar_source": CALENDAR_SOURCE,
        "source_urls": SOURCE_URLS,
        "policy": (
            "Regular weekday sessions only. Official holidays are excluded. "
            "Special weekend sessions are out-of-scope and reported; observed special-session "
            "bars block regular-session research. Unknown calendar years block eligibility. "
            "A current session is not treated as missing until its regular session_end."
        ),
    }
