"""Strict gate combining acquisition validity and observed-session completeness."""
from __future__ import annotations


def evaluate_research_gate(completeness_audit: dict | None, acquisition_manifest: dict | None) -> dict:
    """Permit experiments only when every requested symbol passed both audits.

    Completeness alone can miss provider/OHLC validation failures because the
    acquisition step retains rejected CSVs for inspection. This gate consumes
    the acquisition manifest and fails closed when it is missing or incomplete.
    """
    audit = completeness_audit if isinstance(completeness_audit, dict) else {}
    manifest = acquisition_manifest if isinstance(acquisition_manifest, dict) else {}
    audit_eligible = audit.get("eligible") is True
    audit_issues = list(audit.get("issues") or [])
    acquisition_issues = []
    requested = manifest.get("symbols_requested")
    results = manifest.get("results")
    if not isinstance(requested, list) or not requested:
        acquisition_issues.append("acquisition manifest has no requested-symbol list")
        requested = []
    if not isinstance(results, dict):
        acquisition_issues.append("acquisition manifest has no per-symbol results")
        results = {}

    symbols = {}
    for symbol in requested:
        entry = results.get(symbol)
        if not isinstance(entry, dict):
            acquisition_issues.append(f"{symbol}: missing acquisition result")
            symbols[symbol] = {"status": "MISSING", "eligible": False}
            continue
        status = entry.get("status")
        audit_result = entry.get("audit") if isinstance(entry.get("audit"), dict) else {}
        valid = status == "SAVED" and audit_result.get("eligible") is True
        symbols[symbol] = {"status": status, "eligible": valid, "audit_issues": audit_result.get("issues", [])}
        if status != "SAVED":
            acquisition_issues.append(f"{symbol}: acquisition status {status!r}")
        elif audit_result.get("eligible") is not True:
            acquisition_issues.append(f"{symbol}: acquisition OHLCV audit did not pass")

    acquisition_eligible = not acquisition_issues
    eligible = audit_eligible and acquisition_eligible
    if eligible:
        status = "ELIGIBLE"
    elif not audit_eligible and not acquisition_eligible:
        status = "BLOCKED_DATA_QUALITY_AND_ACQUISITION"
    elif not audit_eligible:
        status = "BLOCKED_DATA_QUALITY"
    else:
        status = "BLOCKED_ACQUISITION"
    return {
        "status": status,
        "eligible": eligible,
        "acquisition_eligible": acquisition_eligible,
        "completeness_eligible": audit_eligible,
        "issues": audit_issues + acquisition_issues,
        "acquisition_issues": acquisition_issues,
        "symbols": symbols,
    }
