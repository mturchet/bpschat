"""
BPS Enrollment Chatbot — Intake orchestrator.

Orchestrates the guided intake flow: grade first, then address/ZIP.
- Updates session state (in-memory, keyed by UI session via Gradio state).
- After grade: runs age/grade guardrail; on failure returns fixed message and stops.
- After address/ZIP: runs geography guardrail; on failure returns fixed message and stops.
- When both are valid: calls eligibility client and returns result for display.

Eligibility and school list come only from the eligibility module; this orchestrator
never invents schools or eligibility.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from . import eligibility

# -----------------------------------------------------------------------------
# Session state (in-memory per Gradio session)
# -----------------------------------------------------------------------------

def initial_intake_state() -> dict[str, Any]:
    """Return a fresh intake state dict. Keys: grade, zip_code, street_number, street_name, city, state."""
    return {
        "grade": None,
        "zip_code": None,
        "street_number": None,
        "street_name": None,
        "city": None,
        "state": None,
    }


# -----------------------------------------------------------------------------
# Outcome types (orchestrator step result)
# -----------------------------------------------------------------------------

# outcome kind: "need_grade" | "need_address" | "guardrail_grade_fail" | "guardrail_geo_fail" | "eligibility_result" | "converse"
# data: for guardrail_* the message str; for eligibility_result the EligibilitySuccess | EligibilityError; else None

def _outcome(kind: str, data: Any = None) -> tuple[str, Any]:
    return (kind, data)


# -----------------------------------------------------------------------------
# Extraction helpers (from user free text)
# -----------------------------------------------------------------------------

def _extract_grade_candidate(text: str) -> Optional[str]:
    """Try to extract a grade mention from free text. Returns a string to pass to validate_grade, or None."""
    if not text or not isinstance(text, str):
        return None
    t = text.strip()
    ok, val = eligibility.validate_grade(t)
    if ok:
        return val
    m = re.search(
        r"(?:grade\s+)?(K1|K2|\d{1,2}(?:st|nd|rd|th)?)\s*(?:grade)?|(K1|K2)\b|\b(?:grade\s+)?([1-9]|1[0-2])\b",
        t,
        re.IGNORECASE,
    )
    if m:
        for g in m.groups():
            if g:
                ok, val = eligibility.validate_grade(g)
                if ok:
                    return val
    return None


def _extract_zip_candidate(text: str) -> Optional[str]:
    """Extract a 5-digit ZIP from text. Prefer one that looks like Boston (021xx, 022xx)."""
    if not text or not isinstance(text, str):
        return None
    matches = re.findall(r"\b(\d{5})(?:-\d{4})?\b", text.strip())
    for m in matches:
        if m in eligibility.BOSTON_ZIP_CODES:
            return m
    return matches[0] if matches else None


# -----------------------------------------------------------------------------
# Orchestrator step
# -----------------------------------------------------------------------------

def step(state: dict[str, Any], user_message: str) -> tuple[dict[str, Any], tuple[str, Any]]:
    """
    Process one user message through the intake flow.

    - state: current intake state (grade, zip_code, ...).
    - user_message: raw user input.

    Returns (updated_state, outcome) where outcome is (kind, data):
    - ("need_grade", None): we don't have grade yet; caller may ask for grade (LLM or fixed).
    - ("need_address", None): we have grade, need address/ZIP; caller may ask for address.
    - ("guardrail_grade_fail", message): grade validation failed; show message and stop.
    - ("guardrail_geo_fail", message): geography validation failed; show message and stop.
    - ("eligibility_result", result): both valid, eligibility client was called; result is EligibilitySuccess | EligibilityError.
    - ("converse", None): could not extract the next field; caller should use LLM for reply.
    - ("already_have_both", None): state already has grade and address; caller may use LLM (e.g. follow-up question).
    """
    state = dict(state or initial_intake_state())
    user_msg = (user_message or "").strip()

    has_grade = bool(state.get("grade"))
    has_address = bool(state.get("zip_code"))

    # Already have both: do not re-run eligibility here; let app/LLM handle (e.g. "show again" or other question)
    if has_grade and has_address:
        return state, _outcome("already_have_both", None)

    # --- Missing grade: try to extract and run grade guardrail
    if not has_grade:
        grade_candidate = _extract_grade_candidate(user_msg)
        if grade_candidate is not None:
            ok, result = eligibility.validate_grade(grade_candidate)
            if not ok:
                return state, _outcome("guardrail_grade_fail", result)
            state["grade"] = result
            return state, _outcome("need_address", None)
        return state, _outcome("need_grade", None)

    # --- Have grade, missing address: try to extract and run geography guardrail, then eligibility
    zip_candidate = _extract_zip_candidate(user_msg)
    if zip_candidate is not None:
        ok_geo, geo_msg = eligibility.validate_geography(zip_candidate, None, None)
        if not ok_geo:
            return state, _outcome("guardrail_geo_fail", geo_msg)
        state["zip_code"] = zip_candidate
        result = eligibility.get_eligible_schools(
            grade=state["grade"],
            zip_code=zip_candidate,
            street_number=state.get("street_number") or "1",
            street_name=state.get("street_name") or "Washington St",
        )
        return state, _outcome("eligibility_result", result)

    return state, _outcome("converse", None)
