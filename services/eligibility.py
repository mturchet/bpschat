"""
BPS eligibility client and guardrails.

- Guardrails (deterministic): validate grade (K1–12) and geography (Boston only)
  before calling the API. On failure, return a clear message and do not call the API.
- Client: calls BPS Discover Service (api.mybps.org) for address → AddressID,
  then HomeSchools for eligible schools. Config via environment variables.
- Never returns fabricated schools; on API/network errors returns an error result.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

import requests
from pydantic import BaseModel, Field

from . import school_data

# -----------------------------------------------------------------------------
# Config (from environment)
# -----------------------------------------------------------------------------

DEFAULT_API_BASE = "http://api.mybps.org/BPSDiscoverService/Schools.svc"


def _get_api_base() -> str:
    return os.environ.get("ELIGIBILITY_API_BASE_URL", "").strip() or DEFAULT_API_BASE


def _use_mock() -> bool:
    return os.environ.get("USE_MOCK_ELIGIBILITY", "").strip().lower() in ("1", "true", "yes")


# Optional API key for future use (BPS Discover Service currently uses no auth).
def _get_api_key() -> str:
    return os.environ.get("ELIGIBILITY_API_KEY", "").strip()


# Request timeout in seconds.
def _get_timeout() -> int:
    try:
        return max(5, int(os.environ.get("ELIGIBILITY_REQUEST_TIMEOUT", "15")))
    except ValueError:
        return 15


# -----------------------------------------------------------------------------
# Constants for guardrails
# -----------------------------------------------------------------------------

# BPS grade levels: K1, K2, 1 through 12 (school-age).
VALID_GRADES = frozenset({"K1", "K2", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"})

# Boston, MA ZIP codes (BPS serves Boston residents only).
# Includes common Boston ZIPs; 022xx are Boston (e.g. City Hall, waterfront).
BOSTON_ZIP_CODES = frozenset({
    "02101", "02102", "02103", "02104", "02105", "02106", "02107", "02108", "02109", "02110",
    "02111", "02112", "02113", "02114", "02115", "02116", "02117", "02118", "02119", "02120",
    "02121", "02122", "02123", "02124", "02125", "02126", "02127", "02128", "02129", "02130",
    "02131", "02132", "02133", "02134", "02135", "02136", "02137", "02163", "02196", "02199",
    "02201", "02202", "02203", "02204", "02205", "02206", "02207", "02210", "02211", "02212",
    "02215", "02216", "02217", "02222", "02228", "02238", "02241", "02266", "02283", "02284",
    "02293", "02295", "02297", "02298",
})


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------


class SchoolInfo(BaseModel):
    """One eligible school from the API (or mock). Enriched with display data when API returns minimal fields."""
    school_id: str = Field(description="BPS school ID")
    school_name: str = Field(description="School name")
    grade: str = Field(description="Grade level")
    eligibility: str = Field(default="", description="Eligibility / tier info")
    distance_miles: Optional[float] = Field(default=None, description="Straight-line distance if available")
    level: Optional[str] = Field(default=None, description="School level for display, e.g. Elementary, K-8, Grades 7-12")
    address: Optional[str] = Field(default=None, description="School address for display")


class EligibilitySuccess(BaseModel):
    """Successful eligibility result: list of schools (never fabricated)."""
    ok: bool = True
    schools: list[SchoolInfo] = Field(default_factory=list)
    message: str = Field(default="", description="Optional message for the user")


class EligibilityError(BaseModel):
    """Error result: guardrail or API/network failure; no schools returned."""
    ok: bool = False
    schools: list[SchoolInfo] = Field(default_factory=list)
    message: str = Field(description="User-facing error message")


def _success(schools: list[SchoolInfo], message: str = "") -> EligibilitySuccess:
    return EligibilitySuccess(ok=True, schools=schools, message=message)


def _error(message: str) -> EligibilityError:
    return EligibilityError(ok=False, message=message)


# -----------------------------------------------------------------------------
# Grade normalization (for flexible user input)
# -----------------------------------------------------------------------------


def _normalize_grade(raw: str) -> str | None:
    """Normalize user input to a BPS grade code (K1, K2, 1..12), or None if invalid."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip().upper()
    # Remove "grade" suffix if present
    s = re.sub(r"\s*GRADE\s*$", "", s, flags=re.IGNORECASE).strip()
    if s in ("K1", "K2"):
        return s
    # Ordinals: 1st, 2nd, 3rd, 4th, ... 12th
    m = re.match(r"^(\d{1,2})(?:ST|ND|RD|TH)?$", s, re.IGNORECASE)
    if m:
        num = m.group(1)
        if num and 1 <= int(num) <= 12:
            return num
    # Plain number 1-12
    if s.isdigit() and 1 <= int(s) <= 12:
        return s
    return None


# -----------------------------------------------------------------------------
# Guardrails (deterministic; no LLM)
# -----------------------------------------------------------------------------


def validate_grade(grade_raw: str) -> tuple[bool, str]:
    """
    Validate that the grade is school-age (K1–12).
    Returns (True, normalized_grade) on success, (False, user_message) on failure.
    """
    normalized = _normalize_grade(grade_raw)
    if normalized is None:
        return (
            False,
            "We only provide eligibility information for school-age children in grades K1 through 12. "
            "If your situation is different, please contact Boston Public Schools directly.",
        )
    return True, normalized


def validate_geography(zip_code: str | None, city: str | None = None, state: str | None = None) -> tuple[bool, str]:
    """
    Validate that the location is in Boston, MA (BPS serves Boston families only).
    At least one of zip_code or (city + state) should be provided.
    Returns (True, "") on success, (False, user_message) on failure.
    """
    if zip_code:
        zip_clean = re.sub(r"\D", "", zip_code.strip())
        if len(zip_clean) >= 5:
            zip_5 = zip_clean[:5]
            if zip_5 in BOSTON_ZIP_CODES:
                return True, ""
        # ZIP provided but not Boston
        return (
            False,
            "Boston Public Schools serves Boston residents only. "
            "The ZIP code you entered is not in Boston. If you live outside Boston, "
            "please contact your local school district for eligibility information.",
        )

    if city and state:
        city_norm = city.strip().lower()
        state_norm = state.strip().upper()
        if state_norm not in ("MA", "MASSACHUSETTS"):
            return (
                False,
                "Boston Public Schools serves Boston families in Massachusetts only. "
                "If you live outside Massachusetts, please contact your local school district.",
            )
        if "boston" in city_norm:
            return True, ""
        return (
            False,
            "Boston Public Schools serves Boston residents only. "
            "The city you entered is not Boston. Please enter a Boston address or ZIP code.",
        )

    return (
        False,
        "We need a Boston address or ZIP code to check eligibility. "
        "Please enter the ZIP code (or full address) where you and your child live.",
    )


# -----------------------------------------------------------------------------
# BPS Discover Service API client
# -----------------------------------------------------------------------------


def _address_matches(base_url: str, street_number: str, street: str, zip_code: str, timeout: int) -> dict[str, Any]:
    """Call AddressMatches to get AddressID(s). Returns JSON response."""
    url = f"{base_url.rstrip('/')}/AddressMatches"
    params = {
        "StreetNumber": street_number or "1",
        "Street": street or "Washington St",
        "ZipCode": zip_code,
    }
    resp = requests.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
    resp.raise_for_status()
    return resp.json()


def _home_schools(base_url: str, address_id: str, grade: str, school_year: str, timeout: int) -> dict[str, Any]:
    """Call HomeSchools to get eligible schools. Returns JSON response."""
    url = f"{base_url.rstrip('/')}/HomeSchools"
    params = {
        "schyear": school_year,
        "Grade": grade,
        "AddressID": address_id,
        "SiblingSchList": "",
        "IsAwc": "false",
    }
    resp = requests.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
    resp.raise_for_status()
    return resp.json()


def _parse_school_list(api_list: list[Any]) -> list[SchoolInfo]:
    """Convert API List of SchoolChoice objects to list of SchoolInfo. No fabrication."""
    schools: list[SchoolInfo] = []
    for item in api_list or []:
        if not isinstance(item, dict):
            continue
        school_id = item.get("SchoolID") or item.get("school_id")
        name = item.get("SchoolName") or item.get("school_name") or "Unknown"
        grade_val = item.get("Grade") or item.get("grade") or ""
        elig = item.get("Eligibility") or item.get("eligibility") or ""
        dist = item.get("StraightLineDistance") or item.get("straightLineDistance")
        if dist is None and "distance" in item:
            dist = item["distance"]
        if school_id is not None:
            schools.append(SchoolInfo(
                school_id=str(school_id),
                school_name=str(name),
                grade=str(grade_val),
                eligibility=str(elig),
                distance_miles=float(dist) if dist is not None else None,
            ))
    return schools


def _current_school_year() -> str:
    """Return current school year string (e.g. 2025). Simple: calendar year as of today."""
    import datetime
    return str(datetime.date.today().year)


# -----------------------------------------------------------------------------
# Mock client (when USE_MOCK_ELIGIBILITY=true or API unavailable)
# -----------------------------------------------------------------------------


def _mock_get_eligible_schools(zip_code: str, grade: str) -> EligibilitySuccess | EligibilityError:
    """Return a small fixed list of mock schools for testing UI/flows. No fabrication of real names."""
    raw = [
        SchoolInfo(school_id="MOCK1", school_name="[Mock] Sample Elementary", grade=grade, eligibility="Eligible", distance_miles=0.5),
        SchoolInfo(school_id="MOCK2", school_name="[Mock] Sample K-8 School", grade=grade, eligibility="Eligible", distance_miles=1.2),
        SchoolInfo(school_id="MOCK3", school_name="[Mock] Sample Academy", grade=grade, eligibility="Eligible", distance_miles=2.0),
    ]
    schools = [school_data.enrich_school_info(s) for s in raw]
    return _success(schools, message="(Mock results — set USE_MOCK_ELIGIBILITY=false and configure API to get real BPS schools.)")


# -----------------------------------------------------------------------------
# Public API: get eligible schools
# -----------------------------------------------------------------------------


def get_eligible_schools(
    *,
    grade: str,
    zip_code: str | None = None,
    street_number: str | None = None,
    street_name: str | None = None,
    city: str | None = None,
    state: str | None = None,
) -> EligibilitySuccess | EligibilityError:
    """
    Run guardrails (grade + geography), then call BPS Discover Service to get
    eligible schools. Never returns fabricated schools.

    - grade: K1, K2, or 1–12 (will be normalized).
    - For geography pass either zip_code, or (city + state). If only ZIP is
      available, street_number and street_name can be placeholders (e.g. 1,
      Washington St) for the address lookup; the API may still return schools
      for that ZIP.

    Returns either EligibilitySuccess (with schools) or EligibilityError (with
    user-facing message). On API/network errors, returns EligibilityError.
    """
    # 1) Grade guardrail
    ok_grade, grade_msg = validate_grade(grade)
    if not ok_grade:
        return _error(grade_msg)
    normalized_grade = grade_msg

    # 2) Geography guardrail — need at least ZIP or city+state
    zip_val = (zip_code or "").strip() or None
    ok_geo, geo_msg = validate_geography(zip_val, city, state)
    if not ok_geo:
        return _error(geo_msg)

    # For API we need a ZIP for Boston. If we only have city/state, we can't
    # call the BPS API (it expects address components). So require ZIP for the
    # actual call when we have no street.
    zip_for_api = re.sub(r"\D", "", zip_val or "")[:5] if zip_val else None
    if not zip_for_api and not (street_name and zip_val):
        return _error(
            "We need a Boston ZIP code to look up eligible schools. "
            "Please enter the ZIP code where you and your child live.",
        )

    if not zip_for_api:
        zip_for_api = next(iter(BOSTON_ZIP_CODES), "02101")  # fallback only if we had city=Boston

    # 3) Mock mode
    if _use_mock():
        return _mock_get_eligible_schools(zip_for_api, normalized_grade)

    # 4) Call BPS Discover Service
    base_url = _get_api_base()
    timeout = _get_timeout()
    street_num = (street_number or "1").strip() or "1"
    street = (street_name or "Washington St").strip() or "Washington St"

    try:
        # Step 1: Address lookup → AddressID
        addr_resp = _address_matches(base_url, street_num, street, zip_for_api, timeout)
        errors = addr_resp.get("Error") or []
        if errors:
            first = errors[0] if isinstance(errors[0], dict) else {}
            msg = first.get("Message") or "Address could not be verified."
            return _error(f"We couldn't verify that address. {msg} Please check and try again.")
        addr_list = addr_resp.get("List") or []
        if not addr_list:
            return _error(
                "We couldn't find that address in Boston. Please check the address or ZIP code and try again. "
                "If you only entered a ZIP code, try entering a full street address.",
            )
        first_addr = addr_list[0] if isinstance(addr_list[0], dict) else {}
        address_id = first_addr.get("AddressID") or first_addr.get("address_id")
        if not address_id:
            return _error("We couldn't get a valid address ID for that location. Please try a full Boston address.")

        # Step 2: HomeSchools
        school_year = _current_school_year()
        schools_resp = _home_schools(base_url, str(address_id), normalized_grade, school_year, timeout)
        errors = schools_resp.get("Error") or []
        if errors:
            first = errors[0] if isinstance(errors[0], dict) else {}
            msg = first.get("Message") or "Eligibility lookup failed."
            return _error(f"We couldn't load eligible schools. {msg}")
        raw_list = schools_resp.get("List") or []
        schools = _parse_school_list(raw_list)
        schools = [school_data.enrich_school_info(s) for s in schools]
        return _success(schools)

    except requests.exceptions.Timeout:
        return _error("The school lookup is taking too long. Please try again in a moment.")
    except requests.exceptions.RequestException as e:
        return _error(
            "We couldn't reach the school eligibility service. Please check your connection and try again. "
            "If the problem continues, contact Boston Public Schools directly."
        )
    except Exception:
        return _error(
            "Something went wrong while looking up schools. Please try again or contact Boston Public Schools directly."
        )
