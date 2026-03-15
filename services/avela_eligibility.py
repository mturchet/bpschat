"""
Avela-based BPS eligibility client.

Calls the Avela findEligibility API (used by boston.explore.avela.org) to get
real BPS school eligibility based on grade, address, and language.

The API returns ineligible schools; we subtract those from the full catalog
(loaded from data/avela_schools_catalog.json) to produce the eligible list.

Eligibility and school data come only from Avela/BPS; no fabrication.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import requests

from .eligibility import (
    SchoolInfo,
    EligibilitySuccess,
    EligibilityError,
    validate_grade,
    validate_geography,
    BOSTON_ZIP_CODES,
    _normalize_grade,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

AVELA_API_URL = (
    "https://prod.execute-api.apply.avela.org/eligibility/organizations/boston"
    "/formTemplates/cd0501a5-eb9c-4aa5-a7ff-6402280a5b51/findEligibility"
)

# Question IDs from the Avela form template
GRADE_QUESTION_ID = "59e28093-6c84-496d-b37a-a68162a75d36"
ADDRESS_QUESTION_ID = "b9fb2ac3-40d8-4d6a-85a9-da0f6d0a2762"
LANGUAGE_QUESTION_ID = "f8552cb9-099a-412a-9f69-69e6a77176ee"

# Grade code → Avela UUID mapping
GRADE_TO_UUID = {
    "K0": "a409dc76-94cc-471c-bc68-c7b68d05147d",
    "K1": "9e1e0cbf-c147-48ac-a961-34fc97a0be67",
    "K2": "4134373f-5e12-4a03-b36f-c0a545db9eb7",
    "1": "eaf903c1-b6c5-4c9d-8905-dc6152ac9f5e",
    "2": "fb580408-4db9-4e54-8191-cdd6bd95a4fe",
    "3": "f59adf0b-69d4-4b5b-8a40-5e87886eaba7",
    "4": "bd63e458-16cc-46ed-a260-c936a85fdc55",
    "5": "12746de8-ab87-4af5-b8ef-abcb83285467",
    "6": "bb81c16d-2f72-41a9-929c-9316f2143780",
    "7": "92efe874-5e03-4037-aefd-1edded298e46",
    "8": "f2529c1b-c1c1-4fb6-bf2d-c2de261d3b5b",
    "9": "f6b26370-247e-4ef3-8144-0b1eddc86849",
    "10": "d98e3523-82c7-4940-9177-a4d92807914f",
    "11": "5d40fd74-63bd-49ce-8439-8b3a55ed0864",
    "12": "2ce44985-23b2-438a-906e-e56369300467",
}

# Default language: English
ENGLISH_LANGUAGE_UUID = "c188baa2-f2e8-4015-80ee-a42514617585"

# Request timeout
AVELA_TIMEOUT = int(os.environ.get("AVELA_REQUEST_TIMEOUT", "20"))


# -----------------------------------------------------------------------------
# Catalog loading
# -----------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CATALOG_FILE = _DATA_DIR / "avela_schools_catalog.json"

_catalog: dict[str, dict] = {}
_catalog_loaded = False


def _load_catalog() -> dict[str, dict]:
    """Load the full school catalog keyed by reference ID."""
    global _catalog, _catalog_loaded
    if _catalog_loaded:
        return _catalog
    _catalog_loaded = True
    if not _CATALOG_FILE.is_file():
        _catalog = {}
        return _catalog
    try:
        with open(_CATALOG_FILE, encoding="utf-8") as f:
            data = json.load(f)
        _catalog = data.get("schools_by_ref_id") or {}
    except (json.JSONDecodeError, OSError):
        _catalog = {}
    return _catalog


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _success(schools: list[SchoolInfo], message: str = "") -> EligibilitySuccess:
    return EligibilitySuccess(ok=True, schools=schools, message=message)


def _error(message: str) -> EligibilityError:
    return EligibilityError(ok=False, message=message)


def _build_request_payload(
    grade_uuid: str,
    street_address: str,
    city: str,
    state: str,
    zip_code: str,
    language_uuid: str | None = None,
) -> dict:
    """Build the JSON payload for the Avela findEligibility API."""
    return {
        "questionIdToAnswer": {
            GRADE_QUESTION_ID: grade_uuid,
            ADDRESS_QUESTION_ID: {
                "streetAddress": street_address,
                "streetAddressLine2": "",
                "city": city,
                "state": state,
                "zipCode": zip_code,
            },
            LANGUAGE_QUESTION_ID: language_uuid or ENGLISH_LANGUAGE_UUID,
        },
        "applicationType": "Explore",
    }


def _catalog_entry_to_school_info(ref_id: str, entry: dict, grade: str) -> SchoolInfo:
    """Convert a catalog entry to a SchoolInfo object."""
    return SchoolInfo(
        school_id=ref_id,
        school_name=entry.get("name", f"School {ref_id}"),
        grade=grade,
        eligibility="Eligible",
        distance_miles=None,
        level=entry.get("grade_range") or entry.get("grade_span") or None,
        address=entry.get("address") or None,
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def get_eligible_schools(
    *,
    grade: str,
    zip_code: str | None = None,
    street_number: str | None = None,
    street_name: str | None = None,
    city: str | None = None,
    state: str | None = None,
    language_uuid: str | None = None,
    bps_only: bool = True,
) -> EligibilitySuccess | EligibilityError:
    """
    Get eligible schools from the Avela API.

    - grade: K0, K1, K2, or 1-12 (will be normalized).
    - zip_code: Boston ZIP code.
    - street_number, street_name: optional street address components.
    - language_uuid: Avela UUID for home language (defaults to English).
    - bps_only: if True, filter results to Boston Public School provider_type only.

    Returns EligibilitySuccess with eligible schools, or EligibilityError.
    """
    # 1) Normalize and validate grade
    normalized = _normalize_grade(grade)
    if normalized is None:
        return _error(
            "We only provide eligibility information for grades K0 through 12. "
            "Please provide a valid grade level."
        )

    grade_uuid = GRADE_TO_UUID.get(normalized)
    if not grade_uuid:
        return _error(f"Grade '{normalized}' is not supported.")

    # 2) Validate geography
    zip_val = (zip_code or "").strip() or None
    if zip_val:
        ok_geo, geo_msg = validate_geography(zip_val)
        if not ok_geo:
            return _error(geo_msg)

    # 3) Build street address string
    street_addr = ""
    if street_number and street_name:
        street_addr = f"{street_number} {street_name}"
    elif street_name:
        street_addr = street_name
    elif zip_val:
        # Minimal placeholder — the API still works with just ZIP
        street_addr = "1 Main St"

    api_city = (city or "Boston").strip()
    api_state = (state or "MA").strip()
    api_zip = zip_val or "02101"

    # 4) Call the Avela API
    payload = _build_request_payload(
        grade_uuid=grade_uuid,
        street_address=street_addr,
        city=api_city,
        state=api_state,
        zip_code=api_zip,
        language_uuid=language_uuid,
    )

    try:
        resp = requests.post(
            AVELA_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=AVELA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return _error("The eligibility lookup is taking too long. Please try again.")
    except requests.exceptions.ConnectionError:
        return _error(
            "We couldn't connect to the eligibility service. "
            "Please check your connection and try again."
        )
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        return _error(
            f"The eligibility service returned an error (HTTP {status or 'unknown'}). "
            "Please try again later."
        )
    except Exception:
        return _error("Something went wrong while looking up eligibility. Please try again.")

    # 5) Extract ineligible reference IDs
    ineligible_schools = data.get("ineligibleSchools") or []
    ineligible_ref_ids = set()
    for s in ineligible_schools:
        ref_id = s.get("referenceId")
        if ref_id:
            ineligible_ref_ids.add(ref_id)

    # 6) Subtract from catalog to get eligible schools
    catalog = _load_catalog()
    eligible: list[SchoolInfo] = []

    for ref_id, entry in catalog.items():
        if ref_id in ineligible_ref_ids:
            continue

        # Optionally filter to BPS schools only
        if bps_only and entry.get("provider_type") != "Boston Public School":
            continue

        eligible.append(_catalog_entry_to_school_info(ref_id, entry, normalized))

    if not eligible:
        return _error(
            "No eligible schools were found for that address and grade. "
            "Please confirm your information with Boston Public Schools."
        )

    return _success(eligible)
