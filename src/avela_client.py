"""
Avela eligibility client for the BPS chatbot.

Calls the Avela findEligibility API (boston.explore.avela.org backend) to get
real BPS school eligibility. Returns schools in the format expected by the
Chatbot recommendation pool.

The API returns ineligible schools; we subtract from the full catalog to get eligible ones.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Avela API constants
# ---------------------------------------------------------------------------

AVELA_API_URL = (
    "https://prod.execute-api.apply.avela.org/eligibility/organizations/boston"
    "/formTemplates/cd0501a5-eb9c-4aa5-a7ff-6402280a5b51/findEligibility"
)

GRADE_QUESTION_ID = "59e28093-6c84-496d-b37a-a68162a75d36"
ADDRESS_QUESTION_ID = "b9fb2ac3-40d8-4d6a-85a9-da0f6d0a2762"
LANGUAGE_QUESTION_ID = "f8552cb9-099a-412a-9f69-69e6a77176ee"

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

ENGLISH_UUID = "c188baa2-f2e8-4015-80ee-a42514617585"

AVELA_TIMEOUT = int(os.environ.get("AVELA_REQUEST_TIMEOUT", "20"))

# Boston ZIP codes for validation
BOSTON_ZIPS = {
    "02101", "02102", "02103", "02104", "02105", "02106", "02107", "02108",
    "02109", "02110", "02111", "02112", "02113", "02114", "02115", "02116",
    "02117", "02118", "02119", "02120", "02121", "02122", "02123", "02124",
    "02125", "02126", "02127", "02128", "02129", "02130", "02131", "02132",
    "02133", "02134", "02135", "02136", "02137", "02163", "02196", "02199",
    "02201", "02202", "02203", "02204", "02205", "02206", "02207", "02210",
    "02211", "02212", "02215", "02216", "02217", "02222", "02228", "02238",
    "02241", "02266", "02283", "02284", "02293", "02295", "02297", "02298",
}


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CATALOG_FILE = _DATA_DIR / "avela_schools_catalog.json"

_catalog: dict = {}
_catalog_loaded = False


def _load_catalog() -> dict:
    global _catalog, _catalog_loaded
    if _catalog_loaded:
        return _catalog
    _catalog_loaded = True
    if not _CATALOG_FILE.is_file():
        return _catalog
    try:
        with open(_CATALOG_FILE, encoding="utf-8") as f:
            data = json.load(f)
        _catalog = data.get("schools_by_ref_id") or {}
    except (json.JSONDecodeError, OSError):
        _catalog = {}
    return _catalog


# ---------------------------------------------------------------------------
# Grade normalization
# ---------------------------------------------------------------------------

def _normalize_grade(raw: str) -> Optional[str]:
    """Normalize user input like '3rd grade', 'K2', 'kindergarten' to Avela grade code."""
    if not raw:
        return None
    s = raw.strip().upper()
    s = re.sub(r"\s*GRADE\s*", "", s, flags=re.IGNORECASE).strip()

    if s in ("K0", "K1", "K2"):
        return s
    if "KINDERGARTEN" in s.upper():
        return "K2"  # default kindergarten to K2

    m = re.match(r"^(\d{1,2})(?:ST|ND|RD|TH)?$", s, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 12:
            return str(num)

    if s.isdigit() and 1 <= int(s) <= 12:
        return s

    return None


def _extract_zip(text: str) -> Optional[str]:
    """Extract a 5-digit Boston ZIP from text."""
    if not text:
        return None
    matches = re.findall(r"\b(\d{5})\b", text)
    for m in reversed(matches):
        if m in BOSTON_ZIPS:
            return m
    return matches[-1] if matches else None


def _extract_street_address(text: str) -> Optional[str]:
    """Try to extract a street address from text."""
    if not text:
        return None
    suffixes = r"(?:st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|ct|court|pl|place|way|pkwy|parkway)"
    pattern = re.compile(rf"\b(\d{{1,6}}\s+[A-Za-z0-9][A-Za-z0-9 .'\-]*?\s+{suffixes})\b", re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_school(entry: dict, user_zip: str = "", user_grade: str = "") -> float:
    """Score a school for ranking. Higher = better."""
    score = 0.0

    # Quality tier
    tier_str = entry.get("bps_quality_tier", "")
    if tier_str:
        try:
            tier = int(tier_str)
            score += max(0, (5 - tier) * 25)
        except ValueError:
            pass

    # Same ZIP
    if user_zip and user_zip in entry.get("address", ""):
        score += 40

    # Grade range breadth
    grade_span = entry.get("grade_span", "")
    if grade_span and user_grade:
        parts = [p.strip().upper() for p in grade_span.replace("–", "-").split("-")]
        if len(parts) == 2:
            grade_map = {"K0": -1, "K1": 0, "K2": 0}
            try:
                high = grade_map.get(parts[1], int(parts[1]) if parts[1].isdigit() else 6)
                user_g = grade_map.get(user_grade.upper(), int(user_grade) if user_grade.isdigit() else 5)
                years = high - user_g
                if years >= 6:
                    score += 20
                elif years >= 3:
                    score += 10
            except ValueError:
                pass

    # After-school
    if entry.get("after_school_program"):
        score += 10

    # Dual language
    if entry.get("dual_language"):
        score += 5

    return score


# ---------------------------------------------------------------------------
# Convert catalog entry to recommendation pool format
# ---------------------------------------------------------------------------

def _catalog_to_recommendation(entry: dict, user_grade: str = "") -> dict:
    """Convert a catalog entry to Chatbot recommendation_pool format."""
    # Extract neighborhood from address
    address = entry.get("address", "")
    neighborhood = ""
    # Address format: "165 Webster St East Boston, MA 02128"
    parts = address.split(",")
    if len(parts) >= 2:
        # Take the part before the comma, strip the street portion
        before_comma = parts[0].strip()
        # Try to extract neighborhood (words after the street suffix)
        suffix_match = re.search(
            r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Pl|Place|Way|Pkwy|Hwy)\s+(.+)$",
            before_comma, re.IGNORECASE
        )
        if suffix_match:
            neighborhood = suffix_match.group(1).strip()

    return {
        "name": entry.get("name", ""),
        "neighborhood": neighborhood,
        "grades": entry.get("grade_span", ""),
        "language_programs": entry.get("dual_language", "") or "",
        "special_education_services": entry.get("specialized_education_programs", "") or "",
        "after_school": entry.get("after_school_program", "") or "",
        "hours": entry.get("hours_of_operation", "") or "",
        "rationale": "",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_eligible_schools(
    target_grade: str,
    zip_or_neighborhood: str,
    street_address: str = "",
    language_uuid: str = "",
) -> list[dict]:
    """
    Call the Avela eligibility API and return eligible BPS schools
    in the Chatbot recommendation_pool format.

    Returns a list of school dicts sorted by score, or empty list on failure.
    """
    # 1) Normalize grade
    grade = _normalize_grade(target_grade)
    if not grade:
        return []

    grade_uuid = GRADE_TO_UUID.get(grade)
    if not grade_uuid:
        return []

    # 2) Extract ZIP
    zip_code = _extract_zip(zip_or_neighborhood)
    if not zip_code:
        # Maybe they gave a neighborhood name — use a default placeholder
        # The API still needs a ZIP; we can't proceed without one
        return []

    # 3) Build street address
    addr = street_address.strip() if street_address else ""
    if not addr:
        addr = _extract_street_address(zip_or_neighborhood) or "1 Main St"

    # 4) Call the API
    payload = {
        "questionIdToAnswer": {
            GRADE_QUESTION_ID: grade_uuid,
            ADDRESS_QUESTION_ID: {
                "streetAddress": addr,
                "streetAddressLine2": "",
                "city": "Boston",
                "state": "MA",
                "zipCode": zip_code,
            },
            LANGUAGE_QUESTION_ID: language_uuid or ENGLISH_UUID,
        },
        "applicationType": "Explore",
    }

    try:
        resp = requests.post(
            AVELA_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=AVELA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    # 5) Get ineligible reference IDs
    ineligible_ids = set()
    for s in data.get("ineligibleSchools", []):
        ref = s.get("referenceId")
        if ref:
            ineligible_ids.add(ref)

    # 6) Subtract from catalog, filter to BPS only
    catalog = _load_catalog()
    eligible = []

    for ref_id, entry in catalog.items():
        if ref_id in ineligible_ids:
            continue
        if entry.get("provider_type") != "Boston Public School":
            continue

        school = _catalog_to_recommendation(entry, grade)
        score = _score_school(entry, zip_code, grade)
        eligible.append((score, school))

    # 7) Sort by score descending
    eligible.sort(key=lambda x: x[0], reverse=True)

    # 8) Add rationale to top school
    if eligible:
        top_score, top_school = eligible[0]
        rationale_parts = []
        entry = None
        for ref_id, e in catalog.items():
            if e.get("name") == top_school["name"]:
                entry = e
                break
        if entry:
            tier = entry.get("bps_quality_tier", "")
            if tier and tier.isdigit() and int(tier) <= 2:
                rationale_parts.append(f"Tier {tier} quality rating")
            if zip_code and zip_code in entry.get("address", ""):
                rationale_parts.append("located in your neighborhood")
            if entry.get("after_school_program"):
                rationale_parts.append("offers after-school programs")
            if entry.get("dual_language"):
                rationale_parts.append("dual-language program available")
        if rationale_parts:
            top_school["rationale"] = "Top pick: " + ", ".join(rationale_parts)
        eligible[0] = (top_score, top_school)

    return [school for _, school in eligible]
