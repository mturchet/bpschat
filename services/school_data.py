"""
Minimal BPS school data for display.

When the eligibility API returns only school IDs or minimal fields (e.g. "Unknown"
for name), this module provides display data from a local mapping. All display
data should be updated from bostonpublicschools.org or the BPS Discover Service
when BPS rules or school list change.

- No PII; data is school reference only (name, level, address).
- Used only for display enrichment; eligibility decisions come from the API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# Path to the JSON mapping file (relative to repo root or this package).
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_BPS_SCHOOLS_FILE = _DATA_DIR / "bps_schools.json"

# In-memory cache after first load.
_schools_by_id: dict[str, dict[str, str]] = {}
_loaded = False


def _load_mapping() -> dict[str, dict[str, str]]:
    """Load schools_by_id from JSON. Returns empty dict if file missing or invalid."""
    global _schools_by_id, _loaded
    if _loaded:
        return _schools_by_id
    _loaded = True
    if not _BPS_SCHOOLS_FILE.is_file():
        _schools_by_id = {}
        return _schools_by_id
    try:
        with open(_BPS_SCHOOLS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        _schools_by_id = data.get("schools_by_id") or {}
        if not isinstance(_schools_by_id, dict):
            _schools_by_id = {}
    except (json.JSONDecodeError, OSError):
        _schools_by_id = {}
    return _schools_by_id


def get_school_display(
    school_id: str,
    api_name: Optional[str] = None,
) -> dict[str, str]:
    """
    Return display fields for a school: name, level, address.

    - Prefer api_name when it is present and not "Unknown"; otherwise use
      the local mapping for name.
    - level and address always come from the mapping when available.
    - Returns dict with keys: name, level, address (empty string if missing).
    """
    mapping = _load_mapping()
    entry = mapping.get(str(school_id)) if school_id else None
    if isinstance(entry, dict):
        name = (entry.get("name") or "").strip() or (api_name or "").strip()
        level = (entry.get("level") or "").strip()
        address = (entry.get("address") or "").strip()
    else:
        name = (api_name or "").strip()
        level = ""
        address = ""

    # If API gave a valid name, use it; else use mapping name
    if api_name and str(api_name).strip() and str(api_name).strip().lower() != "unknown":
        name = str(api_name).strip()
    elif not name and school_id:
        name = f"School {school_id}"

    return {
        "name": name or f"School {school_id}",
        "level": level,
        "address": address,
    }


def enrich_school_info(school: Any) -> Any:
    """
    Enrich a SchoolInfo (or dict with school_id, school_name) with display data
    from the mapping when the API provided minimal fields.

    - If school has school_name and it is not "Unknown", keeps it and adds
      level/address from mapping when available.
    - If school_name is missing or "Unknown", replaces with name from mapping
      (and level, address) when available.

    Returns the same type as input: SchoolInfo if given SchoolInfo, else a new
    dict with keys school_id, school_name, grade, eligibility, distance_miles,
    level, address (level and address added when from mapping).
    """
    sid = getattr(school, "school_id", None) or (school.get("school_id") if isinstance(school, dict) else None)
    api_name = getattr(school, "school_name", None) or (school.get("school_name") if isinstance(school, dict) else None)
    display = get_school_display(sid or "", api_name)

    update = {
        "school_name": display["name"],
        "level": display["level"] or None,
        "address": display["address"] or None,
    }

    if hasattr(school, "model_copy"):
        # Pydantic v2 model (SchoolInfo)
        try:
            return school.model_copy(update=update)
        except Exception:
            return school
    if isinstance(school, dict):
        out = dict(school)
        out.update(update)
        return out
    return school


def get_all_mapped_school_ids() -> set[str]:
    """Return set of school IDs that have display data in the mapping (for tests or UI)."""
    return set(_load_mapping().keys())
