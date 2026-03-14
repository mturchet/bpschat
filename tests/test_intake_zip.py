"""
Tests for ZIP extraction and Boston validation in the intake flow.

These verify that:
- Boston ZIPs (e.g. 02298, 02119) are recognized even with different formatting.
- Non-Boston ZIPs (e.g. 02142 Cambridge) get a clear "not in Boston" message.
- When the user has already provided grade + address, the flow returns already_have_both
  (so the app can tell the LLM not to re-ask for grade/address).
"""

import os
import unittest

# Use mock eligibility so we don't call the real BPS API
os.environ["USE_MOCK_ELIGIBILITY"] = "1"

from services.intake import initial_intake_state, step


def _state_with_grade(grade: str = "3"):
    s = initial_intake_state()
    s["grade"] = grade
    return s


class TestBostonZipExtraction(unittest.TestCase):
    """Boston ZIPs should be extracted and accepted (eligibility_result)."""

    def test_plain_boston_zip_02298(self):
        state = _state_with_grade()
        new_state, (kind, data) = step(state, "02298")
        self.assertEqual(kind, "eligibility_result", msg="02298 should be accepted as Boston")
        self.assertEqual(new_state.get("zip_code"), "02298")

    def test_plain_boston_zip_02119(self):
        state = _state_with_grade()
        new_state, (kind, data) = step(state, "02119")
        self.assertEqual(kind, "eligibility_result")
        self.assertEqual(new_state.get("zip_code"), "02119")

    def test_zip_with_colon_no_space(self):
        """Format like ZIP:02298 (no word boundary before 02298) should still be recognized."""
        state = _state_with_grade()
        new_state, (kind, data) = step(state, "ZIP:02298")
        self.assertEqual(kind, "eligibility_result", msg="ZIP:02298 should be extracted and accepted")
        self.assertEqual(new_state.get("zip_code"), "02298")

    def test_zip_with_equals(self):
        state = _state_with_grade()
        new_state, (kind, data) = step(state, "zip=02298")
        self.assertEqual(kind, "eligibility_result")
        self.assertEqual(new_state.get("zip_code"), "02298")

    def test_correction_last_boston_zip_wins(self):
        """When user says '02142 then 02298', we should use 02298 (last Boston zip)."""
        state = _state_with_grade()
        new_state, (kind, data) = step(state, "I had 02142 but it's actually 02298")
        self.assertEqual(kind, "eligibility_result")
        self.assertEqual(new_state.get("zip_code"), "02298")

    def test_two_bostons_last_wins(self):
        """When user says '02119 and 02298', we prefer the last Boston zip."""
        state = _state_with_grade()
        new_state, (kind, data) = step(state, "Maybe 02119 or 02298")
        self.assertEqual(kind, "eligibility_result")
        self.assertEqual(new_state.get("zip_code"), "02298")


class TestNonBostonZip(unittest.TestCase):
    """Non-Boston ZIP (e.g. 02142 Cambridge) should get guardrail_geo_fail."""

    def test_cambridge_zip_02142(self):
        state = _state_with_grade()
        new_state, (kind, data) = step(state, "02142")
        self.assertEqual(kind, "guardrail_geo_fail")
        self.assertIn("not in Boston", data)
        self.assertIsNone(new_state.get("zip_code"))


class TestAlreadyHaveBoth(unittest.TestCase):
    """When state already has grade and zip, outcome should be already_have_both (no re-ask)."""

    def test_already_have_both_returns_correct_outcome(self):
        state = initial_intake_state()
        state["grade"] = "5"
        state["zip_code"] = "02298"
        new_state, (kind, data) = step(state, "Is 02298 in Boston? Can you show my schools again?")
        self.assertEqual(kind, "already_have_both")
        self.assertIsNone(data)


class TestFullAddressExtraction(unittest.TestCase):
    """When the user provides full address text, capture street fields and ZIP."""

    def test_full_address_with_commas(self):
        state = _state_with_grade("4")
        new_state, (kind, data) = step(state, "100 Warren St, Boston, MA 02119")
        self.assertEqual(kind, "eligibility_result")
        self.assertEqual(new_state.get("zip_code"), "02119")
        self.assertEqual(new_state.get("street_number"), "100")
        self.assertEqual(new_state.get("street_name"), "Warren St")

    def test_full_address_in_sentence(self):
        state = _state_with_grade("2")
        new_state, (kind, data) = step(state, "I live at 12 Beacon Street Boston MA 02108")
        self.assertEqual(kind, "eligibility_result")
        self.assertEqual(new_state.get("zip_code"), "02108")
        self.assertEqual(new_state.get("street_number"), "12")
        self.assertEqual(new_state.get("street_name"), "Beacon Street")


if __name__ == "__main__":
    unittest.main()
