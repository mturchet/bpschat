import json
import unittest

from src.chat import (
    Chatbot,
    ELIGIBILITY_PROMPT,
    INTAKE_PROMPT,
    INTAKE_RESPONSE_PROMPT,
    MAP_PROMPT,
    MATCH_PROMPT,
    ORCHESTRATOR_PROMPT,
    RECOMMENDATION_RESPONSE_PROMPT,
    SCHOOL_SPECIALIST_PROMPT,
)


class StubChatbot(Chatbot):
    def __init__(self):
        super().__init__()
        self.prompt_calls = []

    def _run_agent(self, system_prompt, payload, temperature=0.2, max_tokens=450):
        self.prompt_calls.append(system_prompt)

        if system_prompt == ORCHESTRATOR_PROMPT:
            lowered = payload.lower()
            if "no preferences" in lowered:
                return json.dumps(
                    {
                        "route": "recommendation",
                        "reason": "User approved no preference flow",
                        "should_run_intake": True,
                    }
                )
            if "second grade and zip code 02199" in lowered:
                return json.dumps(
                    {
                        "route": "school_specialist",
                        "reason": "Need one preference before matching",
                        "should_run_intake": True,
                    }
                )
            return json.dumps(
                {
                    "route": "intake",
                    "reason": "Default intake",
                    "should_run_intake": True,
                }
            )

        if system_prompt == INTAKE_PROMPT:
            other_preferences = "No specific preferences" if "no preferences" in payload.lower() else ""
            return json.dumps(
                {
                    "stage": "collecting",
                    "profile": {
                        "target_grade": "2",
                        "zip_or_neighborhood": "02199",
                        "language_needs": "",
                        "special_ed_needs": "",
                        "transport_or_commute": "",
                        "after_school_needs": "",
                        "other_preferences": other_preferences,
                    },
                    "missing_fields": ["preferences"] if not other_preferences else [],
                    "next_topic": "none",
                    "next_question": "",
                    "scope_reason": "",
                    "suggested_redirect_links": [],
                }
            )

        if system_prompt == INTAKE_RESPONSE_PROMPT:
            return "What matters most to your family for school choice?"

        if system_prompt == ELIGIBILITY_PROMPT:
            return json.dumps(
                {
                    "eligibility_summary": "You are eligible to attend these BPS schools.",
                    "flags": [],
                    "required_documents": [],
                    "timing_notes": [],
                    "confidence": 0.9,
                }
            )

        if system_prompt == MATCH_PROMPT:
            return json.dumps(
                {
                    "fit_summary": "",
                    "best_school_name": "John F. Kennedy Elementary School",
                    "candidate_schools": [
                        {
                            "name": "John F. Kennedy Elementary School",
                            "neighborhood": "East Boston",
                            "grades": "PK-5",
                            "language_programs": "Spanish",
                            "special_education_services": "Inclusive classroom",
                            "after_school": "Available",
                            "hours": "8:30-3:00",
                            "rationale": "Strong fit for stated needs",
                        }
                    ],
                    "tradeoffs": [],
                    "confidence": 0.8,
                }
            )

        if system_prompt == MAP_PROMPT:
            return json.dumps(
                {
                    "access_summary": "",
                    "distance_or_commute_considerations": [],
                    "map_export_steps": [],
                    "confidence": 0.8,
                }
            )

        if system_prompt == SCHOOL_SPECIALIST_PROMPT:
            return "Before I recommend schools, what matters most to your family?"

        if system_prompt == RECOMMENDATION_RESPONSE_PROMPT:
            return "Best recommendation: John F. Kennedy Elementary School. I can also show more options."

        return ""


class TestPreferenceOrchestration(unittest.TestCase):
    def test_grade_and_zip_without_preferences_asks_followup(self):
        bot = StubChatbot()

        response = bot.get_response("Second grade and zip code 02199", history=[])

        self.assertIn("what matters most", response.lower())
        self.assertIn(SCHOOL_SPECIALIST_PROMPT, bot.prompt_calls)
        self.assertNotIn(MATCH_PROMPT, bot.prompt_calls)

    def test_no_preferences_signal_allows_recommendations(self):
        bot = StubChatbot()

        response = bot.get_response(
            "Second grade and zip code 02199, no preferences",
            history=[],
        )

        self.assertIn("Best recommendation", response)
        self.assertIn(MATCH_PROMPT, bot.prompt_calls)


if __name__ == "__main__":
    unittest.main()
