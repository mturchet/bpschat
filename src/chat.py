import csv
import json
import re
import tempfile
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import quote_plus

from huggingface_hub import InferenceClient
from openai import OpenAI

from config import (
    BASE_MODEL,
    HF_TOKEN,
    MY_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    LIGHT_MODEL,
)

from src.avela_client import get_eligible_schools as avela_get_schools

CORE_FACTS = """You are a helpful assistant for Boston Public Schools (BPS) enrollment guidance.
You help parents and legal guardians understand enrollment options, eligibility basics, and tradeoffs.
You must be warm, concise, and transparent about uncertainty.
Always remind families to confirm final eligibility with bostonpublicschools.org.
You help families in Boston find the right public school for their children.
You have knowledge about Boston Public Schools, including:
- School locations and neighborhoods
- Grade levels offered (K0, K1, K2, elementary, middle, high school)
- Language programs (dual language, ESL, sheltered English)
- Special education services
- Application and enrollment processes
- Transportation and bus routes
- After-school programs
When helping families:
- Ask clarifying questions about neighborhood, child's age, and preferences
- Provide specific school recommendations when possible
- Be honest when unsure and direct users to bostonpublicschools.org
- Stay warm and supportive because choosing a school is a big family decision
Key facts:
- Boston uses a home-based assignment system where families get a list of schools based on their address
- Families can register at any Welcome Center or online
- Registration typically opens in January for the following school year
- The BPS website is bostonpublicschools.org
"""

GREETING_PROMPT = f"""{CORE_FACTS}
You are the Welcome Agent.
Write a warm first message in plain text for a family starting enrollment chat.

Rules:
- Welcome the family warmly.
- Explain that to find eligible schools, you need two things: their child's target grade level (K0, K1, K2, or grades 1-12) and their Boston ZIP code or neighborhood.
- Keep to 3-4 short sentences. Be warm and concise.
"""

INTAKE_PROMPT = f"""{CORE_FACTS}
You are the Intake Agent.

Return ONLY valid JSON with this schema:
{{
  "stage": "collecting|ready_for_recommendations|filtering|out_of_scope|general_info",
  "profile": {{
    "target_grade": "",
    "zip_or_neighborhood": "",
    "language_needs": "",
    "special_ed_needs": "",
    "transport_or_commute": "",
    "after_school_needs": "",
    "other_preferences": ""
  }},
  "missing_fields": [],
  "next_topic": "",
  "next_question": "",
  "scope_reason": "",
  "suggested_redirect_links": []
}}

Rules:
- Set stage=ready_for_recommendations AS SOON AS target_grade + zip_or_neighborhood are both known. Do not keep collecting optional preferences.
- Only set stage=collecting if target_grade or zip_or_neighborhood is still missing.
- Set stage=filtering when the user has asked for help narrowing down or filtering their options.
- If user provides preferences along with grade+ZIP in the same message, capture ALL of them AND set stage=ready_for_recommendations.
- If user is outside MA/Boston scope, use stage=out_of_scope.
- Once stage=ready_for_recommendations, set next_topic=none.
- Use Current Memory JSON in the user payload.
- Do NOT re-ask a topic listed in memory.asked_topics unless user asked to revisit it.
"""

INTAKE_RESPONSE_PROMPT = f"""{CORE_FACTS}
You are the Intake Response Agent.
Given intake JSON and user message, write a warm, dynamic response.

Rules:
- NEVER re-summarize or re-list what the user has already told you. They know what they said.
- NEVER say "to recap", "to summarize", "so far I have", or list back their grade/ZIP/preferences.
- If stage=collecting: ask only for the specific missing required field (grade, address, or language). One short sentence acknowledging what they shared, then ask for what's missing.
- If stage=ready_for_recommendations or filtering: acknowledge briefly (one sentence), then either ask ONE focused preference question or offer to show schools. Not both.
- If stage=out_of_scope: gently explain scope and include suggested links.
- If the user asks "what else should I share?" or similar, briefly list the preference categories (language programs, special ed, after-school, sports, commute, school hours) without re-stating what they already told you.
- Return plain text only.
- Keep response under 80 words.
"""

ELIGIBILITY_PROMPT = f"""{CORE_FACTS}
You are the Eligibility Agent.
Return ONLY valid JSON:
{{
  "eligibility_summary": "",
  "flags": [],
  "required_documents": [],
  "timing_notes": [],
  "confidence": 0.0
}}
"""

MATCH_PROMPT = f"""{CORE_FACTS}
You are the School-Match Agent.
Return ONLY valid JSON:
{{
  "fit_summary": "",
  "best_school_name": "",
  "candidate_schools": [
    {{
      "name": "",
      "neighborhood": "",
      "grades": "",
      "language_programs": "",
      "special_education_services": "",
      "after_school": "",
      "hours": "",
      "rationale": ""
    }}
  ],
  "tradeoffs": [],
  "confidence": 0.0
}}

Rules:
- Provide 4-12 candidate_schools when possible.
- If uncertain, still provide best-effort BPS options and mention uncertainty in rationale.
"""

MAP_PROMPT = f"""{CORE_FACTS}
You are the Map & Access Agent.
Return ONLY valid JSON:
{{
  "access_summary": "",
  "distance_or_commute_considerations": [],
  "map_export_steps": [],
  "confidence": 0.0
}}
"""

MATCH_RECOVERY_PROMPT = f"""{CORE_FACTS}
You are a Recovery Match Agent.
Return ONLY valid JSON:
{{
  "candidate_schools": [
    {{
      "name": "",
      "neighborhood": "",
      "grades": "",
      "language_programs": "",
      "special_education_services": "",
      "after_school": "",
      "hours": "",
      "rationale": ""
    }}
  ],
  "best_school_name": ""
}}
Rules:
- Return at least 4 schools.
"""

ORCHESTRATOR_PROMPT = f"""{CORE_FACTS}
You are the Orchestrator Agent for a multi-agent enrollment assistant.
Decide which specialist should handle the next turn.

Return ONLY valid JSON:
{{
  "route": "welcome|intake|school_specialist|recommendation|recommendation_followup|export_csv|general_info",
  "reason": "",
  "should_run_intake": true
}}

Rules:
- Route to `welcome` only when the conversation is just starting.
- Route to `recommendation` when the user says anything like "show me the schools", "show me the list", "see my options", "yes show me", "just show me", or any request to see results — AND grade+ZIP are known.
- Route to `recommendation` when target_grade AND zip_or_neighborhood are known AND the user indicates they want results.
- Route to `school_specialist` ONLY when the user explicitly says they want help filtering or want to answer more preference questions.
- Route to `recommendation_followup` for questions about existing options (compare, map, more choices, tradeoffs).
- Route to `export_csv` when user asks to export/download list.
- Route to `intake` when required fields (grade, ZIP) are still missing.
- Keep `should_run_intake=true` for most turns so intake memory stays current.
- When in doubt and required fields are present, prefer `recommendation` over `school_specialist`.
"""

SCHOOL_SPECIALIST_PROMPT = f"""{CORE_FACTS}
You are the School Specialist Agent.
Your job is to gather one decision-critical family preference before final recommendations.

Rules:
- Ask exactly one focused follow-up question.
- NEVER re-summarize or re-list what the user has already told you. They know what they said.
- NEVER say "to recap" or "to summarize" or "so far we know".
- Prefer one of: language needs, special education supports, commute/transport, after-school care, or any other priority.
- If user already said they have no preference, acknowledge briefly and proceed to show schools.
- Keep response under 50 words.
- Return plain text only.
"""

RECOMMENDATION_RESPONSE_PROMPT = f"""{CORE_FACTS}
You are the Recommendation Agent.
You receive structured outputs from Eligibility, School-Match, and Map agents.

Rules:
- Present one best recommendation plus up to three additional eligible schools.
- Mention why the best recommendation fits.
- Be concise and easy to scan with bullets.
- End with one short line offering follow-up actions (more options, compare, map, export).
- Return plain text only.
"""

FOLLOWUP_ACTION_PROMPT = f"""{CORE_FACTS}
You are the Recommendation Follow-up Planner Agent.
Given user request and current recommendation state, choose the follow-up action.

Return ONLY valid JSON:
{{
  "action": "show_more|compare|map|summary",
  "indexes": [],
  "reason": ""
}}

Rules:
- Use `show_more` for requests for more schools/options.
- Use `compare` when user asks to compare schools; include up to 3 indexes.
- Use `map` for route/map/distance questions.
- Use `summary` when unsure.
"""

FOLLOWUP_RESPONSE_PROMPT = f"""{CORE_FACTS}
You are the Recommendation Follow-up Agent.
Use the provided action payload and school/map data to answer clearly.

Rules:
- Be concise and direct. No re-summarizing the conversation.
- NEVER say "to recap", "let me summarize", or re-list previously stated preferences.
- Focus on answering the specific follow-up request (show more, compare, map).
- Keep response under 100 words.
- Return plain text only.
"""

EXPORT_RESPONSE_PROMPT = f"""{CORE_FACTS}
You are the Export Agent.
Confirm CSV export completion using the provided file path and briefly explain what was exported.
Return plain text only.
"""

# ---------------------------------------------------------------------------
# Fast-path templates and light LLM prompts
# ---------------------------------------------------------------------------

GREETING_TEMPLATE = (
    "Hi there! Welcome to the Boston Public Schools enrollment assistant. "
    "I'm here to help you and your family find the right school.\n\n"
    "To look up which schools your child is eligible for, I'll need a few things:\n\n"
    "1. **Grade applying for** — K0 (3 yrs old), K1 (4 yrs old), K2 (5 yrs old), or grades 1–12\n"
    "2. **Your home address** — Because BPS uses a home-based assignment system, "
    "your street address helps me provide the most accurate results. If you're not "
    "comfortable sharing your full address, I can still provide near-accurate results "
    "with just your ZIP code.\n"
    "3. **Home language** — What language do parents or primary caregivers use to "
    "communicate with your child at home? This helps identify bilingual and language "
    "support programs your child may be eligible for.\n\n"
    "Feel free to share all of this at once, or one piece at a time — whatever's easiest!"
)

CHOICE_TEMPLATE = (
    "Great news — I've found eligible schools for your child!\n\n"
    "I can show you the full list right now, or I can help you narrow things down "
    "to find the best fit. If you'd like my help, here are some things that can "
    "make a real difference in finding the right school:\n\n"
    "  — **Language programs** (dual language, bilingual, ESL)\n"
    "  — **Special education services** (IEP, Section 504, ABA — feel free to ask me what any of these mean)\n"
    "  — **After-school programs & extracurriculars**\n"
    "  — **Sports**\n"
    "  — **Commute or transportation needs**\n"
    "  — **School schedule / hours**\n\n"
    "Share as many or as few as you'd like — all at once is perfectly fine. "
    "If you're not sure about something on this list, just ask and I'll explain it.\n\n"
    "At any point in our conversation, you can ask me for the current list of schools "
    "that fit your preferences. The more you share, the better I can match — but "
    "there's no pressure to go through everything."
)

NO_SCHOOLS_TEMPLATE = (
    "I wasn't able to find eligible schools for that combination of grade and address. "
    "This could mean the address wasn't recognized in the BPS system. "
    "Please double-check your information, or contact a BPS Welcome Center directly "
    "at **(617) 635-9010** — they'll be happy to help."
)

# Light LLM prompt: used for single-call natural responses during early turns.
LIGHT_RESPONSE_PROMPT = f"""{CORE_FACTS}
You are a warm, concise enrollment assistant.

You will be given:
- What the user just said
- What information you already have
- What information is still missing

Rules:
- Acknowledge what the user shared naturally (one short sentence).
- Ask for the specific missing information in a warm, conversational way.
- For address: explain BPS uses home-based assignment, so a home address gives the most accurate results, but a ZIP code works too if they prefer not to share their full address.
- For grade: say "grade applying for" and mention the options (K0, K1, K2, or 1-12).
- For language: ask what language parents or primary caregivers use to communicate with their child at home.
- Keep response under 60 words.
- Return plain text only.
"""

# Preference parsing prompt: single LLM call to extract structured preferences
# from a natural, messy user message.
PREFERENCE_PARSE_PROMPT = f"""{CORE_FACTS}
You are a preference parsing assistant. The user has shared their school preferences
in a natural, conversational way. Extract what you can and respond warmly.

Rules:
- Acknowledge their preferences briefly (1-2 sentences max). Do NOT list them back or re-summarize.
- If they asked a question (like "what is Section 504?"), answer it briefly and clearly.
- If they expressed uncertainty about something, be reassuring and explain it simply.
- NEVER re-list or re-summarize everything the user has already told you. They know what they said.
- End with a SHORT question: either ask ONE focused follow-up, or offer to show schools. Not both.
- Keep response under 80 words total.
- Return plain text only.
"""

# Regex patterns for fast extraction
_GRADE_PATTERN = re.compile(
    r"""(?:grade\s*)?
        (K0|K1|K2|kindergarten|
         \d{1,2}(?:st|nd|rd|th)?
        )
        (?:\s*grade)?""",
    re.IGNORECASE | re.VERBOSE,
)

_ZIP_PATTERN = re.compile(r"\b(02\d{3})\b")

_ADDRESS_PATTERN = re.compile(
    r"\b(\d{1,6}\s+[A-Za-z0-9][A-Za-z0-9 .'\-]*?"
    r"\s+(?:st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|ct|court|pl|place|way|pkwy|parkway))\b",
    re.IGNORECASE,
)

# Map common language mentions to Avela UUIDs
LANGUAGE_MAP = {
    "english": "c188baa2-f2e8-4015-80ee-a42514617585",
    "spanish": "3b523e63-a0a8-4782-9ec8-ba9e5ee16b04",
    "arabic": "10b89d82-0751-47f5-8216-66574f7b0bac",
    "cantonese": "5d9314ac-54cb-4c2f-ba11-70df2cb2a7a9",
    "cape verdean": "254a5e6e-e553-40f3-b9be-c4fd949f2e07",
    "french": "1f13bc17-9f93-4d7d-ae27-90476b01b19e",
    "haitian creole": "562093f6-b3bd-4003-bb85-e51210eb2a35",
    "haitian": "562093f6-b3bd-4003-bb85-e51210eb2a35",
    "creole": "562093f6-b3bd-4003-bb85-e51210eb2a35",
    "italian": "89c38e6d-b9b7-4516-a2c7-661a66452684",
    "korean": "61b2a192-594c-4f4f-b9fb-f5e7d3c2df91",
    "mandarin": "5f5820d8-f3c9-40cf-8e3e-9730961c7bf7",
    "chinese": "5f5820d8-f3c9-40cf-8e3e-9730961c7bf7",
    "portuguese": "28d7754c-e035-4ef0-b942-a501ca6e91ad",
    "russian": "2969bff1-dd46-402c-92a9-cb713deeddd6",
    "somali": "fce808a3-f366-409e-9c2b-863b4f7c3b67",
    "vietnamese": "9f580e8e-ca8e-4142-a3c2-5336fab3d1e1",
}


class Chatbot:
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.hf_model_id = MY_MODEL if MY_MODEL else BASE_MODEL
        self.token = HF_TOKEN
        self.openai_model = OPENAI_MODEL
        self.openai_api_key = OPENAI_API_KEY
        self.reset_state()

    def reset_state(self):
        self.recommendation_pool = []
        self.recommendation_cursor = 0
        self.last_intake = {}
        self.last_eligibility = {}
        self.last_map_data = {}
        self.last_export_path = None
        self.has_active_recommendations = False
        self.intake_memory = {
            "profile": {
                "target_grade": "",
                "zip_or_neighborhood": "",
                "language_needs": "",
                "special_ed_needs": "",
                "transport_or_commute": "",
                "after_school_needs": "",
                "other_preferences": "",
            },
            "missing_fields": [],
            "asked_topics": [],
            "stage": "collecting",
        }
        self._fast_stage = "greeting"  # greeting | awaiting_choice | filtering | done
        self._avela_eligible = []       # raw Avela results before preference filtering
        self._early_preferences = ""    # preferences shared before eligibility was ready

    # ------------------------------------------------------------------
    # Fast-path helpers (deterministic, no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _fast_extract_grade(text: str) -> str:
        """Extract grade from free text. Returns normalized grade or empty string."""
        if not text:
            return ""
        m = _GRADE_PATTERN.search(text)
        if not m:
            return ""
        raw = m.group(1).strip().upper()
        if raw in ("K0", "K1", "K2"):
            return raw
        if "KINDERGARTEN" in raw:
            return "K2"
        num_match = re.match(r"(\d{1,2})", raw)
        if num_match:
            n = int(num_match.group(1))
            if 1 <= n <= 12:
                return str(n)
        return ""

    @staticmethod
    def _fast_extract_zip(text: str) -> str:
        """Extract a Boston ZIP from free text."""
        if not text:
            return ""
        matches = _ZIP_PATTERN.findall(text)
        for m in reversed(matches):
            return m
        return ""

    @staticmethod
    def _fast_extract_address(text: str) -> str:
        """Extract a street address from free text."""
        if not text:
            return ""
        m = _ADDRESS_PATTERN.search(text)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _fast_extract_language(text: str) -> tuple[str, str]:
        """
        Extract home language from free text.
        Returns (language_name, avela_uuid) or ("", "").
        """
        if not text:
            return ("", "")
        t = text.strip().lower()
        for lang, uuid in LANGUAGE_MAP.items():
            if lang in t:
                return (lang.title(), uuid)
        return ("", "")

    @staticmethod
    def _is_show_schools_request(text: str) -> bool:
        """Detect if user wants to see schools now."""
        t = text.strip().lower()
        triggers = [
            "show me", "see the schools", "see my options", "list of schools",
            "show the schools", "show schools", "see the list", "show me the list",
            "just show", "give me the schools", "let me see", "ready to see",
            "show results", "yes please", "yes show", "let's see",
        ]
        return any(trigger in t for trigger in triggers)

    @staticmethod
    def _is_filter_request(text: str) -> bool:
        """Detect if user wants help filtering."""
        t = text.strip().lower()
        triggers = [
            "help me filter", "narrow down", "more questions", "help me find",
            "best fit", "personalize", "preferences",
        ]
        return any(trigger in t for trigger in triggers)

    def _light_llm_response(self, user_input: str, have: dict, missing: list) -> str:
        """
        Single lightweight LLM call for natural acknowledgment + asking for missing info.
        Uses the smaller model for speed and token efficiency.
        """
        payload = (
            f"User said: {user_input}\n\n"
            f"Information collected so far: {json.dumps(have)}\n\n"
            f"Still needed: {', '.join(missing)}"
        )
        try:
            return self._run_agent(LIGHT_RESPONSE_PROMPT, payload, temperature=0.5, max_tokens=100, use_light=False)
        except Exception:
            missing_text = " and ".join(missing)
            return f"Thanks for that! I still need your {missing_text} to look up eligible schools."

    def _parse_preferences_light(self, user_input: str) -> str:
        """
        Single LLM call to parse preferences, answer questions about terms,
        and respond naturally. Uses the smaller model for speed.
        """
        profile = self.intake_memory.get("profile", {})
        payload = (
            f"User said: {user_input}\n\n"
            f"We already know about this family:\n"
            f"- Grade: {profile.get('target_grade', 'unknown')}\n"
            f"- Location: {profile.get('zip_or_neighborhood', 'unknown')}\n"
            f"- Language: {profile.get('language_needs', 'unknown')}\n"
            f"- Preferences shared so far: {json.dumps({k: v for k, v in profile.items() if v and k not in ('target_grade', 'zip_or_neighborhood', 'language_needs', 'language_uuid', 'street_address')})}\n\n"
            f"Number of eligible schools we have: {len(self._avela_eligible)}"
        )
        try:
            return self._run_agent(PREFERENCE_PARSE_PROMPT, payload, temperature=0.5, max_tokens=200, use_light=False)
        except Exception:
            return (
                "Thanks for sharing that! I've noted your preferences. "
                "Would you like to add anything else, or shall I show you the schools that match?"
            )

    def _apply_local_filters(self, schools: list[dict], profile: dict) -> list[dict]:
        """Filter eligible schools based on user preferences from catalog data."""
        filtered = list(schools)

        lang = (profile.get("language_needs") or "").strip().lower()
        if lang and lang not in ("no", "none", "english", "just english", "no preference"):
            filtered = [s for s in filtered if s.get("language_programs", "").strip()] or filtered

        sped = (profile.get("special_ed_needs") or "").strip().lower()
        if sped and sped not in ("no", "none", "no preference"):
            filtered = [s for s in filtered if s.get("special_education_services", "").strip()] or filtered

        after = (profile.get("after_school_needs") or "").strip().lower()
        if after and after not in ("no", "none", "no preference"):
            filtered = [s for s in filtered if s.get("after_school", "").strip()] or filtered

        return filtered

    def _format_results_text(self, schools: list[dict]) -> str:
        """Format school results as plain text with best recommendation + additional options."""
        if not schools:
            return NO_SCHOOLS_TEMPLATE

        lines = []

        # Best recommendation
        best = schools[0]
        lines.append("⭐ **Best Recommendation:**")
        best_line = f"**{best['name']}**"
        if best.get("neighborhood"):
            best_line += f" ({best['neighborhood']})"
        if best.get("grades"):
            best_line += f" — Grades: {best['grades']}"
        lines.append(best_line)
        if best.get("rationale"):
            lines.append(f"_{best['rationale']}_")

        details = []
        if best.get("special_education_services"):
            details.append(f"Special Ed: {best['special_education_services']}")
        if best.get("language_programs"):
            details.append(f"Language: {best['language_programs']}")
        if best.get("after_school"):
            detail_text = best["after_school"][:100]
            details.append(f"After-school: {detail_text}")
        if best.get("hours"):
            details.append(f"Hours: {best['hours']}")
        if details:
            lines.append("  " + " | ".join(details))
        lines.append("")

        # Additional options (up to 3)
        additional = schools[1:4]
        if additional:
            lines.append("**Additional Eligible Schools:**")
            for s in additional:
                entry = f"• **{s['name']}**"
                if s.get("neighborhood"):
                    entry += f" ({s['neighborhood']})"
                if s.get("grades"):
                    entry += f" — Grades: {s['grades']}"
                lines.append(entry)
            lines.append("")

        remaining = len(schools) - 4
        if remaining > 0:
            lines.append(f"_There are {remaining} more eligible schools — just ask to see more, "
                         f"or I can help you **compare**, **filter**, or **export** your options._")
            lines.append("")

        lines.append("---")
        lines.append("**Source:** Eligibility data from [Boston Public Schools](https://www.bostonpublicschools.org) "
                     "via Avela.")
        lines.append("**Disclaimer:** This tool is for informational use only. "
                     "Always confirm eligibility with Boston Public Schools directly.")

        return "\n".join(lines)

    def _fast_path(self, user_input: str, history: list) -> str | None:
        """
        Handle early conversation stages efficiently.
        Uses templates + single light LLM calls instead of the full 3-agent pipeline.
        Returns a response string, or None to fall through to the agent system.
        """
        text = (user_input or "").strip()
        is_first_turn = not history
        profile = self.intake_memory["profile"]

        # --- Stage: greeting ---
        if self._fast_stage == "greeting":
            # Extract whatever info is in the message
            grade = self._fast_extract_grade(text)
            zip_code = self._fast_extract_zip(text)
            address = self._fast_extract_address(text)
            lang_name, lang_uuid = self._fast_extract_language(text)

            if grade:
                profile["target_grade"] = grade
            if zip_code:
                profile["zip_or_neighborhood"] = zip_code
            if address:
                profile["street_address"] = address
            if lang_name:
                profile["language_needs"] = lang_name
                profile["language_uuid"] = lang_uuid

            # Save the full message as early preferences if it has substantial content
            # beyond just grade/ZIP/language (e.g. mentions of sports, commute, special ed)
            if len(text.split()) > 6:
                self._early_preferences = text

            has_grade = bool(profile.get("target_grade"))
            has_location = bool(profile.get("zip_or_neighborhood"))
            has_language = bool(profile.get("language_needs"))

            # Nothing provided on first turn → greeting
            if is_first_turn and not has_grade and not has_location and not has_language and not self._early_preferences:
                return GREETING_TEMPLATE

            # Have all three → call Avela and offer choice
            if has_grade and has_location and has_language:
                return self._call_avela_and_offer_choice()

            # Have grade + location but no language → ask for language with light LLM
            if has_grade and has_location and not has_language:
                have = {k: v for k, v in profile.items() if v}
                return self._light_llm_response(text, have, [
                    "home language (what language do parents or caregivers use to communicate with your child at home?)"
                ])

            # Have some info but not grade + location → ask for missing with light LLM
            have = {k: v for k, v in profile.items() if v}
            missing = []
            if not has_grade:
                missing.append("grade applying for (K0, K1, K2, or 1-12)")
            if not has_location:
                missing.append("home address or Boston ZIP code")
            if not has_language:
                missing.append("home language (language parents use to communicate with child at home)")

            return self._light_llm_response(text, have, missing)

        # --- Stage: awaiting_language ---
        if self._fast_stage == "awaiting_language":
            lang_name, lang_uuid = self._fast_extract_language(text)
            if lang_name:
                profile["language_needs"] = lang_name
                profile["language_uuid"] = lang_uuid
            elif text.strip():
                # If they typed something but we can't match a language,
                # treat as English (they might have said "just English" or "we speak English at home")
                eng_keywords = ["english", "no", "none", "just", "same", "normal"]
                if any(k in text.lower() for k in eng_keywords):
                    profile["language_needs"] = "English"
                    profile["language_uuid"] = LANGUAGE_MAP["english"]
                else:
                    # Still can't determine — default to English
                    profile["language_needs"] = "English"
                    profile["language_uuid"] = LANGUAGE_MAP["english"]

            return self._call_avela_and_offer_choice()

        # --- Stage: awaiting_choice ---
        if self._fast_stage == "awaiting_choice":
            if self._is_show_schools_request(text):
                self.recommendation_pool = self._avela_eligible
                self.has_active_recommendations = True
                self.recommendation_cursor = min(4, len(self.recommendation_pool))
                self._fast_stage = "done"
                return self._format_results_text(self.recommendation_pool)

            # User is sharing preferences or asking questions — use light LLM
            if text:
                self._fast_stage = "filtering"
                self.intake_memory["stage"] = "filtering"
                return self._parse_preferences_light(text)

            return None

        # --- Stage: filtering ---
        if self._fast_stage == "filtering":
            if self._is_show_schools_request(text):
                filtered = self._apply_local_filters(self._avela_eligible, self.intake_memory["profile"])
                self.recommendation_pool = filtered
                self.has_active_recommendations = True
                self.recommendation_cursor = min(4, len(self.recommendation_pool))
                self._fast_stage = "done"
                return self._format_results_text(self.recommendation_pool)

            # More preferences or questions — keep using light LLM
            if text:
                return self._parse_preferences_light(text)

            return None

        # --- Stage: done ---
        # Handle "show more" deterministically instead of falling through to agents
        if self._fast_stage == "done" and self.recommendation_pool:
            t = text.strip().lower()
            if "show more" in t or "more schools" in t or "see more" in t or "next" in t:
                start = self.recommendation_cursor
                end = min(start + 4, len(self.recommendation_pool))
                if start >= len(self.recommendation_pool):
                    return "You've seen all the eligible schools I found. Would you like to compare any of them, or is there anything else I can help with?"
                next_batch = self.recommendation_pool[start:end]
                self.recommendation_cursor = end
                remaining = len(self.recommendation_pool) - end

                lines = [f"**More Eligible Schools ({start + 1}–{end} of {len(self.recommendation_pool)}):**\n"]
                for s in next_batch:
                    entry = f"• **{s['name']}**"
                    if s.get("neighborhood"):
                        entry += f" ({s['neighborhood']})"
                    if s.get("grades"):
                        entry += f" — Grades: {s['grades']}"
                    lines.append(entry)
                    details = []
                    if s.get("special_education_services"):
                        details.append(f"Special Ed: {s['special_education_services']}")
                    if s.get("language_programs"):
                        details.append(f"Language: {s['language_programs']}")
                    if s.get("after_school"):
                        details.append(f"After-school: {s['after_school'][:80]}")
                    if details:
                        lines.append("  " + " | ".join(details))

                if remaining > 0:
                    lines.append(f"\n_{remaining} more available. Say **\"show more\"** to continue._")
                else:
                    lines.append("\n_That's all the eligible schools I found. Would you like to compare any of them?_")

                return "\n".join(lines)

        return None

    def _call_avela_and_offer_choice(self) -> str:
        """Call Avela API with collected profile data and present the choice template."""
        profile = self.intake_memory["profile"]
        try:
            self._avela_eligible = avela_get_schools(
                target_grade=profile.get("target_grade", ""),
                zip_or_neighborhood=profile.get("zip_or_neighborhood", ""),
                street_address=profile.get("street_address", ""),
                language_uuid=profile.get("language_uuid", ""),
            )
        except Exception:
            self._avela_eligible = []

        if self._avela_eligible:
            self.intake_memory["stage"] = "ready_for_recommendations"

            # If the user shared preferences earlier (before we had grade/ZIP),
            # skip the choice template and process those preferences now
            if self._early_preferences:
                self._fast_stage = "filtering"
                early = self._early_preferences
                self._early_preferences = ""  # clear so we don't reprocess
                return self._parse_preferences_light(early)

            self._fast_stage = "awaiting_choice"
            return CHOICE_TEMPLATE
        else:
            self._fast_stage = "done"
            return NO_SCHOOLS_TEMPLATE

    def _get_client(self):
        if self.provider == "openai":
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is missing. Set OPENAI_API_KEY in your environment "
                    "or switch LLM_PROVIDER=huggingface."
                )
            return OpenAI(api_key=self.openai_api_key)
        return InferenceClient(model=self.hf_model_id, token=self.token)

    def _build_history_text(self, history=None):
        lines = []
        if history:
            for user_msg, bot_msg in history:
                lines.append(f"User: {user_msg}")
                lines.append(f"Assistant: {bot_msg}")
        return "\n".join(lines).strip()

    def _run_agent(self, system_prompt, payload, temperature=0.2, max_tokens=450, use_light=False):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload},
        ]
        client = self._get_client()
        if self.provider == "openai":
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return self._coerce_to_text(response.choices[0].message.content).strip()

        # For HuggingFace: use light model for simple tasks, heavy model for complex reasoning
        model_id = LIGHT_MODEL if use_light else self.hf_model_id
        response = client.chat_completion(
            messages=messages,
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self._coerce_to_text(response.choices[0].message.content).strip()

    @staticmethod
    def _coerce_to_text(content):
        """
        Normalize model content into plain text.
        Some providers return structured content blocks instead of a raw string.
        """
        if isinstance(content, str):
            return Chatbot._unwrap_text_wrapper_string(content)
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text
            return json.dumps(content, ensure_ascii=True)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "\n".join(parts)
            return json.dumps(content, ensure_ascii=True)
        return str(content)

    @staticmethod
    def _unwrap_text_wrapper_string(text):
        candidate = (text or "").strip()
        if not candidate:
            return candidate

        # If the model emits a dict-like wrapper as a string, extract only its text payload.
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                parsed_json = json.loads(candidate)
                if isinstance(parsed_json, dict) and isinstance(parsed_json.get("text"), str):
                    return parsed_json["text"].strip()
            except json.JSONDecodeError:
                pass

            try:
                parsed_literal = ast.literal_eval(candidate)
                if isinstance(parsed_literal, dict) and isinstance(parsed_literal.get("text"), str):
                    return parsed_literal["text"].strip()
            except (ValueError, SyntaxError):
                pass

        return candidate

    @staticmethod
    def _extract_json(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return {}
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}

    def _route_turn(self, user_input, history_text, history):
        payload = (
            f"Conversation history:\n{history_text or '(none)'}\n\n"
            f"Latest user request:\n{user_input}\n\n"
            f"Current Memory JSON:\n{json.dumps(self.intake_memory, ensure_ascii=True)}\n\n"
            f"Recommendation state:\n"
            f"- has_active_recommendations: {self.has_active_recommendations}\n"
            f"- recommendation_count: {len(self.recommendation_pool)}\n"
            f"- recommendation_cursor: {self.recommendation_cursor}\n"
            f"- is_first_turn: {not bool(history)}"
        )
        route_data = self._extract_json(self._run_agent(ORCHESTRATOR_PROMPT, payload, temperature=0.1, max_tokens=180))
        if not isinstance(route_data, dict):
            route_data = {}
        route = str(route_data.get("route", "")).strip()
        should_run_intake = route_data.get("should_run_intake", True)
        if route not in {
            "welcome",
            "intake",
            "school_specialist",
            "recommendation",
            "recommendation_followup",
            "export_csv",
            "general_info",
        }:
            route = "intake"
        if not isinstance(should_run_intake, bool):
            should_run_intake = True
        return {"route": route, "should_run_intake": should_run_intake}

    def _run_intake_turn(self, user_input, history_text):
        intake_payload = (
            f"Conversation history:\n{history_text or '(none)'}\n\n"
            f"Latest user request:\n{user_input}\n\n"
            f"Current Memory JSON:\n{json.dumps(self.intake_memory, ensure_ascii=True)}"
        )
        intake = self._extract_json(self._run_agent(INTAKE_PROMPT, intake_payload, temperature=0.15, max_tokens=380))
        if not isinstance(intake, dict):
            intake = {"stage": "collecting", "profile": {}, "missing_fields": [], "next_topic": "other_preferences", "next_question": ""}
        self.last_intake = intake
        self._merge_intake_memory(intake)
        return intake

    def _run_recommendation_specialists(self, user_input, history_text):
        specialist_payload = (
            f"Conversation history:\n{history_text or '(none)'}\n\n"
            f"Latest user request:\n{user_input}\n\n"
            f"Intake JSON:\n{json.dumps(self.last_intake, ensure_ascii=True)}\n\n"
            f"Current Memory JSON:\n{json.dumps(self.intake_memory, ensure_ascii=True)}"
        )

        # --- Try Avela real eligibility first ---
        avela_schools = []
        profile = self.intake_memory.get("profile", {})
        target_grade = profile.get("target_grade", "")
        zip_or_neighborhood = profile.get("zip_or_neighborhood", "")
        if target_grade and zip_or_neighborhood:
            try:
                avela_schools = avela_get_schools(
                    target_grade=target_grade,
                    zip_or_neighborhood=zip_or_neighborhood,
                )
            except Exception:
                avela_schools = []

        if avela_schools:
            # Use real Avela data for recommendations
            self.recommendation_pool = avela_schools
            # Still run eligibility and map agents for context
            with ThreadPoolExecutor(max_workers=2) as executor:
                eligibility_future = executor.submit(self._run_agent, ELIGIBILITY_PROMPT, specialist_payload, 0.2, 380)
                map_future = executor.submit(self._run_agent, MAP_PROMPT, specialist_payload, 0.2, 380)
                eligibility = self._extract_json(eligibility_future.result())
                map_data = self._extract_json(map_future.result())
            self.last_eligibility = eligibility if isinstance(eligibility, dict) else {}
            self.last_map_data = map_data if isinstance(map_data, dict) else {}
        else:
            # Fallback: use LLM Match Agent (original behavior)
            with ThreadPoolExecutor(max_workers=3) as executor:
                eligibility_future = executor.submit(self._run_agent, ELIGIBILITY_PROMPT, specialist_payload, 0.2, 380)
                match_future = executor.submit(self._run_agent, MATCH_PROMPT, specialist_payload, 0.3, 520)
                map_future = executor.submit(self._run_agent, MAP_PROMPT, specialist_payload, 0.2, 380)
                eligibility = self._extract_json(eligibility_future.result())
                match = self._extract_json(match_future.result())
                map_data = self._extract_json(map_future.result())

            self.last_eligibility = eligibility if isinstance(eligibility, dict) else {}
            self.last_map_data = map_data if isinstance(map_data, dict) else {}
            self.recommendation_pool = self._build_recommendation_pool(match)
            if not self.recommendation_pool:
                recovery = self._extract_json(self._run_agent(MATCH_RECOVERY_PROMPT, specialist_payload, temperature=0.35, max_tokens=520))
                self.recommendation_pool = self._build_recommendation_pool(recovery)

        self.has_active_recommendations = bool(self.recommendation_pool)

    def _render_recommendations_with_agent(self, user_input, history_text):
        top_schools = self.recommendation_pool[:4]
        self.recommendation_cursor = min(4, len(self.recommendation_pool))
        payload = (
            f"Conversation history:\n{history_text or '(none)'}\n\n"
            f"Latest user request:\n{user_input}\n\n"
            f"Eligibility JSON:\n{json.dumps(self.last_eligibility, ensure_ascii=True)}\n\n"
            f"Top schools JSON:\n{json.dumps(top_schools, ensure_ascii=True)}\n\n"
            f"Map JSON:\n{json.dumps(self.last_map_data, ensure_ascii=True)}"
        )
        return self._run_agent(RECOMMENDATION_RESPONSE_PROMPT, payload, temperature=0.45, max_tokens=360)

    def _run_followup_action(self, user_input, history_text):
        payload = (
            f"Conversation history:\n{history_text or '(none)'}\n\n"
            f"Latest user request:\n{user_input}\n\n"
            f"Current recommendation count: {len(self.recommendation_pool)}\n"
        )
        action_data = self._extract_json(self._run_agent(FOLLOWUP_ACTION_PROMPT, payload, temperature=0.1, max_tokens=140))
        if not isinstance(action_data, dict):
            action_data = {}
        action = str(action_data.get("action", "summary")).strip()
        if action not in {"show_more", "compare", "map", "summary"}:
            action = "summary"
        indexes = action_data.get("indexes", [])
        if not isinstance(indexes, list):
            indexes = []
        parsed_indexes = []
        for idx in indexes:
            try:
                num = int(idx)
            except (TypeError, ValueError):
                continue
            if 1 <= num <= len(self.recommendation_pool) and num not in parsed_indexes:
                parsed_indexes.append(num)
        action_data["action"] = action
        action_data["indexes"] = parsed_indexes[:3]
        return action_data

    def _build_followup_payload(self, action_data):
        action = action_data.get("action", "summary")
        if action == "show_more":
            start = self.recommendation_cursor
            end = min(start + 3, len(self.recommendation_pool))
            schools = self.recommendation_pool[start:end]
            self.recommendation_cursor = end
            return {
                "action": "show_more",
                "schools": schools,
                "has_more": end < len(self.recommendation_pool),
            }
        if action == "compare":
            indexes = action_data.get("indexes", [])
            if not indexes:
                indexes = [1, 2, 3][: min(3, len(self.recommendation_pool))]
            schools = []
            for i in indexes:
                if 1 <= i <= len(self.recommendation_pool):
                    school = dict(self.recommendation_pool[i - 1])
                    school["rank"] = i
                    schools.append(school)
            return {"action": "compare", "schools": schools}
        if action == "map":
            return {
                "action": "map",
                "schools": self.recommendation_pool[:6],
                "map_data": self.last_map_data,
            }
        return {"action": "summary", "schools": self.recommendation_pool[:3]}

    @staticmethod
    def _normalize_school(entry):
        if not isinstance(entry, dict):
            return None
        name = str(entry.get("name", "")).strip()
        if not name:
            return None
        return {
            "name": name,
            "neighborhood": str(entry.get("neighborhood", "")).strip(),
            "grades": str(entry.get("grades", "")).strip(),
            "language_programs": str(entry.get("language_programs", "")).strip(),
            "special_education_services": str(entry.get("special_education_services", "")).strip(),
            "after_school": str(entry.get("after_school", "")).strip(),
            "hours": str(entry.get("hours", "")).strip(),
            "rationale": str(entry.get("rationale", "")).strip(),
        }

    def _build_recommendation_pool(self, match_data):
        schools = []
        seen = set()
        for raw in match_data.get("candidate_schools", []) if isinstance(match_data, dict) else []:
            item = self._normalize_school(raw)
            if not item:
                continue
            key = item["name"].lower()
            if key in seen:
                continue
            seen.add(key)
            schools.append(item)

        best_name = str(match_data.get("best_school_name", "")).strip().lower() if isinstance(match_data, dict) else ""
        if best_name:
            for i, school in enumerate(schools):
                if school["name"].lower() == best_name:
                    schools.insert(0, schools.pop(i))
                    break
        return schools

    def _merge_intake_memory(self, intake):
        if not isinstance(intake, dict):
            return

        profile = intake.get("profile", {})
        if isinstance(profile, dict):
            for key in self.intake_memory["profile"]:
                val = str(profile.get(key, "")).strip()
                if val:
                    self.intake_memory["profile"][key] = val

        missing = intake.get("missing_fields", [])
        if isinstance(missing, list):
            self.intake_memory["missing_fields"] = [str(x).strip() for x in missing if str(x).strip()]

        stage = str(intake.get("stage", "")).strip()
        if stage:
            self.intake_memory["stage"] = stage

        next_topic = str(intake.get("next_topic", "")).strip()
        if next_topic and next_topic != "none" and next_topic not in self.intake_memory["asked_topics"]:
            self.intake_memory["asked_topics"].append(next_topic)

    @staticmethod
    def _school_map_link(name, neighborhood="Boston"):
        return f"https://www.google.com/maps/search/?api=1&query={quote_plus((name + ' ' + neighborhood).strip())}"

    def _export_recommendations_csv(self):
        filename = f"bps_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = f"{tempfile.gettempdir()}/{filename}"
        fields = [
            "rank",
            "name",
            "neighborhood",
            "grades",
            "language_programs",
            "special_education_services",
            "after_school",
            "hours",
            "rationale",
            "map_link",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, school in enumerate(self.recommendation_pool, start=1):
                writer.writerow(
                    {
                        "rank": i,
                        "name": school["name"],
                        "neighborhood": school.get("neighborhood", ""),
                        "grades": school.get("grades", ""),
                        "language_programs": school.get("language_programs", ""),
                        "special_education_services": school.get("special_education_services", ""),
                        "after_school": school.get("after_school", ""),
                        "hours": school.get("hours", ""),
                        "rationale": school.get("rationale", ""),
                        "map_link": self._school_map_link(school["name"], school.get("neighborhood") or "Boston"),
                    }
                )

        self.last_export_path = output_path
        return f"Export complete. CSV saved at: {output_path}"

    def consume_last_export_path(self):
        path = self.last_export_path
        self.last_export_path = None
        return path

    def get_response(self, user_input, history=None):
        # --- Try fast path first (deterministic, no LLM) ---
        fast_response = self._fast_path(user_input, history)
        if fast_response is not None:
            return fast_response

        # --- Fall through to agent system for complex reasoning ---
        history_text = self._build_history_text(history)
        route_data = self._route_turn(user_input, history_text, history)
        route = route_data["route"]
        should_run_intake = route_data["should_run_intake"]

        if route == "welcome":
            # Fast path should have caught this, but just in case
            return GREETING_TEMPLATE

        if should_run_intake:
            self._run_intake_turn(user_input, history_text)

        stage = str(self.intake_memory.get("stage", "collecting")).strip()

        if route in {"intake", "general_info"} or stage in {"out_of_scope", "general_info"}:
            response_payload = (
                f"Conversation history:\n{history_text or '(none)'}\n\n"
                f"Latest user request:\n{user_input}\n\n"
                f"Intake JSON:\n{json.dumps(self.last_intake, ensure_ascii=True)}\n\n"
                f"Current Memory JSON:\n{json.dumps(self.intake_memory, ensure_ascii=True)}"
            )
            return self._run_agent(INTAKE_RESPONSE_PROMPT, response_payload, temperature=0.5, max_tokens=220)

        if route == "school_specialist":
            specialist_payload = (
                f"Conversation history:\n{history_text or '(none)'}\n\n"
                f"Latest user request:\n{user_input}\n\n"
                f"Intake JSON:\n{json.dumps(self.last_intake, ensure_ascii=True)}\n\n"
                f"Current Memory JSON:\n{json.dumps(self.intake_memory, ensure_ascii=True)}"
            )
            return self._run_agent(SCHOOL_SPECIALIST_PROMPT, specialist_payload, temperature=0.45, max_tokens=180)

        if route == "export_csv":
            if not self.recommendation_pool:
                return "I can export once I have a recommendation list. Share grade, Boston ZIP/neighborhood, and top preference first."
            export_message = self._export_recommendations_csv()
            export_payload = (
                f"Latest user request:\n{user_input}\n\n"
                f"System export result:\n{export_message}\n\n"
                f"Export path:\n{self.last_export_path or ''}"
            )
            return self._run_agent(EXPORT_RESPONSE_PROMPT, export_payload, temperature=0.25, max_tokens=120)

        if route == "recommendation_followup" and self.recommendation_pool:
            action_data = self._run_followup_action(user_input, history_text)
            followup_payload = self._build_followup_payload(action_data)
            response_payload = (
                f"Latest user request:\n{user_input}\n\n"
                f"Follow-up action JSON:\n{json.dumps(followup_payload, ensure_ascii=True)}"
            )
            return self._run_agent(FOLLOWUP_RESPONSE_PROMPT, response_payload, temperature=0.4, max_tokens=280)

        if route == "recommendation":
            self._run_recommendation_specialists(user_input, history_text)
            if not self.recommendation_pool:
                recovery_payload = (
                    f"Conversation history:\n{history_text or '(none)'}\n\n"
                    f"Latest user request:\n{user_input}\n\n"
                    f"Intake JSON:\n{json.dumps(self.last_intake, ensure_ascii=True)}\n\n"
                    f"Current Memory JSON:\n{json.dumps(self.intake_memory, ensure_ascii=True)}"
                )
                return self._run_agent(SCHOOL_SPECIALIST_PROMPT, recovery_payload, temperature=0.4, max_tokens=180)
            return self._render_recommendations_with_agent(user_input, history_text)

        fallback_payload = (
            f"Conversation history:\n{history_text or '(none)'}\n\n"
            f"Latest user request:\n{user_input}\n\n"
            f"Intake JSON:\n{json.dumps(self.last_intake, ensure_ascii=True)}\n\n"
            f"Current Memory JSON:\n{json.dumps(self.intake_memory, ensure_ascii=True)}"
        )
        return self._run_agent(INTAKE_RESPONSE_PROMPT, fallback_payload, temperature=0.45, max_tokens=220)