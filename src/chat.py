import csv
import json
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
)

CORE_FACTS = """You are a helpful assistant for Boston Public Schools (BPS) enrollment guidance.
You help parents and legal guardians understand enrollment options, eligibility basics, and tradeoffs.
You must be warm, concise, and transparent about uncertainty.
Always remind families to confirm final eligibility with bostonpublicschools.org.
"""

GREETING_PROMPT = f"""{CORE_FACTS}
You are the Welcome Agent.
Write a warm first message in plain text for a family starting enrollment chat.
Ask for exactly two required items to begin: child's target grade and Boston ZIP/neighborhood.
Keep to 2-4 short sentences.
"""

INTAKE_PROMPT = f"""{CORE_FACTS}
You are the Intake Agent.

Return ONLY valid JSON with this schema:
{{
  "stage": "collecting|ready_for_recommendations|out_of_scope|general_info",
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
- Prefer stage=collecting until you have enough to make recommendations.
- Set stage=ready_for_recommendations once target_grade + zip_or_neighborhood are known and there are no critical blockers.
- Do not over-collect optional preferences before moving to recommendations.
- If user is outside MA/Boston scope, use stage=out_of_scope and provide official state/district resource links.
- Keep next_question natural and specific to what is still missing.
- Set next_topic to one of: target_grade, zip_or_neighborhood, language_needs, special_ed_needs, transport_or_commute, after_school_needs, other_preferences, none.
- Use Current Memory JSON in the user payload.
- Do NOT re-ask a topic listed in memory.asked_topics unless user asked to revisit it.
"""

INTAKE_RESPONSE_PROMPT = f"""{CORE_FACTS}
You are the Intake Response Agent.
Given intake JSON and user message, write a warm, dynamic response.

Rules:
- If stage=collecting: ask only one focused next question.
- If stage=out_of_scope: gently explain scope and include suggested links.
- Avoid repeating the same template; vary phrasing naturally.
- Avoid repeating prior details. Use at most one short acknowledgment phrase.
- Do not restate all known profile fields unless the user explicitly asks for a summary.
- Return plain text only. Do not return JSON, dicts, or code blocks.
- Keep response under 120 words.
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

    def _run_agent(self, system_prompt, payload, temperature=0.2, max_tokens=450):
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

        response = client.chat_completion(
            messages=messages,
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

    @staticmethod
    def _is_greeting(text):
        lowered = (text or "").strip().lower()
        return lowered in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}

    @staticmethod
    def _is_more_request(text):
        lowered = (text or "").lower()
        return any(token in lowered for token in ["show more", "more schools", "next 3", "next three", "more options"])

    @staticmethod
    def _is_compare_request(text):
        return "compare" in (text or "").lower()

    @staticmethod
    def _is_map_request(text):
        lowered = (text or "").lower()
        return any(token in lowered for token in ["map", "directions", "route"])

    @staticmethod
    def _is_export_request(text):
        lowered = (text or "").lower()
        return "export" in lowered or "csv" in lowered or "download" in lowered

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

    def _has_minimum_intake(self):
        profile = self.intake_memory.get("profile", {})
        return bool(str(profile.get("target_grade", "")).strip()) and bool(str(profile.get("zip_or_neighborhood", "")).strip())

    @staticmethod
    def _school_map_link(name, neighborhood="Boston"):
        return f"https://www.google.com/maps/search/?api=1&query={quote_plus((name + ' ' + neighborhood).strip())}"

    def _format_school_card(self, idx, school, best=False):
        title = f"{idx}. {school['name']}"
        if best:
            title += " (Best recommendation)"
        lines = [f"- {title}"]
        for label, key in [
            ("Neighborhood", "neighborhood"),
            ("Grades", "grades"),
            ("Language programs", "language_programs"),
            ("Special education", "special_education_services"),
            ("After-school", "after_school"),
            ("Hours", "hours"),
            ("Why it may fit", "rationale"),
        ]:
            if school.get(key):
                lines.append(f"- {label}: {school[key]}")
        lines.append(f"- Map: {self._school_map_link(school['name'], school.get('neighborhood') or 'Boston')}")
        return "\n".join(lines)

    def _format_initial_recommendations(self):
        best = self.recommendation_pool[0]
        others = self.recommendation_pool[1:4]
        self.recommendation_cursor = 4

        parts = [
            "Here is one best recommendation plus three additional eligible schools (unordered).",
            "",
            "Best recommendation:",
            self._format_school_card(1, best, best=True),
        ]
        if others:
            parts += ["", "Additional eligible schools (unordered):"]
            for i, school in enumerate(others, start=2):
                parts.append(self._format_school_card(i, school))

        parts += [
            "",
            "Ask me: `show more`, `compare 1 and 3`, `map options`, or `export csv`.",
            "Always confirm final eligibility with BPS: https://www.bostonpublicschools.org/",
        ]
        return "\n".join(parts)

    def _format_next_batch(self):
        start = self.recommendation_cursor
        end = min(start + 3, len(self.recommendation_pool))
        if start >= len(self.recommendation_pool):
            return "You have reached the end of the current list."

        lines = [f"Next {end - start} eligible schools (unordered):"]
        for i in range(start, end):
            lines.append(self._format_school_card(i + 1, self.recommendation_pool[i]))
        self.recommendation_cursor = end
        if end < len(self.recommendation_pool):
            lines.append("\nSay `show more` to see the next 3.")
        return "\n".join(lines)

    @staticmethod
    def _parse_compare_indexes(text, max_index):
        values = []
        for token in text.split():
            if token.isdigit():
                num = int(token)
                if 1 <= num <= max_index and num not in values:
                    values.append(num)
        return values[:3]

    def _format_comparison(self, indexes):
        if not indexes:
            indexes = [1, 2, 3][: min(3, len(self.recommendation_pool))]

        lines = [
            "| # | School | Grades | Language | Special Ed | After-School | Neighborhood |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for idx in indexes:
            s = self.recommendation_pool[idx - 1]
            lines.append(
                f"| {idx} | {s['name']} | {s.get('grades') or '-'} | {s.get('language_programs') or '-'} | "
                f"{s.get('special_education_services') or '-'} | {s.get('after_school') or '-'} | {s.get('neighborhood') or '-'} |"
            )
        return "Here is a side-by-side comparison:\n\n" + "\n".join(lines)

    def _format_map_options(self):
        lines = ["Map and route links for your current options:"]
        for i, school in enumerate(self.recommendation_pool[:6], start=1):
            lines.append(f"- {i}. {school['name']}: {self._school_map_link(school['name'], school.get('neighborhood') or 'Boston')}")

        steps = self.last_map_data.get("map_export_steps", [])
        if isinstance(steps, list) and steps:
            lines += ["", "Suggested map export steps:"] + [f"- {step}" for step in steps[:4]]
        return "\n".join(lines)

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
        history_text = self._build_history_text(history)

        if not history and self._is_greeting(user_input):
            return self._run_agent(
                GREETING_PROMPT,
                "Start the conversation with a warm, concise welcome.",
                temperature=0.6,
                max_tokens=120,
            )

        if self.has_active_recommendations:
            if self._is_export_request(user_input):
                return self._export_recommendations_csv()
            if self._is_compare_request(user_input):
                return self._format_comparison(self._parse_compare_indexes(user_input, len(self.recommendation_pool)))
            if self._is_map_request(user_input):
                return self._format_map_options()
            if self._is_more_request(user_input):
                return self._format_next_batch()

        intake_payload = (
            f"Conversation history:\n{history_text or '(none)'}\n\n"
            f"Latest user request:\n{user_input}\n\n"
            f"Current Memory JSON:\n{json.dumps(self.intake_memory, ensure_ascii=True)}"
        )
        intake = self._extract_json(self._run_agent(INTAKE_PROMPT, intake_payload, temperature=0.15, max_tokens=380))
        if not isinstance(intake, dict):
            intake = {"stage": "collecting", "profile": {}, "missing_fields": [], "next_topic": "other_preferences", "next_question": ""}

        prior_topics = set(self.intake_memory.get("asked_topics", []))
        next_topic = str(intake.get("next_topic", "")).strip()
        if next_topic and next_topic in prior_topics and self._has_minimum_intake():
            intake["stage"] = "ready_for_recommendations"

        self.last_intake = intake
        self._merge_intake_memory(intake)
        stage = self.intake_memory.get("stage", intake.get("stage", "collecting"))
        if self._has_minimum_intake() and stage == "collecting":
            stage = "ready_for_recommendations"

        if stage in {"collecting", "out_of_scope", "general_info"}:
            response_payload = (
                f"Conversation history:\n{history_text or '(none)'}\n\n"
                f"Latest user request:\n{user_input}\n\n"
                f"Intake JSON:\n{json.dumps(self.last_intake, ensure_ascii=True)}\n\n"
                f"Current Memory JSON:\n{json.dumps(self.intake_memory, ensure_ascii=True)}"
            )
            return self._run_agent(INTAKE_RESPONSE_PROMPT, response_payload, temperature=0.5, max_tokens=220)

        specialist_payload = (
            f"Conversation history:\n{history_text or '(none)'}\n\n"
            f"Latest user request:\n{user_input}\n\n"
            f"Intake JSON:\n{json.dumps(self.last_intake, ensure_ascii=True)}\n\n"
            f"Current Memory JSON:\n{json.dumps(self.intake_memory, ensure_ascii=True)}"
        )

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

        if not self.recommendation_pool:
            # Let intake-response agent ask for next best input instead of static fallback text.
            recovery_intake = {
                "stage": "collecting",
                "missing_fields": ["preference details"],
                "next_question": "Which should I prioritize first: shorter commute, language programs, special education supports, or after-school options?",
                "profile": self.last_intake.get("profile", {}) if isinstance(self.last_intake, dict) else {},
            }
            response_payload = (
                f"Conversation history:\n{history_text or '(none)'}\n\n"
                f"Latest user request:\n{user_input}\n\n"
                f"Intake JSON:\n{json.dumps(recovery_intake, ensure_ascii=True)}"
            )
            return self._run_agent(INTAKE_RESPONSE_PROMPT, response_payload, temperature=0.5, max_tokens=220)

        intro = self.last_eligibility.get("eligibility_summary", "") if isinstance(self.last_eligibility, dict) else ""
        if intro:
            return f"{intro}\n\n{self._format_initial_recommendations()}"
        return self._format_initial_recommendations()
