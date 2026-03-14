"""
BPS Enrollment Chatbot — Entry point.

Runnable locally: python app.py
Also used as the HuggingFace Space app entry (no other entry points per PRD).

Optional env: PORT — port to use (default 7860). Use if 7860 is already in use.
"""

from __future__ import annotations

import os
import re

# Load .env so HF_TOKEN, PORT, etc. are available when running locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr

from services import eligibility, intake, llm

# -----------------------------------------------------------------------------
# Session state: in-memory only, keyed by UI session ID (no server-side PII persistence).
# Data is lost on server restart; we never write to disk or a database.
# -----------------------------------------------------------------------------
_session_store: dict[str, dict] = {}


def _session_id(request: gr.Request | None) -> str:
    """Return a stable session ID for the current request (for session store lookup)."""
    if request is not None and getattr(request, "session_hash", None):
        return request.session_hash
    return str(id(request) if request is not None else os.urandom(8).hex())


def _history_to_messages(history: list) -> list:
    """Convert Gradio chat history [(user, assistant), ...] to list of {role, content} for LLM."""
    messages = []
    for user_msg, asst_msg in history or []:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if asst_msg:
            messages.append({"role": "assistant", "content": asst_msg})
    return messages


def _looks_like_grade_or_question(text: str) -> bool:
    """
    True if the message might be about grade (e.g. "K2", "grade 5") or a clear BPS/grade-related question.
    False for gibberish or clearly unrelated input — we'll reply with "I didn't understand" instead of the LLM.
    """
    if not text or not text.strip():
        return False
    t = text.strip().lower()
    # Could be a grade: digits, or K1/K2, or words like "grade", "kindergarten"
    if any(c.isdigit() for c in t):
        return True
    grade_related = ("k1", "k2", "grade", "kindergarten", "child", "kid", "school", "bps", "boston", "enroll", "attend")
    return any(w in t for w in grade_related)


def _looks_like_address_or_question(text: str) -> bool:
    """
    True if the message might be an address/ZIP or a clear question about it.
    False for gibberish — we'll reply with "I didn't understand" and re-ask for address.
    """
    if not text or not text.strip():
        return False
    t = text.strip().lower()
    # Likely address: contains digits (ZIP or street number) or address-related words
    if any(c.isdigit() for c in t):
        return True
    address_related = ("zip", "address", "street", "boston", "live", "location", "021", "022")
    return any(w in t for w in address_related)


def _contains_zip_5(text: str) -> bool:
    """Return True if text contains a 5-digit ZIP (or ZIP+4)."""
    if not text:
        return False
    return bool(re.search(r"\b\d{5}(?:-\d{4})?\b", text))


def _looks_like_street_address(text: str) -> bool:
    """Heuristic for street-style address text."""
    if not text or not text.strip():
        return False
    t = text.lower()
    suffixes = (
        " st", " street", " ave", " avenue", " rd", " road", " blvd", " boulevard",
        " ln", " lane", " dr", " drive", " ct", " court", " pl", " place", " pkwy", " parkway",
    )
    has_suffix = any(s in t for s in suffixes)
    has_number = bool(re.search(r"\b\d{1,6}\b", t))
    return has_suffix and has_number


def _is_greeting(text: str) -> bool:
    """Detect simple greetings so we can welcome users instead of saying we did not understand."""
    if not text or not text.strip():
        return False
    t = text.strip().lower()
    return bool(re.match(r"^(hi|hello|hey|good (morning|afternoon|evening)|yo)\b", t))


def _is_reask_schools_request(text: str) -> bool:
    """Detect requests to show schools again from users who already completed intake."""
    if not text or not text.strip():
        return False
    t = text.lower()
    triggers = ("show", "list", "again", "schools", "eligible", "results")
    return ("school" in t or "eligible" in t) and any(k in t for k in triggers)


def _is_zip_confirmation_question(text: str) -> bool:
    """Detect questions asking whether a ZIP/address is in Boston."""
    if not text or not text.strip():
        return False
    t = text.lower()
    return ("zip" in t or "021" in t or "022" in t or "boston" in t) and "?" in t


# Citation and disclaimer shown with every eligibility result (plan: school-output).
BPS_SOURCE_CITATION = (
    "**Source:** Eligibility and school information are based on BPS eligibility rules "
    "via [Boston Public Schools](https://www.bostonpublicschools.org)."
)
BPS_DISCLAIMER = (
    "**Disclaimer:** This tool is for informational use only. "
    "Always confirm eligibility and enrollment details with Boston Public Schools directly."
)


def _format_schools_list(schools: list) -> str:
    """Format eligible schools for display. All data from eligibility only; no LLM fabrication."""
    if not schools:
        return (
            "No eligible schools were found for that address and grade. "
            "Please confirm your address with Boston Public Schools.\n\n"
            f"{BPS_SOURCE_CITATION}\n\n{BPS_DISCLAIMER}"
        )
    lines = []
    for i, s in enumerate(schools[:20], 1):  # cap at 20 for MVP
        name = getattr(s, "school_name", None) or (s.get("school_name") if isinstance(s, dict) else "—")
        level = getattr(s, "level", None) or (s.get("level") if isinstance(s, dict) else "")
        addr = getattr(s, "address", None) or (s.get("address") if isinstance(s, dict) else "")
        parts = [f"{i}. **{name}**"]
        if level:
            parts.append(f" ({level})")
        if addr:
            parts.append(f" — {addr}")
        lines.append("".join(parts))
    citation_block = f"\n\n{BPS_SOURCE_CITATION}\n\n{BPS_DISCLAIMER}"
    return "\n".join(lines) + citation_block


def chat(message: str, history: list, request: gr.Request | None = None) -> tuple[list, str]:
    """
    Chat handler: delegates to intake orchestrator (grade → address), then LLM when
    needed. Eligibility and school list come only from the eligibility module.
    Session state is kept in-memory keyed by UI session ID (no PII persistence).
    Returns (updated_history, reply).
    """
    history = history or []
    sid = _session_id(request)
    state = _session_store.get(sid) or intake.initial_intake_state()
    state = dict(state)

    user_msg = (message or "").strip()
    if not user_msg:
        if not state.get("grade"):
            reply = "Hi! What grade is your child in? (K1, K2, or grade 1-12.)"
        elif not state.get("zip_code"):
            reply = "What is your Boston address or ZIP code? (We need this to look up eligible schools.)"
        else:
            reply = "How can I help? I can show your eligible schools again if you want."
        history.append((message, reply))
        _session_store[sid] = state
        return history, ""

    # Run intake orchestrator: grade → address/ZIP, guardrails, then eligibility
    state, (outcome_kind, outcome_data) = intake.step(state, user_msg)
    _session_store[sid] = state

    if outcome_kind == "guardrail_grade_fail":
        reply = outcome_data
        history.append((user_msg, reply))
        return history, ""

    if outcome_kind == "guardrail_geo_fail":
        reply = outcome_data
        history.append((user_msg, reply))
        return history, ""

    if outcome_kind == "need_grade":
        if _is_greeting(user_msg):
            reply = "Hi! I can help you find eligible BPS schools. What grade is your child in? (K1, K2, or grade 1-12.)"
            history.append((user_msg, reply))
            return history, ""
        # Unclear or gibberish input: deterministic fallback for reliability
        if not _looks_like_grade_or_question(user_msg):
            reply = (
                "I didn't quite get that. What grade is your child in? (K1, K2, or grade 1-12.)"
            )
            history.append((user_msg, reply))
            return history, ""
        # Grade-related question but no grade value provided yet.
        reply = "I can help with that. To start, what grade is your child in? (K1, K2, or grade 1-12.)"
        history.append((user_msg, reply))
        return history, ""

    if outcome_kind == "need_address":
        # Unclear or gibberish: fixed "I didn't understand" and re-ask for address
        if not _looks_like_address_or_question(user_msg):
            reply = (
                "I didn't quite get that. What is your Boston address or ZIP code? (We need this to look up eligible schools.)"
            )
            history.append((user_msg, reply))
            return history, ""
        reply = "Thanks! What is your Boston address or ZIP code? (We need this to look up eligible schools.)"
        history.append((user_msg, reply))
        return history, ""

    if outcome_kind == "eligibility_result":
        result = outcome_data
        if not result.ok:
            reply = result.message
        else:
            intro = llm.get_intro_for_schools()
            body = _format_schools_list(result.schools)
            reply = intro + "\n\n" + body
        history.append((user_msg, reply))
        return history, ""

    if outcome_kind == "converse":
        if _looks_like_street_address(user_msg) and not _contains_zip_5(user_msg):
            reply = (
                "Thanks. Please include your 5-digit Boston ZIP code too (for example, 02119) "
                "so I can look up eligible schools."
            )
            history.append((user_msg, reply))
            return history, ""
        reply = (
            "Please share your Boston address or ZIP code, and I will look up eligible schools. "
            "If you only know the ZIP code, that is okay."
        )
        history.append((user_msg, reply))
        return history, ""

    if outcome_kind == "already_have_both":
        if _is_reask_schools_request(user_msg):
            result = eligibility.get_eligible_schools(
                grade=state.get("grade") or "",
                zip_code=state.get("zip_code") or "",
                street_number=state.get("street_number") or "1",
                street_name=state.get("street_name") or "Washington St",
            )
            if not result.ok:
                reply = result.message
            else:
                reply = llm.get_intro_for_schools() + "\n\n" + _format_schools_list(result.schools)
            history.append((user_msg, reply))
            return history, ""
        if _is_zip_confirmation_question(user_msg):
            reply = (
                "Yes, thanks. I already have your location details in this chat. "
                "I can show your eligible schools again whenever you want."
            )
            history.append((user_msg, reply))
            return history, ""

    # default conversational fallback when we are outside intake prompts
    messages = _history_to_messages(history) + [{"role": "user", "content": user_msg}]
    system_prompt = llm.BPS_SYSTEM_PROMPT
    # User already gave grade and Boston address; don't re-ask or deflect
    if outcome_kind == "already_have_both":
        grade = state.get("grade") or ""
        zip_code = state.get("zip_code") or ""
        system_prompt = (
            system_prompt
            + "\n\nIMPORTANT: The user has already provided their child's grade ("
            + str(grade)
            + ") and Boston address/ZIP ("
            + str(zip_code)
            + "). Do NOT ask for grade or address again. Keep your reply short and helpful."
        )
    reply = llm.get_chat_reply(messages, system_prompt=system_prompt)
    if not reply:
        reply = (
            "What is your Boston address or ZIP code? (We need this to look up eligible schools.)"
            if state.get("grade")
            else "What grade is your child in? (K1, K2, or grade 1–12.)"
        )
    history.append((user_msg, reply))
    return history, ""


def _clear_session(request: gr.Request | None = None) -> tuple[list, str]:
    """Clear chat and in-memory session state for this UI session (no PII persisted)."""
    sid = _session_id(request)
    _session_store.pop(sid, None)
    return [], ""


def main():
    """Launch the chat UI."""
    with gr.Blocks(title="BPS Enrollment Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Boston Public Schools Enrollment Chatbot")
        gr.Markdown(
            "This tool helps Boston families find eligible BPS schools based on grade and address. "
            "*(Informational only — always confirm with Boston Public Schools.)*"
        )
        chatbot = gr.Chatbot(label="Chat")
        msg = gr.Textbox(placeholder="Type your message...", label="Message", show_label=False)
        clear = gr.Button("Clear")

        def submit(user_msg, hist, request: gr.Request):
            hist_new, _ = chat(user_msg, hist, request)
            return hist_new, ""

        # request is injected by Gradio (not listed in inputs); state is in-memory keyed by session ID
        msg.submit(submit, [msg, chatbot], [chatbot, msg])
        clear.click(_clear_session, None, [chatbot, msg], queue=False)

    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", share=False, server_port=port)


if __name__ == "__main__":
    main()
