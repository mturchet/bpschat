"""
BPS Enrollment Chatbot — Entry point.

Runnable locally: python app.py
Also used as the HuggingFace Space app entry (no other entry points per PRD).

Optional env: PORT — port to use (default 7860). Use if 7860 is already in use.
"""

from __future__ import annotations

import os

# Load .env so HF_TOKEN, PORT, etc. are available when running locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr

from services import intake, llm

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
        reply = llm.get_chat_reply(
            _history_to_messages(history),
            system_prompt=llm.BPS_SYSTEM_PROMPT,
        )
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
        messages = _history_to_messages(history) + [{"role": "user", "content": user_msg}]
        reply = llm.get_chat_reply(messages, system_prompt=llm.BPS_SYSTEM_PROMPT)
        history.append((user_msg, reply))
        return history, ""

    if outcome_kind == "need_address":
        messages = _history_to_messages(history) + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "Great. What is your Boston address or ZIP code? (We need this to look up eligible schools.)"},
        ]
        reply = llm.get_chat_reply(
            messages,
            system_prompt=llm.BPS_SYSTEM_PROMPT + "\n\nYou just received the grade. Reply with exactly one short sentence asking for their Boston address or ZIP code. Do not repeat the grade.",
        )
        if not reply or ("address" not in reply.lower() and "zip" not in reply.lower()):
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

    # "converse" or "already_have_both": use LLM for reply
    messages = _history_to_messages(history) + [{"role": "user", "content": user_msg}]
    reply = llm.get_chat_reply(messages, system_prompt=llm.BPS_SYSTEM_PROMPT)
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
