"""
LLM integration for the BPS Enrollment Chatbot.

Uses HuggingFace Inference API (or fallback when no token). The LLM is used only for:
- Conversation: asking one question at a time and friendly wording.
- Optional short intro when presenting eligibility results.

Eligibility and school names NEVER come from the LLM; they come only from the
eligibility module (single source of truth). The system prompt enforces no fabrication.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Config (from environment)
# -----------------------------------------------------------------------------

def _get_token() -> str:
    return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip()


def _get_model() -> str:
    return (os.environ.get("HF_INFERENCE_MODEL") or "HuggingFaceH4/zephyr-7b-beta").strip()


# -----------------------------------------------------------------------------
# System prompt (no fabrication of eligibility or school names)
# -----------------------------------------------------------------------------

BPS_SYSTEM_PROMPT = """You are a friendly guide for Boston Public Schools (BPS) enrollment. Your role is to help Boston families find which BPS schools their child may be eligible to attend based on grade and address.

Rules you must follow:
- Ask one question at a time. Do not ask for grade and address in the same message.
- First ask for the child's grade (K1, K2, or grade 1 through 12). Then ask for the family's Boston address or ZIP code.
- You do not decide eligibility or list schools. The system will look up eligible schools; you only ask questions and use a warm, clear tone.
- You do NOT determine whether a ZIP code is in Boston. The system checks that automatically. If the user gives a ZIP, ask them to submit it (e.g. "Please enter your ZIP code so the system can check") or wait for the system's response; never tell them yourself that a ZIP is or isn't in Boston.
- Never invent school names, eligibility rules, or which schools a family can attend. If the user asks for a list of schools, say that the system will look them up once you have their grade and address.
- This tool is informational only. Do not help with enrollment or applications; suggest families confirm with Boston Public Schools directly.
- Keep replies short (one or two sentences). Be welcoming and concise."""


# -----------------------------------------------------------------------------
# Chat completion
# -----------------------------------------------------------------------------

def get_chat_reply(
    messages: List[Dict[str, str]],
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.3,
) -> str:
    """
    Get the next assistant reply from the LLM given conversation messages.

    messages: list of {"role": "user"|"assistant"|"system", "content": "..."}.
    system_prompt: if provided, prepended as a system message (otherwise uses BPS_SYSTEM_PROMPT).
    model: HuggingFace model ID; uses env HF_INFERENCE_MODEL or default if not set.
    max_tokens: cap on response length.
    temperature: lower = more deterministic.

    Returns the assistant message text. If the LLM is unavailable (no token or API error),
    returns a short fallback message so the app still works.
    """
    token = _get_token()
    if not token:
        return (
            "I'm here to help you find BPS schools. What grade is your child in? (K1, K2, or grade 1–12.) "
            "*(To use the full conversational guide, set HF_TOKEN or HUGGINGFACE_TOKEN in your environment.)*"
        )

    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        return (
            "What grade is your child in? (K1, K2, or grade 1–12.) "
            "*(LLM not available: install huggingface_hub and set HF_TOKEN.)*"
        )

    prompt = system_prompt or BPS_SYSTEM_PROMPT
    full_messages: List[Dict[str, str]] = [{"role": "system", "content": prompt}]
    for m in messages:
        if isinstance(m, dict) and m.get("content"):
            full_messages.append({"role": m.get("role", "user"), "content": m["content"]})

    model_id = model or _get_model()
    try:
        client = InferenceClient(token=token)
        out = client.chat_completion(
            messages=full_messages,
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception:
        return (
            "What grade is your child in? (K1, K2, or grade 1–12.) "
            "*(The assistant is temporarily unavailable; you can still try entering your grade and address.)*"
        )

    # ChatCompletionOutput has .choices[0].message.content
    choices = getattr(out, "choices", None) or []
    if not choices:
        return "I didn't get a reply. What grade is your child in? (K1, K2, or grade 1–12.)"
    msg = getattr(choices[0], "message", None)
    if not msg:
        return "What grade is your child in? (K1, K2, or grade 1–12.)"
    content = getattr(msg, "content", None)
    if content is None:
        content = getattr(msg, "content", "")
    text = (content or "").strip()
    return text or "What grade is your child in? (K1, K2, or grade 1–12.)"


def get_intro_for_schools() -> str:
    """
    Optional: ask the LLM for a one-line intro before we display the eligibility list.
    We never pass school names to the LLM; we only ask for a generic intro.
    If LLM is unavailable, returns a fixed intro.
    """
    token = _get_token()
    if not token:
        return "Based on BPS eligibility rules, here are schools your child may be eligible to attend:"

    intro_prompt = (
        "Say exactly one short sentence to introduce a list of school eligibility results. "
        "Do not list any school names. Example: 'Based on BPS eligibility rules, here are schools your child may be eligible to attend.'"
    )
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token)
        out = client.chat_completion(
            messages=[
                {"role": "system", "content": "You output only the requested sentence, nothing else."},
                {"role": "user", "content": intro_prompt},
            ],
            model=_get_model(),
            max_tokens=80,
            temperature=0.2,
        )
        choices = getattr(out, "choices", None) or []
        if choices:
            msg = getattr(choices[0], "message", None)
            content = getattr(msg, "content", None) if msg else None
            if content and isinstance(content, str) and content.strip():
                return content.strip()
    except Exception:
        pass
    return "Based on BPS eligibility rules, here are schools your child may be eligible to attend:"
