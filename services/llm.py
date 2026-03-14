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
    # Match benchmark/studentversion defaults unless explicitly overridden.
    return (os.environ.get("HF_INFERENCE_MODEL") or "meta-llama/Llama-3.1-8B-Instruct").strip()


# -----------------------------------------------------------------------------
# System prompt (no fabrication of eligibility or school names)
# -----------------------------------------------------------------------------

BPS_SYSTEM_PROMPT = """You are a warm, practical assistant for Boston Public Schools (BPS) enrollment questions.

Rules you must follow:
- Use a natural, conversational style similar to an advising chat.
- If the user asks a broad question, you may ask 1-3 concise clarifying questions, often as a numbered list.
- Give practical guidance about school choice and enrollment steps when possible.
- You do not make official eligibility determinations. The system handles official eligibility once it has grade and address/ZIP.
- Never claim to have run official eligibility checks unless the system has already returned results in the chat.
- Avoid over-verbose answers; 1 short paragraph or a brief bulleted/numbered list is usually best."""


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
    returns an empty string so the app can apply an outcome-specific fallback.
    """
    token = _get_token()
    if not token:
        return ""

    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        return ""

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
        return ""

    # ChatCompletionOutput has .choices[0].message.content
    choices = getattr(out, "choices", None) or []
    if not choices:
        return ""
    msg = getattr(choices[0], "message", None)
    if not msg:
        return ""
    content = getattr(msg, "content", None)
    if content is None:
        content = getattr(msg, "content", "")
    text = (content or "").strip()
    return text


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
