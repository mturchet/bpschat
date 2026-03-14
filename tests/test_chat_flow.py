"""
Behavior tests for the chat intake flow.

These tests focus on deterministic user experience:
- Greeting should welcome + ask grade (not "I didn't get that").
- After grade is collected, unclear input should re-ask for Boston address/ZIP.
- When user asks to see schools again, bot should re-run eligibility and return results.
"""

import os
import sys
import types
import unittest

# Keep tests local and deterministic: no real API or LLM calls.
os.environ["USE_MOCK_ELIGIBILITY"] = "1"
os.environ["HF_TOKEN"] = ""
os.environ["HUGGINGFACE_TOKEN"] = ""

# app.py imports gradio at module import time; tests don't need UI features.
if "gradio" not in sys.modules:
    sys.modules["gradio"] = types.SimpleNamespace(Request=object)

import app


class _Req:
    def __init__(self, session_hash: str):
        self.session_hash = session_hash


class TestChatFlow(unittest.TestCase):
    def setUp(self):
        app._session_store.clear()
        os.environ["USE_MOCK_ELIGIBILITY"] = "1"
        os.environ["HF_TOKEN"] = ""
        os.environ["HUGGINGFACE_TOKEN"] = ""

    def test_hi_gets_welcome_and_grade_question(self):
        history, _ = app.chat("Hi", [], _Req("t-hi"))
        self.assertTrue(history)
        reply = history[-1][1]
        self.assertIn("What grade is your child in?", reply)
        self.assertNotIn("didn't quite get that", reply.lower())

    def test_after_grade_unclear_text_reasks_for_address(self):
        history, _ = app.chat("Grade 3", [], _Req("t-addr"))
        self.assertIn("address or ZIP code", history[-1][1])

        history, _ = app.chat("hmmm", history, _Req("t-addr"))
        reply = history[-1][1]
        self.assertIn("Boston address or ZIP code", reply)

    def test_can_show_schools_again_after_intake(self):
        req = _Req("t-repeat")
        history, _ = app.chat("3", [], req)
        self.assertIn("address or ZIP code", history[-1][1])

        history, _ = app.chat("02119", history, req)
        first_result = history[-1][1]
        self.assertIn("eligible to attend", first_result)

        history, _ = app.chat("Can you show my schools again?", history, req)
        repeat_result = history[-1][1]
        self.assertIn("eligible to attend", repeat_result)
        self.assertIn("[Mock] Sample Elementary", repeat_result)

    def test_street_without_zip_prompts_for_zip(self):
        req = _Req("t-nozip")
        history, _ = app.chat("5", [], req)
        self.assertIn("address or ZIP code", history[-1][1])

        history, _ = app.chat("100 Warren St Boston MA", history, req)
        reply = history[-1][1]
        self.assertIn("5-digit Boston ZIP code", reply)
        self.assertIn("02119", reply)


if __name__ == "__main__":
    unittest.main()
