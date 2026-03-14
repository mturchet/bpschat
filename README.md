---
title: 6.C395 Chatbot
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.23.3
python_version: "3.10"
app_file: app.py
pinned: false
secrets:
  - HF_TOKEN
---

# Boston Public Schools Enrollment Chatbot

A multi-agent conversational chatbot that helps parents and legal guardians explore Boston Public Schools enrollment options.

## Run locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Environment variables

Create a `.env` file:

```env
# Provider switch
LLM_PROVIDER=huggingface

# Hugging Face provider
HF_TOKEN=your_hf_token

# Optional OpenAI provider for local testing
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_api_key
# OPENAI_MODEL=gpt-4o-mini
```

## Deploy to Hugging Face Space

1. Create a Gradio Space on Hugging Face.
2. Push this repo to your Space remote.
3. Add required secrets in Space settings:
   - `HF_TOKEN` (if `LLM_PROVIDER=huggingface`)
4. The Space entrypoint is `app.py`.

## Notes

- This assistant is scoped to Boston Public Schools guidance.
- Recommendations are informational; families should confirm final eligibility with Boston Public Schools directly.
