---
title: BPS Enrollment Chatbot
emoji: 🏫
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
short_description: Find eligible BPS schools by grade and address.
---

# Boston Public Schools Enrollment Chatbot

A conversational chatbot that guides Boston families through BPS school eligibility: it collects grade and address (or ZIP), applies guardrails, and returns eligible schools—all based on official BPS rules. Informational only; users should always confirm with Boston Public Schools.

**Product spec:** [docs/prd/prd.md](docs/prd/prd.md)

---

## Run locally

1. **Create and activate a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**

   ```bash
   python app.py
   ```

   The Gradio UI will open in your browser (often at `http://127.0.0.1:7860`).

---

## Environment variables

- **PORT** — Port for the web server (default `7860`). If you see "address already in use", run with another port, e.g. `PORT=7862 python app.py`.

**Eligibility client (Phase 1):**

- **ELIGIBILITY_API_BASE_URL** — Base URL for the BPS Discover Service (default: `http://api.mybps.org/BPSDiscoverService/Schools.svc`). Only change if you use a different endpoint.
- **USE_MOCK_ELIGIBILITY** — Set to `true`, `1`, or `yes` to use mock eligibility results (no real API calls). Useful for local UI testing when the API is unavailable.
- **ELIGIBILITY_REQUEST_TIMEOUT** — Timeout in seconds for API calls (default `15`).
- **ELIGIBILITY_API_KEY** — Optional; BPS Discover Service currently uses no authentication. Reserved for future use.

**LLM (conversational guide, Phase 2):**

- **HF_TOKEN** or **HUGGINGFACE_TOKEN** — HuggingFace token for the Inference API. Without it, the app still runs with fixed prompts. Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- **HF_INFERENCE_MODEL** — Model ID for chat (default: `HuggingFaceH4/zephyr-7b-beta`). Use a model that supports chat completion on the free tier.

Copy `.env.example` to `.env` and set values as needed. Never commit `.env` (see `.gitignore`).

---

## HuggingFace Space

This app is designed to run as a HuggingFace Space. The Space’s entry point is `app.py` (no other entry points). The YAML block at the top of this README tells HuggingFace to use Gradio and `app.py`.

**How to deploy**

1. On [huggingface.co](https://huggingface.co), go to **Spaces** → **Create new Space**.
2. Choose **Gradio** as the SDK and pick a Space name (e.g. `bps-enrollment-chatbot`).
3. **Push this repo to the Space** — see below (Git from your computer, or GitHub).
4. In the Space, open **Settings** → **Variables and secrets** and add any needed variables (e.g. `HF_TOKEN` for the LLM; optional `USE_MOCK_ELIGIBILITY` for testing).
5. The Space will build and run; the chat UI will be the only entry point.

**Push this repo to your Space (step-by-step)**

You have two options. **Option A** uses only HuggingFace and your computer (no GitHub). **Option B** uses GitHub, then connects the Space to it.

**Option A — Push from your computer to HuggingFace (no GitHub)**

1. **Create the Space** (if you haven’t yet): [huggingface.co/spaces](https://huggingface.co/spaces) → **Create new Space** → name it (e.g. `bps-enrollment-chatbot`), choose **Gradio**, then **Create Space**.
2. **Copy the Space’s Git URL.** On the new Space page, click the **“Files and versions”** tab (or the **<>** icon). You’ll see something like:
   `https://huggingface.co/spaces/YOUR_USERNAME/bps-enrollment-chatbot`
   The Git URL to use is: **`https://huggingface.co/spaces/YOUR_USERNAME/bps-enrollment-chatbot`** (same URL; HuggingFace uses it for Git).
3. **Open Terminal** (on Mac: Terminal app; on Windows: Command Prompt or PowerShell) and go to your project folder:
   ```bash
   cd /Users/mturchet/Projects/bpschat
   ```
4. **Turn the folder into a Git repo and push to HuggingFace** (run these commands one by one). Replace `YOUR_USERNAME` and `bps-enrollment-chatbot` with your Space’s URL if different:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: BPS enrollment chatbot"
   git branch -M main
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/bps-enrollment-chatbot
   git push -u space main
   ```
   When Git asks for **password**, paste your **HuggingFace token** (do *not* use your account password — HuggingFace no longer accepts passwords for Git). Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **Write** access, then use your HuggingFace username and that token when prompted.
5. After the push, the Space will rebuild automatically; in a minute or two your app should be live.

**Option B — Use GitHub, then connect the Space to it**

1. Create a **GitHub** account (if you don’t have one) at [github.com](https://github.com).
2. Create a **new repository** on GitHub (e.g. `bps-enrollment-chatbot`). Do **not** add a README or .gitignore (your project already has them).
3. In your project folder, run (replace `YOUR_GITHUB_USERNAME` and repo name with yours):
   ```bash
   cd /Users/mturchet/Projects/bpschat
   git init
   git add .
   git commit -m "Initial commit: BPS enrollment chatbot"
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/bps-enrollment-chatbot.git
   git branch -M main
   git push -u origin main
   ```
4. Create the Space: [huggingface.co/spaces](https://huggingface.co/spaces) → **Create new Space**. Choose **Gradio**, then under **Repository** or **Clone from**, select **Clone from a repository** and paste your GitHub repo URL (e.g. `https://github.com/YOUR_GITHUB_USERNAME/bps-enrollment-chatbot`). Create the Space.
5. HuggingFace will copy the code from GitHub and build. To update the Space later, push changes to GitHub; you can then trigger a rebuild on the Space (or use HF’s “Sync” if you link the Space to the GitHub repo).

After deploying:

- Set any required environment variables in the Space’s **Settings → Variables and secrets**.
- The Space README is this file; the title and short description come from the YAML at the top.

---

## Project structure (MVP)

- `app.py` — Single entry point: launches the chat UI (Gradio), runnable locally and as the HF Space app. Wires intake (grade → address), LLM for conversation, and eligibility for school results.
- `services/eligibility.py` — Eligibility client and guardrails: grade (K1–12) and geography (Boston only); calls BPS Discover Service for eligible schools. Config via env (see above).
- `services/llm.py` — LLM integration (HuggingFace Inference API): system prompt and chat replies. Eligibility and school names never come from the LLM; they come only from the eligibility module.
- `services/school_data.py` — Minimal BPS school display data: when the API returns only IDs or minimal fields, provides name, level, and address for display. Uses `data/bps_schools.json`.
- `data/bps_schools.json` — Mapping of school ID → display fields (name, level, address). Update from [bostonpublicschools.org](https://www.bostonpublicschools.org/schools) or the BPS Discover Service when BPS rules or school list change.
- `requirements.txt` — Python dependencies (Gradio, requests, pydantic, etc.).
- `docs/prd/prd.md` — Product requirements and goals.
- `docs/api/avela-bps-eligibility-api.md` — BPS Discover Service API notes.
