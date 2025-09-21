# Digital Footprint Analyzer

Strategist‑led, human‑in‑the‑loop OSINT UI that investigates a subject’s public digital footprint. The app shows the strategist’s current instruction (what to search and why), then reveals findings one‑by‑one when you click Continue. Designed to keep AI token usage under control.

## Features
- Strategist proposes a short, prioritized plan with rationale (visible on screen)
- Continue button executes each step and paginates findings one‑at‑a‑time
- Token‑aware by design: brief plan uses minimal tokens; searches/analysis are lightweight
- Hacker‑style, two‑pane UI (left: inputs; right: outputs)

## Requirements
- Python 3.10 – 3.13
- A `.env` file with:
  - `SERPER_API_KEY` (required for web search)
  - Azure OpenAI (for the strategist plan):
    - `AZURE_OPENAI_API_KEY`
    - `AZURE_OPENAI_ENDPOINT` (e.g., https://<resource>.openai.azure.com/)
    - `OPENAI_API_VERSION` (e.g., 2023-05-15 or your deployment’s supported version)
    - `OPENAI_ENGINE` (your Azure deployment name, e.g., gpt-4)

## Installation
```bash
pip install -U pip
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

## Run the UI
```bash
.\.venv\Scripts\python run_ui.py
```
Then open the local URL shown in the terminal.

## Using the App
1. Enter inputs on the left:
   - Full Name; optional LinkedIn/Twitter/Facebook/Instagram/GitHub/Website
   - Usernames/handles; Location hint
2. Click `Start Audit`:
   - You’ll see “Generating strategy plan…” then Strategy Step 1 (title, rationale, suggested queries)
3. Click `Continue`:
   - You’ll see “Running searches…” then Finding 1 of N for that step
   - Click `Continue` to paginate findings; when finished, the next Strategy Step appears

## Token Control Philosophy
- Minimal strategist tokens (short JSON plan)
- Tokenless execution for search + heuristic analysis
- Human‑in‑the‑loop gating with `Continue` prevents runaway costs

## Troubleshooting Azure
- “Resource not found” almost always means one of these is mismatched:
  - `OPENAI_ENGINE` (must be your Azure Deployment name)
  - `AZURE_OPENAI_ENDPOINT` (resource URL, correct region)
  - `OPENAI_API_VERSION` (must be supported by the deployed model)
- Verify in Azure Portal → your OpenAI resource → Deployments.

## Repository Structure (high level)
- `src/dfa/ui_app.py` — Gradio UI, strategist plan generation, finder, analyzer, step‑by‑step flow
- `src/dfa/crew.py` — Crew/agents definition (LLM client injection for robustness)
- `src/dfa/config/` — Task and agent prompts/config
- `check_azure_openai.py` — Minimal sanity test for Azure OpenAI env

---
Build thoughtfully: show the strategist’s thinking, execute deliberately, control spend.
