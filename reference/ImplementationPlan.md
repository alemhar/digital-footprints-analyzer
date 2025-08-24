# Digital Footprint Analyzer — Implementation Plan (BYOK + Gradio)

This plan aligns the current crewAI template in `src/dfa/` with the BYOK Gradio application described in `reference/Notes.MD`.

---

## Findings

- **Current stack**
  - `crewAI` template present with YAML configs.
  - Entrypoints in `pyproject.toml`: `dfa`, `run_crew`, `train`, `replay`, `test`.
  - No Gradio UI yet.
- **Key files**
  - `src/dfa/crew.py`: defines agents (`researcher`, `reporting_analyst`) and tasks (`research_task`, `reporting_task`) from YAML configs.
  - `src/dfa/config/agents.yaml` and `tasks.yaml`: generic research/report example using `topic` and `current_year`.
  - `src/dfa/main.py`: runs the crew locally with sample inputs; no UI.
  - `reference/Notes.MD`: specifies a BYOK model with Gradio Blocks, dynamic API-key visibility, and a 3-agent privacy audit.

---

## Gaps vs Target (Notes.MD)

- **UI**: Gradio Blocks app is missing.
- **BYOK**: No dynamic LLM provider selection or runtime API key handling.
- **Agents/Tasks**: Current crew is generic research/report; needs OSINT/privacy agents and tasks (finder, risk classifier, removal advisor) and report output.
- **Search Tool**: Needs Serper integration with `SERPER_API_KEY` sourced from `.env` or env var.
- **Dependencies**: `pyproject.toml` only has `crewai[tools]`. Missing `gradio`, `python-dotenv`, `openai`, `langchain-google-genai`.
- **Docs/Run**: README lacks Gradio run instructions.

---

## Implementation Plan

1. Project setup and dependencies
   - Add dependencies in `pyproject.toml`:
     - gradio
     - python-dotenv
     - openai
     - langchain-google-genai
   - Keep `crewai[tools]` (provides `SerperDevTool`).
   - Optional: add `google-search-results` only if needed by Serper library in your environment.

2. Environment and BYOK policy
  - BYOK runtime:
    - Accept user-provided key for OpenAI/Groq/Gemini in UI (masked). Hide provider key when provider is `Ollama`.
    - Accept user-provided Search API key (`SERPER_API_KEY`) in UI (masked). This is REQUIRED; no environment fallback.

3. LLM configuration layer
   - Create `src/dfa/llm_loader.py` (or integrate within UI file) implementing:
     - `LLM_CONFIGS` mapping (OpenAI, Groq, Ollama, Google Gemini).
     - `get_llm_instance(provider_name, api_key)` returning either a `ChatGoogleGenerativeAI` instance or an OpenAI-compatible config dict (api_base, model_name, api_key).

4. Crew logic integration
   - Option A (minimal change): Implement a function `run_privacy_crew(full_name, llm_config)` in the UI file to assemble crewAI `Agent`/`Task`/`Crew` objects directly (finder, risk classifier, removal advisor) using `SerperDevTool`.
   - Option B (modular): Extend `src/dfa/crew.py` to include privacy agents/tasks and select them based on a mode flag. For speed, start with Option A.

5. Gradio BYOK UI
  - Create `src/dfa/ui_app.py` (or `app.py` at repo root) with Gradio Blocks UI:
    - Inputs: `Full Name to Audit`, `LLM Provider` radio, `Provider API Key` textbox (dynamic visibility), `Search API Key (Serper)` textbox (REQUIRED).
    - Output: Markdown report.
    - Events:
      - `provider.change` toggles API key visibility.
      - `submit.click` runs `process_audit_request()` generator yielding status and result.
      - Validate that Search API key is provided; otherwise show a clear error.

6. Connectors and outputs
  - Use `SerperDevTool(api_key=search_key)` for search. If the key is missing, abort with an actionable error.
  - Generate a structured markdown output with sources, risk classification, and removal recommendations. Optionally also write `report.md` to disk for downloads.

7. Tests and manual verification
   - Add a smoke test invoking `run_privacy_crew("Jane Doe", mock_llm_or_small_model)` in offline mode where feasible.
   - Manual: Run the Gradio UI and validate provider switching and key requirements.

8. Documentation
   - Update `README.md` with:
     - How to run crew (`crewai run`) vs UI (`python -m dfa.ui_app` or `python app.py`).
     - BYOK policy and required environment variables.
     - Notes for Hugging Face Spaces: remove cloud LLM keys; keep only `SERPER_API_KEY`.

---

## File/Change Summary

- Add: `src/dfa/ui_app.py` (Gradio UI + BYOK + crew run wrapper)
- Add: `src/dfa/llm_loader.py` (optional; otherwise inline in UI)
- Update: `pyproject.toml` to include UI dependencies
- Keep: `src/dfa/crew.py` (template untouched initially)
- Docs: update `README.md`

---

## Execution Order (Suggested)

1) Dependencies + env setup
2) Implement `ui_app.py` with BYOK + crew wrapper
3) Verify UI locally with OpenAI or Gemini
4) Add optional file-based report output
5) Update README and add a basic test

---

## Open Questions

- Should we keep the generic research crew alongside the privacy crew or fully migrate? (Plan assumes we keep both for now.)
- Preferred default provider and models?
- Deployment target (local only vs Hugging Face Spaces)?
