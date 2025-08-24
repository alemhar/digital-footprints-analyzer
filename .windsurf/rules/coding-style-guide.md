---
trigger: manual
---

# Coding Rules for the Digital Footprint Analyzer Project

### Philosophy

These rules are designed to maintain a clean, readable, and maintainable codebase for our single-file Gradio application. The primary goal is to ensure clarity for both human developers and the AI assistants within an IDE like Windsurf. By following these conventions, we can guide the AI's suggestions effectively without triggering overly complex refactoring on our intentionally simple project.

---

### 1. General Principles

1.1. **Simplicity First.**
Prefer simple, clear solutions over complex implementations. The goal is a straightforward and understandable script.

1.2. **Avoid Duplication (DRY).**
Before writing new code or importing a library, check if similar functionality already exists within the file.

1.3. **Focused Changes.**
Make only the requested changes. Do not modify unrelated code unless it is a necessary part of the task. This helps the AI stay focused.

1.4. **Clean Codebase.**
Maintain a clean and organized codebase. Remove any temporary or unnecessary scripts, files, or commented-out code blocks after use.

### 2. Project Structure & Files

2.1. **Maintain a Single-File Application.**
The core application logic **must** remain within a single `app.py` file. This is a deliberate design choice for simplicity. Do not let AI tools refactor the core logic into multiple files.

2.2. **Standard File Organization.**
Your project directory should only contain the following essential files:
```

/
|-- app.py              \# Main application script
|-- requirements.txt    \# Project dependencies
|-- .env                \# Environment variables (API keys)
|-- .gitignore          \# Files to be ignored by Git
|-- README.md           \# Project documentation
|-- CODING\_RULES.md     \# This file



### 3. The Single Source of Truth

3.1. **The `LLM_CONFIGS` Dictionary is the Source of Truth.**
The `LLM_CONFIGS` dictionary at the top of `app.py` is the **single source of truth** for all LLM provider information.

3.2. **Consult Before Changing.**
Before modifying any agent or LLM-related logic, **always** consult the `LLM_CONFIGS` dictionary first to ensure consistency.

3.3. **Update First.**
When adding a new LLM provider or changing a model name, the `LLM_CONFIGS` dictionary **must** be the first place you make the update. All other logic will follow from this central configuration.

### 4. Coding Workflow

4.1. **Consider Dependencies.**
Before implementing changes in one function, consider how it might affect other functions within the `app.py` file.

4.2. **Preserve Working Patterns.**
The existing patterns (e.g., the dynamic LLM loader, the separation of UI and core logic) work well. Do not refactor them unless explicitly needed for a new feature.

4.3. **Refresh AI Sessions.**
When working in Windsurf or any AI-powered IDE, restart the AI chat session periodically to clear the context and maintain optimal performance, especially after making significant changes.

### 5. Naming, Formatting, and Style (PEP 8)

5.1. **Use `snake_case` for variables and functions.**
    - **Good:** `run_privacy_audit`, `api_key_input`
    - **Bad:** `RunPrivacyAudit`, `apiKeyInput`

5.2. **Use `UPPER_SNAKE_CASE` for constants.**
    - **Good:** `LLM_CONFIGS`
    - **Bad:** `llm_configs`

5.3. **Be Descriptive.**
Variable and function names should clearly describe their purpose to provide context for the AI.
    - **Good:** `finder_agent`, `update_key_input_visibility`
    - **Bad:** `agent1`, `update_ui`

5.4. **Use Logical Sections.**
Break up the `app.py` script into clear, logical sections using commented headers.
```python
# --- 1. Central LLM Configuration ---
# ...
# --- 2. Dynamic LLM Loader Function ---
# ...


### 6\. Comments and Docstrings

6.1. **Write Docstrings for All Functions.**
Every function must have a docstring explaining its purpose, arguments (`Args:`), and return value (`Returns:`). This is the most effective way to give the AI context.

6.2. **Comment the "Why," Not the "What."**
Use inline comments to explain non-obvious logic.
\- **Good:** `# Hide the API key box for Ollama as it runs locally`
\- **Bad:** `# Set visible to False`

### 7\. Environment and Dependencies

7.1. **Never Hardcode Secrets.**
All API keys **must** be loaded from the `.env` file. Never commit the `.env` file to version control.

7.2. **Keep `requirements.txt` Clean.**
The `requirements.txt` file should only list the project's direct dependencies.

### 8\. Gradio UI Development

8.1. **Separate UI Logic from Core Logic.**
The Gradio `gr.Blocks` section defines the UI. The actual work should be done in separate functions (`run_privacy_crew`, `process_audit_request`).

8.2. **Use Descriptive Names for UI Components.**
Give Gradio components meaningful variable names to make event listener logic clear.
\- **Good:** `llm_provider_radio`, `submit_button`
\- **Bad:** `radio1`, `btn`
