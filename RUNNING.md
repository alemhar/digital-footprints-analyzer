# How to Run the Digital Footprint Analyzer (Windows + existing venv)

This guide shows how to run the new Gradio UI that accepts Full Name, Nick Name, and Online Cosine inputs. It assumes you already have a virtual environment created in the project (e.g., `.venv/`).

## 1) Activate your virtual environment

From the project root `d:\Projects\DigitalFootprintAudit\dfa`:

```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1
```

If you are using Command Prompt (cmd.exe):

```bat
.\.venv\Scripts\activate.bat
```

If you are using Git Bash:

```bash
source .venv/Scripts/activate
```

Once activated, your prompt should show `(venv)` or similar.

## 2) Install project dependencies

Install the package in editable mode so that the `src/` layout is importable and dependencies from `pyproject.toml` (like `gradio`) are installed:

```powershell
python -m pip install -e .
```

If you prefer using the venv interpreter explicitly:

```powershell
.\.venv\Scripts\python -m pip install -e .
```

## 3) Run the Gradio UI

Option A (direct file):

```powershell
python src/dfa/ui_app.py
```

Option B (module form) — only if your environment is set to discover `src/` as a module path:

```powershell
python -m dfa.ui_app
```

After launch, the terminal will display a local URL, usually:

```
http://127.0.0.1:7860
```

Open that URL in your web browser. You should see a simple form with:
- Full Name (required)
- Nick Name (optional)
- Online Cosine (numeric)

Click Submit to see a summary of your inputs.

## 4) Notes and troubleshooting

- If you see `ModuleNotFoundError: No module named 'gradio'`, it means dependencies are not installed in the active environment. Ensure the venv is activated and run `python -m pip install -e .` again.
- If `python -m dfa.ui_app` fails with `No module named dfa`, use the direct file path method: `python src/dfa/ui_app.py`.
- Python version should be between 3.10 and 3.13 as specified in `pyproject.toml`.

## 5) Crew (CLI) — optional

The repository also includes a crewAI example. To run it via the crew CLI from the project root:

```powershell
crewai run
```

This will execute the example crew defined under `src/dfa/` and may generate a `report.md` as output. This is separate from the Gradio UI and is optional.

---

In the next steps, we will connect the UI inputs to the privacy audit crew, add search/LLM API keys, and generate a structured report.
