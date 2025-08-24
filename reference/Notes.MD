# Building a Personal Digital Footprint Analyzer with crewAI and Gradio

This document provides a complete guide to building a professional-grade "Personal Digital Footprint Analyzer." This version implements a **"Bring Your Own Key" (BYOK)** model, allowing users to run the analysis using their own API keys. This is the most secure and scalable way to deploy a public-facing AI tool.

The application features a dynamic UI where the API key input field appears based on the user's selected LLM provider.

**Technology Stack:**
* **Python:** The core programming language.
* **crewAI:** The framework for orchestrating AI agents.
* **Gradio:** Used for building the interactive web interface with dynamic components.

---

## 1. Important Ethical Considerations

The ethical foundation of this project remains paramount. The BYOK model enhances this by placing the responsibility and cost of API usage directly on the end-user, ensuring the developer is not an unwilling party to any misuse. The tool's purpose remains personal privacy empowerment.

---

## 2. Project Architecture

The core architecture still uses the "Privacy Audit Crew" of AI agents. The key change is in the configuration and user interface layer.

### "Bring Your Own Key" (BYOK) Model
Instead of the server's secret keys being used for every user, the application now requires the user to provide their own API key at runtime.

* **User Responsibility:** The cost and rate limits of API calls are tied to the user's key, not the developer's.
* **Dynamic UI:** The Gradio interface will intelligently show or hide the API key input field. If the user selects "Ollama" (which runs locally and needs no key), the field will disappear. For all other cloud-based providers, it will appear.

---

## 3. Setup and Installation

### Hugging Face Secrets: A Critical Update

With the BYOK model, your Hugging Face space no longer needs to store the LLM API keys. This is a major security improvement.

1.  Go to your Hugging Face Space **Settings**.
2.  Under **"Repository secrets,"** you must **DELETE** the following secrets if you added them previously:
    * `OPENAI_API_KEY`
    * `GROQ_API_KEY`
    * `GOOGLE_API_KEY`
3.  You **MUST KEEP** the `SERPER_API_KEY`, as the application's search functionality still runs on the server side using your key.

### Local `.env` File
Your local `.env` file is now purely for your own development and testing. It should still contain your keys to allow you to run the app locally without pasting your key every time. The structure is the same.

### `requirements.txt`
The `requirements.txt` file remains the same as the previous version. Ensure it is in your project folder before deploying.
```
gradio
crewai
crewai-tools
python-dotenv
openai
google-search-results
langchain-google-genai
```

---

## 4. Full Python Code (`app.py`)

The application code requires a significant update to handle the dynamic UI and the BYOK logic. We will switch from `gr.Interface` to the more powerful `gr.Blocks` for this.

Paste this entire script into your `app.py` file.

```python
import os
import gradio as gr
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# Import for Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables for local development
load_dotenv()

# --- 1. Central LLM Configuration (Simplified for BYOK) ---
LLM_CONFIGS = {
    "OpenAI": {"model": "gpt-4o"},
    "Groq": {"model": "llama3-8b-8192", "api_base": "[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)"},
    "Ollama": {"model": os.getenv("OLLAMA_MODEL_NAME", "llama3:8b"), "api_base": "http://localhost:11434/v1"},
    "Google Gemini": {"model": "gemini-1.5-pro-latest", "is_google": True}
}

# --- 2. Dynamic LLM Loader (Refactored for BYOK) ---
def get_llm_instance(provider_name, api_key):
    """Loads the correct LLM instance based on the provider and user's API key."""
    if provider_name == "Ollama":
        api_key = "ollama" # crewAI requires a placeholder

    if not api_key:
        raise ValueError(f"API key is required for {provider_name}.")

    config = LLM_CONFIGS[provider_name]

    if config.get("is_google"):
        return ChatGoogleGenerativeAI(model=config["model"], google_api_key=api_key)
    
    # For OpenAI-compatible APIs
    return {
        "api_base": config.get("api_base", "[https://api.openai.com/v1](https://api.openai.com/v1)"),
        "model_name": config["model"],
        "api_key": api_key
    }

# --- 3. Main Crew Logic ---
def run_privacy_crew(full_name, llm_config):
    """Runs the privacy audit crew with the dynamically configured LLM."""
    search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))
    
    # For Google Gemini, llm_config is the instantiated object
    # For others, it's a dict of parameters for the environment
    llm_instance = None
    if isinstance(llm_config, dict):
        os.environ["OPENAI_API_BASE"] = llm_config["api_base"]
        os.environ["OPENAI_MODEL_NAME"] = llm_config["model_name"]
        os.environ["OPENAI_API_KEY"] = llm_config["api_key"]
    else:
        llm_instance = llm_config

    finder_agent = Agent(role='Personal Digital Footprint Finder', goal='Find public data for a name.', backstory='An OSINT expert.', verbose=True, tools=[search_tool], llm=llm_instance)
    risk_classifier_agent = Agent(role='Data Privacy Risk Classifier', goal='Analyze data and classify privacy risk.', backstory='A privacy analyst.', verbose=True, llm=llm_instance)
    removal_advisor_agent = Agent(role='Privacy Removal Advisor', goal='Create a report with removal advice.', backstory='An expert in data removal.', verbose=True, llm=llm_instance)

    # Define Tasks
    find_data_task = Task(description=f"Search for '{full_name}'.", expected_output=f"A list of URLs and snippets mentioning '{full_name}'.", agent=finder_agent)
    classify_risk_task = Task(description="Classify findings by risk (High, Medium, Low).", expected_output="A list of findings with risk classifications.", agent=risk_classifier_agent, context=[find_data_task])
    generate_report_task = Task(description="Compile a final report with removal steps.", expected_output=f"A markdown report for {full_name}.", agent=removal_advisor_agent, context=[classify_risk_task])

    # Create and Run the Crew
    privacy_crew = Crew(agents=[finder_agent, risk_classifier_agent, removal_advisor_agent], tasks=[find_data_task, classify_risk_task, generate_report_task], process=Process.sequential, verbose=2)
    return privacy_crew.kickoff()

# Wrapper function for Gradio UI
def process_audit_request(full_name, llm_provider, user_api_key):
    if not full_name:
        return "Please enter a full name."
    if llm_provider != "Ollama" and not user_api_key:
        return f"Please enter an API key for {llm_provider}."
        
    try:
        yield "Configuring AI model..."
        llm_config = get_llm_instance(llm_provider, user_api_key)
        
        yield "Running privacy audit... (This may take a moment)"
        result = run_privacy_crew(full_name, llm_config)
        yield result
    except Exception as e:
        yield f"An error occurred: {e}"

# --- 4. Gradio UI with Dynamic Components (gr.Blocks) ---
def update_key_input_visibility(provider_name):
    """Updates the visibility of the API key input box."""
    if provider_name == "Ollama":
        # Hide the API key box for Ollama
        return gr.Textbox(visible=False)
    else:
        # Show the API key box for all other providers
        return gr.Textbox(
            visible=True,
            placeholder=f"Enter your {provider_name} API Key...",
            label=f"{provider_name} API Key"
        )

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Personal Digital Footprint Analyzer")
    gr.Markdown("Enter your name and provide your own API key to generate a privacy report. Your key is not stored.")

    with gr.Row():
        with gr.Column(scale=1):
            name_input = gr.Textbox(label="Full Name to Audit")
            llm_provider_radio = gr.Radio(
                list(LLM_CONFIGS.keys()),
                label="Select LLM Provider",
                value="OpenAI"
            )
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                placeholder="Enter your OpenAI API Key...",
                type="password",
                visible=True # Initially visible for the default "OpenAI" option
            )
            submit_button = gr.Button("Start Audit", variant="primary")
        
        with gr.Column(scale=2):
            output_report = gr.Markdown(label="Audit Report")

    # Event listener to update the API key box visibility when the radio button changes
    llm_provider_radio.change(
        fn=update_key_input_visibility,
        inputs=llm_provider_radio,
        outputs=api_key_input
    )

    # Event listener for the submit button
    submit_button.click(
        fn=process_audit_request,
        inputs=[name_input, llm_provider_radio, api_key_input],
        outputs=output_report
    )
    
    gr.Examples(
        [["Jane Doe", "OpenAI"]],
        inputs=[name_input, llm_provider_radio]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
```

---

## 5. How to Use the Deployed App

The workflow for an end-user is now:
1.  Navigate to your Hugging Face Space URL.
2.  Select their preferred LLM Provider from the radio buttons.
3.  If they choose anything other than "Ollama," the API key input box will appear.
4.  They must paste their **own** API key into the box.
5.  They enter the full name they wish to audit.
6.  Click "Start Audit". The analysis runs using their key, and the report is displayed.

---

## 6. Positioning on LinkedIn (Updated Example Post)

This new model is a significant professional feature that you should highlight.

> I'm excited to share the latest version of my Personal Digital Footprint Analyzer!
>
> Based on great feedback, I've re-architected it to use a **"Bring Your Own Key" (BYOK)** model. This is the standard for public AI tools, ensuring user privacy and security while making the tool freely accessible to anyone with an API key.
>
> **Key Features:**
> * ✅ **Choose Your Engine:** Select from OpenAI, Google Gemini, Groq, or a local Ollama model.
> * 🔐 **BYOK Security:** Your API key is used directly for the request and is never stored. This puts you in full control.
> * 🕵️‍♂️ **In-Depth Analysis:** The AI crew still performs a deep-dive OSINT search to find and classify your public data.
>
> This was a fantastic exercise in building dynamic UIs with Gradio Blocks and creating a more robust, scalable application architecture.
>
> Check it out and audit your own digital presence!
>
> [Link to your App]
>
> `#Python #AI #CyberSecurity #Privacy #OSINT #Gradio #crewAI #BYOK #SoftwareArchitecture`