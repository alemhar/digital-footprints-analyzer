import os
import sys
from typing import Optional

# Optional: load .env if available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def mask(value: Optional[str], keep: int = 4) -> str:
    if not value:
        return "<missing>"
    v = str(value)
    if len(v) <= keep * 2:
        return "*" * len(v)
    return v[:keep] + "*" * (len(v) - keep * 2) + v[-keep:]


def echo_env():
    print("Azure OpenAI environment:")
    print(f"  AZURE_OPENAI_ENDPOINT = {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"  OPENAI_API_VERSION    = {os.getenv('OPENAI_API_VERSION')}")
    print(f"  OPENAI_ENGINE         = {os.getenv('OPENAI_ENGINE')}  (deployment name)")
    print(f"  AZURE_OPENAI_API_KEY  = {mask(os.getenv('AZURE_OPENAI_API_KEY'))}")
    print()


def main() -> int:
    echo_env()

    try:
        from openai import AzureOpenAI  # type: ignore
    except Exception as e:
        print("[ERROR] Could not import openai. Install it and retry:")
        print("        python -m pip install --upgrade openai\n")
        print("Import error:", e)
        return 2

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    version = os.getenv("OPENAI_API_VERSION")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("OPENAI_ENGINE")  # must be your Azure DEPLOYMENT NAME

    missing = [
        name for name, val in [
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("OPENAI_API_VERSION", version),
            ("AZURE_OPENAI_API_KEY", key),
            ("OPENAI_ENGINE", deployment),
        ] if not val
    ]
    if missing:
        print("[ERROR] Missing required environment variables:", ", ".join(missing))
        print("Ensure they are set in your .env and then run this script again.")
        return 3

    print("Attempting a minimal chat.completions call...\n")
    try:
        client = AzureOpenAI(
            api_key=key,
            api_version=version,
            azure_endpoint=endpoint,
        )
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Azure connectivity check: reply with 'OK'"}],
            temperature=0,
            max_tokens=5,
        )
        # Newer SDKs return .choices[0].message.content; keep it robust
        choice = resp.choices[0]
        content = getattr(getattr(choice, "message", {}), "content", None) or getattr(choice, "text", None) or str(resp)
        print("[SUCCESS] Received response from Azure OpenAI.")
        print("  Deployment:", deployment)
        print("  Endpoint:  ", endpoint)
        print("  Version:   ", version)
        print("  Sample:    ", content)
        return 0
    except Exception as e:
        print("[ERROR] Azure OpenAI request failed.")
        print("  This often means the deployment name (OPENAI_ENGINE) is wrong, or the")
        print("  endpoint/version doesn’t match the deployed model.")
        print("  Double-check in Azure Portal → OpenAI → Deployments → copy the Deployment name.")
        print("\nException:")
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
