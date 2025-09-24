import os
import json
import time
import re
from urllib.parse import urlparse
import gradio as gr
from typing import Optional
from urllib import request, error
from datetime import datetime

try:
    from dfa.crew import Dfa  # type: ignore
except Exception:
    Dfa = None  # UI will guard against missing crew import

# Attempt to load environment variables from .env if python-dotenv is available,
# otherwise fall back to a minimal loader for SERPER_API_KEY.
def _fallback_load_env():
    # Attempt to locate a .env file starting from current working directory,
    # then the directory of this file.
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
        os.path.join(os.path.dirname(__file__), ".env"),
    ]
    for path in candidates:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
                break
        except Exception:
            # Silently ignore; this is a best-effort fallback.
            pass

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    _fallback_load_env()
else:
    # Even if python-dotenv loaded, ensure SERPER_API_KEY is present; if not, try fallback.
    if not os.getenv("SERPER_API_KEY"):
        _fallback_load_env()


def process_inputs(
    full_name: str,
    linkedin_url: Optional[str],
    facebook_url: Optional[str],
    twitter_url: Optional[str],
    instagram_url: Optional[str],
    github_url: Optional[str],
    website_url: Optional[str],
    usernames: Optional[str],
    location_hint: Optional[str],
):
    """Stub handler that validates and echoes inputs.

    Args:
        full_name: Person's full name.
        linkedin_url: Optional LinkedIn profile URL.
        facebook_url: Optional Facebook profile URL.
        twitter_url: Optional Twitter/X profile URL.
        instagram_url: Optional Instagram profile URL.
        github_url: Optional GitHub profile URL.
        website_url: Optional personal website/portfolio URL.
        usernames: Optional comma-separated list of usernames/handles (e.g., @jdoe, j_doe, john.doe).
        location_hint: Optional city/country hint to disambiguate.

    Returns:
        Markdown string summarizing provided inputs.
    """
    errors = []
    if not full_name or len(full_name.strip()) == 0:
        errors.append("- Full name is required.")

    # Basic URL validations (optional fields)
    def looks_like_url(u: Optional[str]) -> bool:
        if not u:
            return True
        u = u.strip()
        return u.startswith("http://") or u.startswith("https://")

    if linkedin_url and not looks_like_url(linkedin_url):
        errors.append("- LinkedIn URL should start with http:// or https://")
    if facebook_url and not looks_like_url(facebook_url):
        errors.append("- Facebook URL should start with http:// or https://")
    if twitter_url and not looks_like_url(twitter_url):
        errors.append("- Twitter/X URL should start with http:// or https://")
    if instagram_url and not looks_like_url(instagram_url):
        errors.append("- Instagram URL should start with http:// or https://")
    if github_url and not looks_like_url(github_url):
        errors.append("- GitHub URL should start with http:// or https://")
    if website_url and not looks_like_url(website_url):
        errors.append("- Personal website URL should start with http:// or https://")

    # Validate usernames and location
    cleaned_usernames = []
    if usernames:
        if len(usernames) > 200:
            errors.append("- Usernames/handles is too long (max 200 characters).")
        else:
            # Split on commas, strip whitespace, remove leading '@'
            for token in [t.strip() for t in usernames.split(',') if t.strip()]:
                token = token.lstrip('@').strip()
                if token:
                    cleaned_usernames.append(token)

    cleaned_location = (location_hint or "").strip()
    if cleaned_location and len(cleaned_location) > 120:
        errors.append("- Location hint is too long (max 120 characters).")

    if errors:
        return "\n".join(["### Please fix the following:"] + errors)

    md = [
        "## Input Summary",
        f"- Full name: **{full_name.strip()}**",
        f"- LinkedIn URL: **{(linkedin_url or '').strip()}**",
        f"- Facebook URL: **{(facebook_url or '').strip()}**",
        f"- Twitter/X URL: **{(twitter_url or '').strip()}**",
        f"- Instagram URL: **{(instagram_url or '').strip()}**",
        f"- GitHub URL: **{(github_url or '').strip()}**",
        f"- Personal website: **{(website_url or '').strip()}**",
        f"- Usernames/handles: **{', '.join(cleaned_usernames) if cleaned_usernames else ''}**",
        f"- Location hint: **{cleaned_location}**",
        "",
        "This is a stub UI. Next step: wire these inputs into the crew and generate a report.",
    ]
    return "\n".join(md)


def build_research_seed(
    full_name: str,
    linkedin_url: Optional[str],
    facebook_url: Optional[str],
    twitter_url: Optional[str],
    instagram_url: Optional[str],
    github_url: Optional[str],
    website_url: Optional[str],
    usernames: Optional[str],
    location_hint: Optional[str],
):
    """Create a structured JSON payload for the Finder step (no external calls).

    Returns a dict with normalized inputs, suggested search queries, and seed sources.
    """
    # Reuse simple validations from process_inputs for consistency
    def clean_url(u: Optional[str]) -> Optional[str]:
        if not u:
            return None
        s = u.strip()
        return s if s.startswith("http://") or s.startswith("https://") else s

    # Parse usernames (comma-separated, drop leading @)
    normalized_usernames = []
    if usernames:
        for token in [t.strip() for t in usernames.split(',') if t.strip()]:
            token = token.lstrip('@').strip()
            if token:
                normalized_usernames.append(token)

    loc = (location_hint or "").strip()

    provided_urls = {
        "linkedin": clean_url(linkedin_url),
        "facebook": clean_url(facebook_url),
        "twitter": clean_url(twitter_url),
        "instagram": clean_url(instagram_url),
        "github": clean_url(github_url),
        "website": clean_url(website_url),
    }

    # Build suggested queries
    name = (full_name or "").strip()
    quoted_name = f'"{name}"' if name else ""
    base_queries = [
        f"{quoted_name} site:linkedin.com",
        f"{quoted_name} site:facebook.com",
        f"{quoted_name} site:twitter.com",
        f"{quoted_name} site:instagram.com",
        f"{quoted_name} site:github.com",
        f"{quoted_name} site:about.me",
        f"{quoted_name} site:medium.com",
        f"{quoted_name} email OR \"contact\"",
        f"{quoted_name} CV OR resume",
    ]
    if loc:
        base_queries.extend([
            f"{quoted_name} {loc}",
            f"{quoted_name} site:linkedin.com/in {loc}",
        ])

    for uname in normalized_usernames:
        base_queries.extend([
            f"{uname} site:twitter.com",
            f"{uname} site:instagram.com",
            f"{uname} site:github.com",
            f"{uname} site:reddit.com",
            f"{uname} site:youtube.com",
        ])

    payload = {
        "input": {
            "full_name": name,
            "location_hint": loc,
            "usernames": normalized_usernames,
            "provided_urls": provided_urls,
        },
        "suggested_queries": [q for q in base_queries if q.strip()],
        "notes": [
            "Use site: filters to target specific platforms.",
            "Prefer exact-quoted name searches to reduce noise.",
            "Leverage provided profile URLs as ground truth for disambiguation.",
        ],
    }
    return payload


def serper_search(query: str, api_key: str, num_results: int = 5) -> dict:
    """Call Serper's web search API for a single query and return a compact result.

    Uses urllib to avoid adding heavy dependencies.
    """
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query}).encode("utf-8")
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    req = request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        return {"query": query, "error": f"HTTPError {e.code}: {e.reason}"}
    except error.URLError as e:
        return {"query": query, "error": f"URLError: {e.reason}"}
    except Exception as e:
        return {"query": query, "error": f"Unexpected error: {e}"}

    # Extract a compact set of fields
    organic = data.get("organic", [])
    compact = []
    for item in organic[:num_results]:
        compact.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "site": item.get("sitelinks", {}),
            }
        )
    return {"query": query, "results": compact}


def run_finder_serper(
    full_name: str,
    linkedin_url: Optional[str],
    facebook_url: Optional[str],
    twitter_url: Optional[str],
    instagram_url: Optional[str],
    github_url: Optional[str],
    website_url: Optional[str],
    usernames: Optional[str],
    location_hint: Optional[str],
):
    """Run a minimal Finder using Serper across a few suggested queries.

    Reads SERPER_API_KEY from environment or returns an error JSON.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return {
            "error": "SERPER_API_KEY not found in environment. Add it to your .env or environment variables.",
        }

    seed = build_research_seed(
        full_name,
        linkedin_url,
        facebook_url,
        twitter_url,
        instagram_url,
        github_url,
        website_url,
        usernames,
        location_hint,
    )

    queries = seed.get("suggested_queries", [])
    # Limit the number of queries for responsiveness
    max_queries = 6
    selected = queries[:max_queries]

    aggregated = {
        "seed": seed["input"],
        "queries": selected,
        "results": [],
        "meta": {
            "provider": "serper",
            "limit_per_query": 5,
            "executed": 0,
        },
    }

    for q in selected:
        r = serper_search(q, api_key, num_results=5)
        aggregated["results"].append(r)
        aggregated["meta"]["executed"] += 1
        time.sleep(0.4)  # be polite and avoid hitting rate limits

    return aggregated


def _extract_domain(url: Optional[str]) -> str:
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
        # Strip www.
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4})\b")
_HIGH_RISK_DOMAINS = {
    "pastebin.com", "ghostbin.com", "hastebin.com",
    "truepeoplesearch.com", "mylife.com", "spokeo.com", "beenverified.com", "whitepages.com",
}
_HIGH_RISK_KEYWORDS = {"password", "credential", "leak", "ssn", "api_key", "token", "private key"}


def _score_risk(domain: str, title: str, snippet: str) -> str:
    t = (title or "").lower()
    s = (snippet or "").lower()

    if domain in _HIGH_RISK_DOMAINS:
        return "High"
    if any(kw in t or kw in s for kw in _HIGH_RISK_KEYWORDS):
        return "High"
    if _EMAIL_RE.search(snippet or "") or _PHONE_RE.search(snippet or ""):
        return "Medium"
    # Default to Low
    return "Low"


def analyze_aggregated(aggregated: dict) -> dict:
    """Deduplicate links, group by domain, and assign heuristic risk levels."""
    seen = set()
    grouped: dict = {}
    results = aggregated.get("results", [])
    for entry in results:
        items = entry.get("results", [])
        for it in items:
            link = it.get("link")
            if not link or link in seen:
                continue
            seen.add(link)
            domain = _extract_domain(link)
            title = it.get("title")
            snippet = it.get("snippet")
            risk = _score_risk(domain, title or "", snippet or "")
            rec = {"title": title, "link": link, "snippet": snippet, "domain": domain, "risk": risk}
            g = grouped.setdefault(domain or "unknown", {"items": [], "risk_counts": {"High": 0, "Medium": 0, "Low": 0}})
            g["items"].append(rec)
            g["risk_counts"][risk] += 1

    # Add counts per domain
    for d, g in grouped.items():
        g["count"] = len(g["items"])

    # Summary totals
    summary = {"domains": len(grouped), "totals": {"High": 0, "Medium": 0, "Low": 0}}
    for g in grouped.values():
        for k in ("High", "Medium", "Low"):
            summary["totals"][k] += g["risk_counts"].get(k, 0)

    return {"summary": summary, "grouped": grouped}


def generate_markdown_report(full_name: str, analysis: dict) -> str:
    lines = []
    lines.append(f"# Privacy Findings for {full_name}\n")
    totals = analysis.get("summary", {}).get("totals", {})
    lines.append("## Risk Summary")
    lines.append(f"- High: {totals.get('High', 0)}")
    lines.append(f"- Medium: {totals.get('Medium', 0)}")
    lines.append(f"- Low: {totals.get('Low', 0)}\n")

    lines.append("## Findings by Domain")
    grouped = analysis.get("grouped", {})
    # Sort by risk priority then count
    def domain_sort_key(item):
        d, g = item
        rc = g.get("risk_counts", {})
        return (-rc.get("High", 0), -rc.get("Medium", 0), -g.get("count", 0), d)
    for domain, g in sorted(grouped.items(), key=domain_sort_key):
        lines.append(f"### {domain} ({g.get('count', 0)} results)")
        rc = g.get("risk_counts", {})
        lines.append(f"- High: {rc.get('High', 0)} | Medium: {rc.get('Medium', 0)} | Low: {rc.get('Low', 0)}")
        for it in g.get("items", [])[:10]:  # cap per domain in the report preview
            title = it.get("title") or it.get("link")
            link = it.get("link")
            risk = it.get("risk")
            snippet = it.get("snippet") or ""
            lines.append(f"  - [{title}]({link}) — Risk: {risk}\n    - {snippet}")
        lines.append("")

    lines.append("---")
    lines.append("This is an automated, heuristic review of publicly accessible search results. Verify each item before taking action.")
    return "\n".join(lines)


def analyze_and_summarize(
    full_name: str,
    linkedin_url: Optional[str],
    facebook_url: Optional[str],
    twitter_url: Optional[str],
    instagram_url: Optional[str],
    github_url: Optional[str],
    website_url: Optional[str],
    usernames: Optional[str],
    location_hint: Optional[str],
):
    aggregated = run_finder_serper(
        full_name,
        linkedin_url,
        facebook_url,
        twitter_url,
        instagram_url,
        github_url,
        website_url,
        usernames,
        location_hint,
    )
    if isinstance(aggregated, dict) and aggregated.get("error"):
        # Return error on JSON output and a short markdown message
        return aggregated, f"### Error\n{aggregated.get('error')}"
    analysis = analyze_aggregated(aggregated)
    report = generate_markdown_report(full_name, analysis)
    return analysis, report


def run_full_crew(
    full_name: str,
    linkedin_url: Optional[str],
    facebook_url: Optional[str],
    twitter_url: Optional[str],
    instagram_url: Optional[str],
    github_url: Optional[str],
    website_url: Optional[str],
    usernames: Optional[str],
    location_hint: Optional[str],
):
    if Dfa is None:
        # Try a lazy import by adding project src/ to sys.path
        try:
            import sys
            here = os.path.abspath(os.path.dirname(__file__))
            project_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir, os.pardir))
            src_path = os.path.join(project_root, "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            from dfa.crew import Dfa as _Dfa  # type: ignore
        except Exception:
            return "### Error\nCrew module not available. Ensure dependencies are installed and run via editable install or PYTHONPATH includes src/."
        else:
            # assign for subsequent calls
            globals()["Dfa"] = _Dfa

    if not os.getenv("OPENAI_API_KEY"):
        # Try Azure OpenAI fallback
        az_key = os.getenv("AZURE_OPENAI_API_KEY")
        az_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        az_version = os.getenv("OPENAI_API_VERSION")
        az_engine = os.getenv("OPENAI_ENGINE")  # Azure deployment name
        if az_key and az_endpoint and az_version and az_engine:
            os.environ["OPENAI_API_KEY"] = az_key
            os.environ["OPENAI_API_BASE"] = az_endpoint.rstrip("/")
            os.environ["OPENAI_BASE_URL"] = az_endpoint.rstrip("/")
            os.environ["OPENAI_API_TYPE"] = "azure"
            os.environ["OPENAI_API_VERSION"] = az_version
            # Some clients look for deployment name under different vars
            os.environ["OPENAI_ENGINE"] = az_engine
            os.environ["OPENAI_DEPLOYMENT_NAME"] = az_engine
            os.environ["OPENAI_API_MODEL"] = az_engine
            # Also set LiteLLM/Azure-specific aliases for broader compatibility
            os.environ["AZURE_API_KEY"] = az_key
            os.environ["AZURE_API_BASE"] = az_endpoint.rstrip("/")
            os.environ["AZURE_API_VERSION"] = az_version
            os.environ["AZURE_API_DEPLOYMENT_NAME"] = az_engine
            # Force model name envs to match the Azure deployment name
            os.environ["MODEL"] = az_engine
            os.environ["OPENAI_MODEL_NAME"] = az_engine
            os.environ["LLM_MODEL"] = az_engine
            # Base URL alias used by some clients
            os.environ["BASE_URL"] = az_endpoint.rstrip("/")
        else:
            return (
                "### Error\n"
                "No OPENAI_API_KEY found. For Azure, set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
                "OPENAI_API_VERSION, and OPENAI_ENGINE (deployment name)."
            )

    # Prepare inputs expected by YAML / crew
    resolved_model = (
        os.getenv("OPENAI_ENGINE")
        or os.getenv("OPENAI_MODEL_NAME")
        or os.getenv("MODEL")
        or "gpt-4"
    )
    inputs = {
        "topic": (full_name or "").strip() or "Unknown Subject",
        "current_year": str(datetime.now().year),
        "model": resolved_model,
        # Additional context not used by YAML by default but can be referenced later
        "linkedin_url": (linkedin_url or "").strip(),
        "facebook_url": (facebook_url or "").strip(),
        "twitter_url": (twitter_url or "").strip(),
        "instagram_url": (instagram_url or "").strip(),
        "github_url": (github_url or "").strip(),
        "website_url": (website_url or "").strip(),
        "usernames": (usernames or "").strip(),
        "location_hint": (location_hint or "").strip(),
    }

    # Emit debug info to help diagnose model routing
    debug = (
        "### Crew Debug\n"
        f"- Resolved model (inputs.model): `{inputs['model']}`\n"
        f"- OPENAI_ENGINE: `{os.getenv('OPENAI_ENGINE')}`\n"
        f"- OPENAI_API_BASE: `{os.getenv('OPENAI_API_BASE')}`\n"
        f"- OPENAI_BASE_URL: `{os.getenv('OPENAI_BASE_URL')}`\n"
        f"- OPENAI_API_VERSION: `{os.getenv('OPENAI_API_VERSION')}`\n"
        f"- OPENAI_API_TYPE: `{os.getenv('OPENAI_API_TYPE')}`\n"
    )

    try:
        result = Dfa().crew().kickoff(inputs=inputs)
    except Exception as e:
        return debug + "\n\n" + f"### Error\n{e}"

    # crewAI may return rich objects; cast to string for display
    return debug + "\n\n" + f"## Crew Output\n\n{result}"

def _flatten_analysis(analysis: dict) -> list:
    grouped = analysis.get("grouped", {}) or {}
    items = []
    for domain, g in grouped.items():
        for it in g.get("items", []) or []:
            items.append(it)
    # Sort by risk High > Medium > Low, then by domain
    order = {"High": 0, "Medium": 1, "Low": 2}
    items.sort(key=lambda x: (order.get((x or {}).get("risk", "Low"), 3), (x or {}).get("domain", "")))
    return items


def _render_finding_md(idx: int, total: int, it: dict) -> str:
    if not it:
        return "### Review\nNo findings."
    title = it.get("title") or it.get("link") or "Untitled"
    link = it.get("link") or ""
    risk = it.get("risk") or "Low"
    domain = it.get("domain") or ""
    snippet = it.get("snippet") or ""
    return (
        f"### Finding {idx+1} of {total}\n"
        f"- Domain: {domain}\n"
        f"- Risk: {risk}\n"
        f"- Title: [{title}]({link})\n\n"
        f"{snippet}"
    )


def _render_instruction_md(step_idx: int, total_steps: int, step: dict) -> str:
    title = step.get("title") or f"Step {step_idx+1}"
    rationale = step.get("rationale") or ""
    queries = step.get("prioritized_queries") or step.get("queries") or []
    lines = [
        f"## Strategy Step {step_idx+1} of {total_steps}",
        f"### {title}",
        f"{rationale}\n",
        "#### Suggested queries:",
    ]
    for q in queries[:5]:
        lines.append(f"- {q}")
    return "\n".join(lines)


def _generate_strategy_plan(
    full_name: str,
    linkedin_url: Optional[str],
    facebook_url: Optional[str],
    twitter_url: Optional[str],
    instagram_url: Optional[str],
    github_url: Optional[str],
    website_url: Optional[str],
    usernames: Optional[str],
    location_hint: Optional[str],
    max_steps: int = 5,
):
    # Use Azure OpenAI to produce a lightweight plan of steps (instructions only)
    required = [
        ("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY")),
        ("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT")),
        ("OPENAI_API_VERSION", os.getenv("OPENAI_API_VERSION")),
        ("OPENAI_ENGINE (deployment name)", os.getenv("OPENAI_ENGINE")),
    ]
    missing = [k for k, v in required if not v]
    if missing:
        return {"error": f"Missing Azure settings: {', '.join(missing)}"}

    try:
        from openai import AzureOpenAI  # type: ignore
    except Exception as e:
        return {"error": f"openai package missing/incompatible: {e}"}

    subject = (full_name or "").strip() or "Unknown Subject"
    sys_msg = (
        "You are a world-class Online Security & Privacy Strategist. Create a short, prioritized plan "
        "to uncover potentially sensitive public information via OSINT. Return strictly valid JSON."
    )
    user_msg = f"""
Design up to {max_steps} concise strategy steps to investigate the public footprint of {subject}.
Use optional anchors if helpful to disambiguate identity:
- LinkedIn: {linkedin_url}
- Facebook: {facebook_url}
- Twitter/X: {twitter_url}
- Instagram: {instagram_url}
- GitHub: {github_url}
- Personal website: {website_url}
- Usernames/handles: {usernames}
- Location: {location_hint}

Each step should include fields: title, rationale, prioritized_queries (2-4 queries).
Focus on exposed PII (emails, phones, addresses), leaked credentials/tokens/API keys, sensitive docs (resume/spreadsheet), doxxing threats.
Use site: and filetype: filters when helpful. Keep steps short and high impact.

Return JSON: {{"steps": [{{"title": "...", "rationale": "...", "prioritized_queries": ["..."]}}]}}
"""
    try:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        deployment = os.getenv("OPENAI_ENGINE")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=700,
        )
        choice = resp.choices[0]
        text = (
            (getattr(choice, "message", None) and getattr(choice.message, "content", ""))
            or getattr(choice, "text", "")
        )
        data = None
        try:
            data = json.loads(text)
        except Exception:
            start = text.find('{'); end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start:end+1])
                except Exception:
                    data = None
        if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
            return {"error": "Invalid strategy JSON from model", "raw": text}
        return data
    except Exception as e:
        return {"error": str(e)}


def _get_serper_api_key() -> Optional[str]:
    # Try environment first
    key = os.getenv("SERPER_API_KEY")
    if key:
        return key
    # Try loading .env via fallback
    try:
        # If python-dotenv is available, it may have already loaded .env at startup
        # but we also included a minimal fallback earlier in this file.
        _fallback_load_env()
    except Exception:
        pass
    return os.getenv("SERPER_API_KEY")


def start_review(
    full_name: str,
    linkedin_url: Optional[str],
    facebook_url: Optional[str],
    twitter_url: Optional[str],
    instagram_url: Optional[str],
    github_url: Optional[str],
    website_url: Optional[str],
    usernames: Optional[str],
    location_hint: Optional[str],
    state: dict,
):
    # Show immediate feedback
    yield "⏳ Generating strategy plan...", {"mode": "instruction", "steps": [], "step_idx": 0, "findings": [], "finding_idx": -1}
    # 1) Generate a short strategy plan (LLM) — minimal tokens
    plan = _generate_strategy_plan(
        full_name, linkedin_url, facebook_url, twitter_url, instagram_url,
        github_url, website_url, usernames, location_hint, max_steps=5,
    )
    if isinstance(plan, dict) and plan.get("error"):
        yield f"### Strategy Error\n{plan.get('error')}", {"mode": "instruction", "steps": [], "step_idx": 0, "findings": [], "finding_idx": -1}
        return
    steps = plan.get("steps", [])
    if not steps:
        yield "### Strategy\nNo steps generated.", {"mode": "instruction", "steps": [], "step_idx": 0, "findings": [], "finding_idx": -1}
        return
    # 2) Show first instruction, wait for Continue to fetch findings
    state = {
        "mode": "instruction",
        "steps": steps,
        "step_idx": 0,
        "findings": [],
        "finding_idx": -1,
        # Carry inputs for subsequent Serper calls
        "inputs": {
            "full_name": full_name,
            "linkedin_url": linkedin_url,
            "facebook_url": facebook_url,
            "twitter_url": twitter_url,
            "instagram_url": instagram_url,
            "github_url": github_url,
            "website_url": website_url,
            "usernames": usernames,
            "location_hint": location_hint,
        },
    }
    md = _render_instruction_md(0, len(steps), steps[0])
    yield md, state


def next_finding(state: dict):
    state = state or {}
    mode = state.get("mode", "instruction")
    steps = state.get("steps", [])
    step_idx = int(state.get("step_idx", 0))
    findings = state.get("findings", [])
    finding_idx = int(state.get("finding_idx", -1))
    inputs = state.get("inputs", {})

    # If no steps, end
    if not steps or step_idx >= len(steps):
        yield "### Review\nEnd of findings.", state
        return

    if mode == "instruction":
        # Indicate we are executing searches for this step
        yield "⏳ Running searches for current strategy step...", state
        # Run Serper for current step using its prioritized_queries
        step = steps[step_idx]
        queries = step.get("prioritized_queries") or step.get("queries") or []
        if not queries:
            # No queries, advance to next step
            step_idx += 1
            if step_idx >= len(steps):
                yield "### Review\nEnd of findings.", {**state, "step_idx": step_idx}
                return
            md = _render_instruction_md(step_idx, len(steps), steps[step_idx])
            yield md, {**state, "mode": "instruction", "step_idx": step_idx, "findings": [], "finding_idx": -1}
            return

        # Execute a compact finder run for this step (tokenless)
        aggregated = {"results": [], "meta": {"executed": 0}}
        api_key = _get_serper_api_key()
        if not api_key:
            yield "### Error\nSERPER_API_KEY is missing. Add it to your .env and restart.", state
            return
        for q in queries[:3]:
            r = serper_search(q, api_key, num_results=5)
            aggregated["results"].append(r)
            aggregated["meta"]["executed"] += 1
            time.sleep(0.3)
        analysis = analyze_aggregated(aggregated)
        items = _flatten_analysis(analysis)
        if not items:
            # No findings for this step; advance to next instruction
            step_idx += 1
            if step_idx >= len(steps):
                yield "### Review\nEnd of findings.", {**state, "step_idx": step_idx}
                return
            md = _render_instruction_md(step_idx, len(steps), steps[step_idx])
            yield md, {**state, "mode": "instruction", "step_idx": step_idx, "findings": [], "finding_idx": -1}
            return
        # Show first finding for this step
        finding_idx = 0
        md = _render_finding_md(finding_idx, len(items), items[finding_idx])
        yield md, {**state, "mode": "findings", "findings": items, "finding_idx": finding_idx, "step_idx": step_idx}
        return

    # mode == 'findings'
    finding_idx += 1
    if finding_idx < len(findings):
        md = _render_finding_md(finding_idx, len(findings), findings[finding_idx])
        yield md, {**state, "mode": "findings", "finding_idx": finding_idx}
        return
    # Move to next step's instruction
    step_idx += 1
    if step_idx >= len(steps):
        yield "### Review\nEnd of findings.", {**state, "step_idx": step_idx}
        return
    md = _render_instruction_md(step_idx, len(steps), steps[step_idx])
    yield md, {**state, "mode": "instruction", "step_idx": step_idx, "findings": [], "finding_idx": -1}
    return

    # Prepare inputs expected by YAML
    # Resolve model/deployment name for agent configs (agents.yaml uses {model})
    # For Azure, we must pass the DEPLOYMENT NAME (e.g., 'gpt-4').
    resolved_model = (
        os.getenv("OPENAI_ENGINE")
        or os.getenv("OPENAI_MODEL_NAME")
        or os.getenv("MODEL")
        or "gpt-4"
    )
    inputs = {
        "topic": (full_name or "").strip() or "Unknown Subject",
        "current_year": str(datetime.now().year),
        "model": resolved_model,
        # Additional context not used by YAML by default but can be referenced later
        "linkedin_url": (linkedin_url or "").strip(),
        "facebook_url": (facebook_url or "").strip(),
        "twitter_url": (twitter_url or "").strip(),
        "instagram_url": (instagram_url or "").strip(),
        "github_url": (github_url or "").strip(),
        "website_url": (website_url or "").strip(),
        "usernames": (usernames or "").strip(),
        "location_hint": (location_hint or "").strip(),
    }

    # Emit debug info to help diagnose model routing
    debug = (
        "### Crew Debug\n"
        f"- Resolved model (inputs.model): `{inputs['model']}`\n"
        f"- OPENAI_ENGINE: `{os.getenv('OPENAI_ENGINE')}`\n"
        f"- OPENAI_API_BASE: `{os.getenv('OPENAI_API_BASE')}`\n"
        f"- OPENAI_BASE_URL: `{os.getenv('OPENAI_BASE_URL')}`\n"
        f"- OPENAI_API_VERSION: `{os.getenv('OPENAI_API_VERSION')}`\n"
        f"- OPENAI_API_TYPE: `{os.getenv('OPENAI_API_TYPE')}`\n"
    )

    try:
        result = Dfa().crew().kickoff(inputs=inputs)
    except Exception as e:
        return debug + "\n\n" + f"### Error\n{e}"

    # crewAI may return rich objects; cast to string for display
    return debug + "\n\n" + f"## Crew Output\n\n{result}"


def check_azure_connection():
    """Validate Azure OpenAI env and perform a minimal chat completion.

    Returns a Markdown string describing the result or the specific error.
    """
    required = [
        ("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY")),
        ("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT")),
        ("OPENAI_API_VERSION", os.getenv("OPENAI_API_VERSION")),
        ("OPENAI_ENGINE (deployment name)", os.getenv("OPENAI_ENGINE")),
    ]
    missing = [k for k, v in required if not v]
    if missing:
        return (
            "### Azure OpenAI Check\n"
            + "Missing required settings: " + ", ".join(missing) + "\n"
            + "Ensure these are set in your .env and restart the UI."
        )

    try:
        from openai import AzureOpenAI  # type: ignore
    except Exception as e:
        return (
            "### Azure OpenAI Check\n"
            "The 'openai' package is missing or incompatible. Install it in your venv:\n\n"
            "```powershell\n.\\.venv\\Scripts\\python -m pip install --upgrade openai\n```\n\n"
            f"Import error: {e}"
        )

    try:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        deployment = os.getenv("OPENAI_ENGINE")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Azure ping"}],
            temperature=0,
            max_tokens=5,
        )
        choice = resp.choices[0]
        # Newer SDKs: choice.message.content; older: choice.text
        msg = None
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            msg = choice.message.content
        elif hasattr(choice, "text"):
            msg = choice.text
        else:
            msg = str(resp)
        return (
            "### Azure OpenAI Check\n"
            "Connection OK. Received a response from Azure OpenAI.\n\n"
            f"- Deployment: `{deployment}`\n"
            f"- Endpoint: `{os.getenv('AZURE_OPENAI_ENDPOINT')}`\n"
            f"- API Version: `{os.getenv('OPENAI_API_VERSION')}`\n\n"
            f"Sample reply: {msg}"
        )
    except Exception as e:
        return (
            "### Azure OpenAI Check\n"
            "Connection failed. Please verify your Deployment name (OPENAI_ENGINE), Endpoint, and API Version.\n\n"
            f"Error: {e}"
        )


def build_ui():
    hacker_css = """
    /* Base dark background */
    html, body, .gradio-container { background: #0b0f0a !important; color: #baf7c6 !important; }
    /* Panels and blocks */
    .gr-block, .gr-panel, .gr-column, .gr-row, .gr-group, .wrap, .form { background: transparent !important; }
    .block, .form, .row { background: transparent !important; }
    .hacker-pane { border: 1px solid #1f3b20; background: #0e130f !important; box-shadow: 0 0 12px rgba(56,240,88,0.15) inset; }
    /* Titles and text */
    .hacker-title h1 { color: #38f058 !important; font-family: 'Courier New', monospace; text-shadow: 0 0 10px #38f058; }
    .status-bar { color: #6de58a !important; font-family: 'Courier New', monospace; }
    .gradio-container, .gradio-container * { font-family: 'Courier New', monospace; }
    /* Markdown/prose */
    .gr-markdown, .prose, .prose * { color: #baf7c6 !important; }
    .gr-markdown pre, .gr-markdown code, pre, code { background: #0b120d !important; color: #9cf3ad !important; border: 1px solid #1f3b20 !important; }
    .gr-markdown a, .prose a { color: #38f058 !important; text-decoration-color: rgba(56,240,88,0.6) !important; }
    /* Inputs */
    input[type=text], input[type=url], input[type=search], textarea, select { background: #0b120d !important; color: #baf7c6 !important; border: 1px solid #1f3b20 !important; }
    .gr-textbox, .gr-textarea, .gr-input, .gr-select, .gr-number, .gr-text { background: transparent !important; }
    /* Force dark on Gradio label/container wrappers */
    label.container, label.container .input-container { background: #0e130f !important; border-color: #1f3b20 !important; box-shadow: none !important; }
    label.container.show_textbox_border { background: #0e130f !important; border: 1px solid #1f3b20 !important; }
    .input-container textarea, .input-container input { background: #0b120d !important; color: #baf7c6 !important; }
    .input-container textarea::placeholder, .input-container input::placeholder { color: rgba(109,229,138,0.65) !important; }
    /* Buttons */
    .gr-button { background: #102114 !important; border: 1px solid #38f058 !important; color: #baf7c6 !important; }
    .gr-button:hover { background: #16301c !important; }
    /* Tabs (if any) */
    .tabs, .tabitem, .tabitem.selected { background: #0e130f !important; border-color: #1f3b20 !important; }
    /* Scrollbars */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #0b0f0a; }
    ::-webkit-scrollbar-thumb { background: #16301c; border: 1px solid #1f3b20; }
    ::-webkit-scrollbar-thumb:hover { background: #1f2c22; }
    /* Split layout */
    .split { display: grid; grid-template-columns: 1fr 1.25fr; gap: 14px; }
    .soft-sep { height: 1px; background: linear-gradient(90deg, rgba(56,240,88,0.0), rgba(56,240,88,0.35), rgba(56,240,88,0.0)); margin: 8px 0; }
    """

    with gr.Blocks(title="Digital Footprint Analyzer", css=hacker_css) as demo:
        gr.Markdown("""
        # Digital Footprint Analyzer
        """, elem_classes=["hacker-title"])

        with gr.Row(elem_classes=["split"]):
            # Left: Inputs
            with gr.Column(elem_classes=["hacker-pane"]):
                gr.Markdown("Inputs", elem_classes=["status-bar"])
                with gr.Row():
                    full_name = gr.Textbox(label="Full Name", placeholder="e.g., Jane Doe", lines=1)
                    linkedin_url = gr.Textbox(label="LinkedIn URL", placeholder="https://www.linkedin.com/in/username", lines=1)
                with gr.Row():
                    facebook_url = gr.Textbox(label="Facebook URL", placeholder="https://www.facebook.com/username", lines=1)
                    twitter_url = gr.Textbox(label="Twitter/X URL", placeholder="https://twitter.com/username", lines=1)
                with gr.Row():
                    instagram_url = gr.Textbox(label="Instagram URL", placeholder="https://www.instagram.com/username", lines=1)
                    github_url = gr.Textbox(label="GitHub URL", placeholder="https://github.com/username", lines=1)
                with gr.Row():
                    website_url = gr.Textbox(label="Website", placeholder="https://example.com", lines=1)
                    usernames = gr.Textbox(
                        label="Usernames / Handles",
                        placeholder="Comma-separated, e.g., @jdoe, j_doe, john.doe",
                        info="X/Twitter, Instagram, GitHub, Reddit, YouTube, TikTok, Medium, Kaggle, Stack Overflow, etc.",
                        lines=1,
                    )
                location_hint = gr.Textbox(
                    label="Location hint",
                    placeholder="City, Country (e.g., Manila, Philippines)",
                    lines=1,
                )
                with gr.Row():
                    start_audit_btn = gr.Button("Start Audit")
                    continue_btn = gr.Button("Continue")
                    run_full_crew_btn = gr.Button("Run Full Crew")
                    azure_check_btn = gr.Button("Check Azure Connection")

            # Right: Outputs
            with gr.Column(elem_classes=["hacker-pane"]):
                gr.Markdown("Output", elem_classes=["status-bar"])
                review_md = gr.Markdown(label="Finding Review")
                gr.HTML('<div class="soft-sep"></div>')
                azure_md = gr.Markdown(label="Azure Connectivity")
        
        review_state = gr.State({"items": [], "index": 0})

        start_audit_btn.click(
            fn=start_review,
            inputs=[full_name, linkedin_url, facebook_url, twitter_url, instagram_url, github_url, website_url, usernames, location_hint, review_state],
            outputs=[review_md, review_state],
        ).then(
            None, None, None, show_progress=True
        )

        continue_btn.click(
            fn=next_finding,
            inputs=[review_state],
            outputs=[review_md, review_state],
        ).then(
            None, None, None, show_progress=True
        )

        run_full_crew_btn.click(
            fn=run_full_crew,
            inputs=[full_name, linkedin_url, facebook_url, twitter_url, instagram_url, github_url, website_url, usernames, location_hint],
            outputs=[review_md],
        ).then(
            None, None, None, show_progress=True
        )

        azure_check_btn.click(
            fn=check_azure_connection,
            inputs=[],
            outputs=[azure_md],
        )

    return demo


def main():
    app = build_ui()
    app.launch()


if __name__ == "__main__":
    main()
