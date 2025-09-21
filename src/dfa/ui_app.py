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
        return "### Error\nCrew module not available. Ensure dependencies are installed and run via editable install or PYTHONPATH includes src/."
    if not os.getenv("OPENAI_API_KEY"):
        return "### Error\nOPENAI_API_KEY not found in environment/.env. Please set it to run the full crew."

    # Prepare inputs expected by YAML
    inputs = {
        "topic": (full_name or "").strip() or "Unknown Subject",
        "current_year": str(datetime.now().year),
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

    try:
        result = Dfa().crew().kickoff(inputs=inputs)
    except Exception as e:
        return f"### Error\n{e}"

    # crewAI may return rich objects; cast to string for display
    return f"## Crew Output\n\n{result}"

def build_ui():
    with gr.Blocks(title="Digital Footprint Analyzer - Input Form") as demo:
        gr.Markdown("""
        # Digital Footprint Analyzer
        Enter your details below to begin. This first step only validates inputs.
        """)

        with gr.Row():
            full_name = gr.Textbox(label="Full Name", placeholder="e.g., Jane Doe", lines=1)
            linkedin_url = gr.Textbox(label="LinkedIn Profile URL (optional)", placeholder="https://www.linkedin.com/in/username", lines=1)
            facebook_url = gr.Textbox(label="Facebook Profile URL (optional)", placeholder="https://www.facebook.com/username", lines=1)

        with gr.Row():
            twitter_url = gr.Textbox(label="Twitter/X URL (optional)", placeholder="https://twitter.com/handle", lines=1)
            instagram_url = gr.Textbox(label="Instagram URL (optional)", placeholder="https://www.instagram.com/username", lines=1)

        with gr.Row():
            github_url = gr.Textbox(label="GitHub URL (optional)", placeholder="https://github.com/username", lines=1)
            website_url = gr.Textbox(label="Personal Website (optional)", placeholder="https://yourdomain.com", lines=1)

        with gr.Row():
            usernames = gr.Textbox(
                label="Usernames/handles (optional)",
                placeholder="Comma-separated, e.g., @jdoe, j_doe, john.doe",
                info="Examples from: X/Twitter, Instagram, GitHub, Reddit, YouTube, TikTok, Medium, Kaggle, Stack Overflow, etc.",
                lines=1,
            )
            location_hint = gr.Textbox(
                label="Location hint (optional)",
                placeholder="City, Country (e.g., Manila, Philippines)",
                lines=1,
            )

        submit = gr.Button("Submit")
        preview_json = gr.Button("Preview Research JSON")
        run_serper = gr.Button("Run Finder (Serper)")
        analyze_btn = gr.Button("Analyze & Summarize")
        full_crew_btn = gr.Button("Run Full Crew (OpenAI)")
        output = gr.Markdown(label="Result")
        json_out = gr.JSON(label="Finder Seed (JSON)")
        serper_out = gr.JSON(label="Finder Results (Serper)")
        analyzed_out = gr.JSON(label="Analyzed Findings (Grouped)")
        report_md = gr.Markdown(label="Markdown Report")
        crew_md = gr.Markdown(label="Full Crew Output")

        submit.click(
            fn=process_inputs,
            inputs=[full_name, linkedin_url, facebook_url, twitter_url, instagram_url, github_url, website_url, usernames, location_hint],
            outputs=[output],
        )

        preview_json.click(
            fn=build_research_seed,
            inputs=[full_name, linkedin_url, facebook_url, twitter_url, instagram_url, github_url, website_url, usernames, location_hint],
            outputs=[json_out],
        )

        run_serper.click(
            fn=run_finder_serper,
            inputs=[full_name, linkedin_url, facebook_url, twitter_url, instagram_url, github_url, website_url, usernames, location_hint],
            outputs=[serper_out],
        )

        analyze_btn.click(
            fn=analyze_and_summarize,
            inputs=[full_name, linkedin_url, facebook_url, twitter_url, instagram_url, github_url, website_url, usernames, location_hint],
            outputs=[analyzed_out, report_md],
        )

        full_crew_btn.click(
            fn=run_full_crew,
            inputs=[full_name, linkedin_url, facebook_url, twitter_url, instagram_url, github_url, website_url, usernames, location_hint],
            outputs=[crew_md],
        )

    return demo


def main():
    app = build_ui()
    app.launch()


if __name__ == "__main__":
    main()
