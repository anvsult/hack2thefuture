"""Ticket summaries (Gemini, with a template fallback) and Slack delivery (with dry-run).

Configuration comes from environment variables (loaded from .env by app.py):
- SLACK_WEBHOOK_URL: Slack incoming webhook
- GEMINI_API_KEY: Google AI Studio key (free tier)
- GEMINI_MODEL: optional model override
"""
from __future__ import annotations

import os

import requests

SLACK_ENV = "SLACK_WEBHOOK_URL"
GEMINI_ENV = "GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-3.5-flash-lite"

SYSTEM = (
    "You write short alerts for an insurance claims operations team. "
    "Use only the facts given. Two sentences, under 45 words, plain language. "
    "Sentence 1: what is wrong and why it matters (renewal risk or review delay). "
    "Sentence 2: the recommended next step for the named owner. No greetings, no markdown."
)


def template_summary(a: dict) -> str:
    risk = (f" About ${a['premium_at_risk']:,.0f} of renewal premium is at risk."
            if a["premium_at_risk"] else "")
    return (f"{a['claim_id']} ({a['claim_type']}, {a['city']}, {a['team']}) is flagged: "
            f"{a['reason']}.{risk} Next step: {a['next_step']}.")


def ai_configured() -> bool:
    return bool(os.environ.get(GEMINI_ENV))


def ai_summary(a: dict, timeout_s: float = 8.0) -> tuple[str, str]:
    """Return (text, source). Falls back to the template on any failure, so the demo never blocks."""
    if not ai_configured():
        return template_summary(a), "template"
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ[GEMINI_ENV],
                              http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        facts = "\n".join(f"{k}: {a[k]}" for k in (
            "claim_id", "claim_type", "claim_amount", "team", "city", "rule", "tier",
            "reason", "age", "predicted", "days_to_renewal", "premium_at_risk", "owner",
            "next_step"))
        resp = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
            contents=facts,
            config=types.GenerateContentConfig(system_instruction=SYSTEM, max_output_tokens=1024),
        )
        text = (resp.text or "").strip()
        return (text, "gemini") if text else (template_summary(a), "template")
    except Exception:  # bad key, quota, network, timeout: keep the demo moving
        return template_summary(a), "template"


def slack_text(a: dict, summary: str, sim_date: str) -> str:
    icon = ":red_circle:" if a["tier"] == "critical" else ":large_yellow_circle:"
    label = "Slow claim" if a["rule"] == "slow_claim" else "Senior-review stall"
    return (f"{icon} *{label} ({a['tier']})* | {a['claim_id']} | {a['team']}, {a['city']} | "
            f"sim date {sim_date}\n{summary}\n*Owner:* {a['owner']}")


def send_slack(text: str, dry_run: bool) -> str:
    if dry_run:
        return "dry-run"
    url = os.environ.get(SLACK_ENV)
    if not url:
        return f"failed: {SLACK_ENV} not set"
    try:
        r = requests.post(url, json={"text": text}, timeout=5)
        return "sent" if r.ok else f"failed: HTTP {r.status_code}"
    except requests.RequestException as e:
        return f"failed: {type(e).__name__}"


def slack_configured() -> bool:
    return bool(os.environ.get(SLACK_ENV))
