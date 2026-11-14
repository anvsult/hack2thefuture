"""Ticket summaries (Claude, with a template fallback) and Slack delivery (with dry-run)."""
from __future__ import annotations

import os

import requests

MODEL = "claude-opus-5-5"
SLACK_ENV = "SLACK_WEBHOOK_URL"

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


def claude_summary(a: dict, timeout: float = 8.0) -> tuple[str, str]:
    """Return (text, source). Falls back to the template on any failure, so the demo never blocks."""
    try:
        import anthropic

        client = anthropic.Anthropic(timeout=timeout, max_retries=0)
        facts = "\n".join(f"{k}: {a[k]}" for k in (
            "claim_id", "claim_type", "claim_amount", "team", "city", "rule", "tier",
            "reason", "age", "predicted", "days_to_renewal", "premium_at_risk", "owner",
            "next_step"))
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=2048,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM,
            messages=[{"role": "user", "content": facts}],
        )
        if resp.stop_reason == "refusal":
            return template_summary(a), "template"
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return (text, "claude") if text else (template_summary(a), "template")
    except Exception:  # missing key, network, timeout, API error: keep the demo moving
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
