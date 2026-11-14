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


def ai_summary(a: dict, timeout_s: float = 15.0) -> tuple[str, str]:  # Gemini rejects deadlines < 10 s
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


def digest_text(alerts: list[dict], sim_date: str, max_lines: int = 10) -> str:
    """One Slack message for the alerts beyond the per-step cap, listing each claim."""
    lines = [f":package: *{len(alerts)} more alert{'s' if len(alerts) > 1 else ''} on {sim_date}* "
             "(lower priority; tickets created, see the ticket queue):"]
    for a in alerts[:max_lines]:
        dot = ":red_circle:" if a["tier"] == "critical" else ":large_yellow_circle:"
        lines.append(f"{dot} {a['claim_id']} · {a['team']}, {a['city']} · {a['reason']} · {a['next_step'].split(';')[0]}")
    if len(alerts) > max_lines:
        lines.append(f"…and {len(alerts) - max_lines} more in the ticket queue.")
    return "\n".join(lines)


def slack_blocks(a: dict, text: str, ticket_id: int) -> list[dict]:
    """Alert text plus action buttons. Button values carry the ticket id (and target team)."""
    buttons = []
    step = a["next_step"]
    if a["rule"] == "review_stall":
        buttons.append(_button("Escalate to senior reviewer", "escalate", f"{ticket_id}", "primary"))
    elif "Reassign to " in step:
        team = step.split("Reassign to ", 1)[1].split(" (")[0]
        buttons.append(_button(f"Reassign to {team}", "reassign", f"{ticket_id}|{team}", "primary"))
    else:
        buttons.append(_button("Fast-track", "fasttrack", f"{ticket_id}", "primary"))
    if "Retention" in step:
        buttons.append(_button("Book retention call", "book_call", f"{ticket_id}"))
    buttons.append(_button("Mark resolved", "resolve", f"{ticket_id}"))
    public = os.environ.get("PUBLIC_URL")
    if public:
        buttons.append({"type": "button", "action_id": "open", "url": public,
                        "text": {"type": "plain_text", "text": "Open dashboard"}})
    return [{"type": "section", "text": {"type": "mrkdwn", "text": text}},
            {"type": "actions", "elements": buttons}]


def _button(label: str, action_id: str, value: str, style: str | None = None) -> dict:
    b = {"type": "button", "action_id": action_id, "value": value,
         "text": {"type": "plain_text", "text": label}}
    if style:
        b["style"] = style
    return b


def send_slack(text: str, dry_run: bool, blocks: list[dict] | None = None) -> str:
    if dry_run:
        return "dry-run"
    url = os.environ.get(SLACK_ENV)
    if not url:
        return f"failed: {SLACK_ENV} not set"
    try:
        body = {"text": text, **({"blocks": blocks} if blocks else {})}
        r = requests.post(url, json=body, timeout=5)
        return "sent" if r.ok else f"failed: HTTP {r.status_code}"
    except requests.RequestException as e:
        return f"failed: {type(e).__name__}"


def slack_configured() -> bool:
    return bool(os.environ.get(SLACK_ENV))
