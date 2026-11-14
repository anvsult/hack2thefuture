"""Receive Slack button clicks (interactivity), update the ticket, and update the Slack message.

Slack sends a signed, form-encoded POST with one `payload` field (JSON). We must answer
within 3 seconds, so the handler only verifies, updates SQLite, and returns 200; the Slack
message update (and the calendar call) run in a background task.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from urllib.parse import parse_qs

import requests
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

import gcal
import store

MAX_SKEW_S = 300

ACTION_TEXT = {
    "reassign": ":arrows_counterclockwise: Reassigned to {team}",
    "escalate": ":arrow_up: Escalated to a senior reviewer",
    "fasttrack": ":zap: Fast-tracked: named adjuster assigned",
    "resolve": ":white_check_mark: Resolved",
}
STATUS_AFTER = {"reassign": "in progress", "escalate": "in progress", "fasttrack": "in progress",
                "resolve": "resolved", "book_call": None}


def verify(headers, body: bytes, secret: str | None, now: float | None = None) -> bool:
    """Slack request signing: HMAC-SHA256 of 'v0:{timestamp}:{body}' with the signing secret."""
    if not secret:
        return False
    ts = headers.get("x-slack-request-timestamp", "")
    sig = headers.get("x-slack-signature", "")
    if not ts.isdigit() or abs((now or time.time()) - int(ts)) > MAX_SKEW_S:
        return False
    base = b"v0:" + ts.encode() + b":" + body
    expected = "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


def _rebuild(message: dict, clicked: str | None, note: str, keep_buttons: bool) -> list[dict]:
    """Original text blocks + a context note; drop the clicked button (or all buttons)."""
    blocks = [b for b in message.get("blocks", []) if b.get("type") not in ("actions",)]
    actions = next((b for b in message.get("blocks", []) if b.get("type") == "actions"), None)
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": note}]})
    if keep_buttons and actions:
        rest = [e for e in actions["elements"]
                if e.get("action_id") not in (clicked, "noop") and e.get("action_id") != "open"]
        rest += [e for e in actions["elements"] if e.get("action_id") == "open"]
        if any(e.get("action_id") != "open" for e in rest):
            blocks.append({"type": "actions", "elements": rest})
    return blocks


def _post_update(response_url: str, blocks: list[dict], fallback: str) -> None:
    try:
        requests.post(response_url, json={"replace_original": True, "text": fallback, "blocks": blocks},
                      timeout=5)
    except requests.RequestException:
        pass


def _book_call_and_update(ticket_id: int, who: str, user_ref: str, payload: dict) -> None:
    with store.connect() as con:
        t = store.get_ticket(con, ticket_id)
        if t is None:
            return
        link = t.get("calendar_link")
        mode = "api" if link and "calendar.google.com/calendar/event" in link else "link"
        if not link:
            link, mode = gcal.create_event(t)
            label = f":calendar: Retention call booked for {gcal.slot_label()}" if mode == "api" \
                else f":calendar: Retention call link created for {gcal.slot_label()}"
            store.record_action(con, ticket_id, status=None, action=label.split(": ", 1)[1], who=who,
                                calendar_link=link)
    if mode == "api":
        note = f":calendar: Retention call booked for {gcal.slot_label()} by {user_ref} · <{link}|Open event>"
    else:
        note = f":calendar: <{link}|Add the retention call to Google Calendar> ({gcal.slot_label()}) · requested by {user_ref}"
    _post_update(payload["response_url"], _rebuild(payload["message"], "book_call", note, True), note)


async def slack_actions(request: Request) -> Response:
    """HTTP delivery (Interactivity Request URL). Not used when the Slack app runs in Socket Mode."""
    body = await request.body()
    if not verify(request.headers, body, os.environ.get("SLACK_SIGNING_SECRET")):
        return PlainTextResponse("invalid signature", status_code=401)
    payload = json.loads(parse_qs(body.decode())["payload"][0])
    work = process_payload(payload)
    return Response(status_code=200, background=BackgroundTask(*work) if work else None)


def process_payload(payload: dict) -> tuple | None:
    """Apply a button click. Returns (function, *args) for the slow follow-up work, or None.

    Shared by HTTP delivery and Socket Mode: the caller acknowledges Slack first, then runs
    the returned work (Slack message update, calendar call) in the background.
    """
    if payload.get("type") != "block_actions" or not payload.get("actions"):
        return None
    act = payload["actions"][0]
    action_id = act.get("action_id", "")
    if action_id == "open":  # link button: Slack still reports the click; nothing to do
        return None

    user = payload.get("user", {})
    who = user.get("name") or user.get("username") or "someone"
    user_ref = f"<@{user['id']}>" if user.get("id") else who
    value = act.get("value", "")
    ticket_id = int(value.split("|")[0]) if value.split("|")[0].isdigit() else -1
    team = value.split("|")[1] if "|" in value else ""
    message, response_url = payload.get("message", {}), payload.get("response_url")

    with store.connect() as con:
        t = store.get_ticket(con, ticket_id)
        if t is None:
            note = ":broom: This ticket was cleared (demo reset or clock moved back)."
            return (_post_update, response_url, _rebuild(message, None, note, False), note)
        if t["status"] == "resolved":
            note = f":white_check_mark: Already resolved by {t['handled_by'] or 'someone'}."
            return (_post_update, response_url, _rebuild(message, None, note, False), note)
        if action_id == "book_call":
            return (_book_call_and_update, ticket_id, who, user_ref, payload)
        if action_id not in ACTION_TEXT:
            return None

        text = ACTION_TEXT[action_id].format(team=team or "another team")
        plain = text.split(" ", 1)[1]
        if t.get("action_taken") != plain:  # idempotent: a double click is not logged twice
            store.record_action(con, ticket_id, status=STATUS_AFTER[action_id], action=plain, who=who)
    note = f"{text} by {user_ref}"
    keep = action_id != "resolve"
    return (_post_update, response_url, _rebuild(message, action_id, note, keep), note)
