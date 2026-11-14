"""Retention-call events in Google Calendar.

Two modes:
- API mode (GOOGLE_SERVICE_ACCOUNT_FILE and GOOGLE_CALENDAR_ID set): create the event directly
  with a service account that the calendar owner shared their calendar with.
- Link mode (default, no setup): return a prefilled "Add to Google Calendar" link.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from urllib.parse import quote, urlencode
from zoneinfo import ZoneInfo

import requests

TZ = ZoneInfo(os.environ.get("CALENDAR_TIMEZONE", "America/Toronto"))
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def next_slot(now: datetime | None = None) -> tuple[datetime, datetime]:
    """Next business day, 10:00 to 10:30 local time."""
    now = now or datetime.now(TZ)
    day = now.date() + timedelta(days=1)
    while day.weekday() >= 5:  # skip Saturday and Sunday
        day += timedelta(days=1)
    start = datetime(day.year, day.month, day.day, 10, 0, tzinfo=TZ)
    return start, start + timedelta(minutes=30)


def event_text(t: dict) -> tuple[str, str]:
    title = f"Retention call: policy {t['policy_id']} (claim {t['claim_id']})"
    details = (f"Claim {t['claim_id']} ({t['team']}, {t['city']}): {t['reason']}.\n"
               f"Renewal premium at risk: ${t['premium_at_risk'] or 0:,.0f}.\n"
               f"Goal: apologise for the delay, give a firm close date, protect the renewal.\n"
               f"Created from Claims Delay Watch.")
    return title, details


def template_link(t: dict) -> str:
    title, details = event_text(t)
    start, end = next_slot()
    fmt = "%Y%m%dT%H%M%S"
    q = urlencode({"action": "TEMPLATE", "text": title, "details": details,
                   "dates": f"{start.strftime(fmt)}/{end.strftime(fmt)}", "ctz": str(TZ)},
                  quote_via=quote)
    return f"https://calendar.google.com/calendar/render?{q}"


def api_configured() -> bool:
    return bool(os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE") and os.environ.get("GOOGLE_CALENDAR_ID"))


def create_event(t: dict, timeout: float = 8.0) -> tuple[str, str]:
    """Return (link, mode). mode is 'api' when the event was created, else 'link'."""
    if not api_configured():
        return template_link(t), "link"
    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"], scopes=SCOPES)
        creds.refresh(Request())
        title, details = event_text(t)
        start, end = next_slot()
        cal = quote(os.environ["GOOGLE_CALENDAR_ID"], safe="")
        r = requests.post(
            f"https://www.googleapis.com/calendar/v3/calendars/{cal}/events",
            headers={"Authorization": f"Bearer {creds.token}"},
            json={"summary": title, "description": details,
                  "start": {"dateTime": start.isoformat(), "timeZone": str(TZ)},
                  "end": {"dateTime": end.isoformat(), "timeZone": str(TZ)}},
            timeout=timeout)
        r.raise_for_status()
        return r.json().get("htmlLink", template_link(t)), "api"
    except Exception:  # bad key, calendar not shared, network: fall back to the link
        return template_link(t), "link"


def slot_label() -> str:
    start, _ = next_slot()
    return start.strftime("%a %b %-d, %-I:%M %p")
