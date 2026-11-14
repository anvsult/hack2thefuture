"""Slack Socket Mode: receive button clicks over an outbound WebSocket (no public URL, no tunnel).

Needs SLACK_APP_TOKEN (an app-level token, "xapp-...", with the connections:write scope).
Started once per process by asgi_app.py.
"""
from __future__ import annotations

import logging
import os
import threading

from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

from slack_actions import process_payload

log = logging.getLogger("claims.socket_mode")
_client: SocketModeClient | None = None


def handle(client: SocketModeClient, req: SocketModeRequest) -> None:
    if req.type != "interactive":
        return
    # Acknowledge first: Slack expects an answer within 3 seconds.
    client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
    try:
        work = process_payload(req.payload)
        if work:
            fn, *args = work
            threading.Thread(target=fn, args=args, daemon=True).start()
    except Exception:
        log.exception("Slack button click failed")


def start() -> bool:
    """Connect if SLACK_APP_TOKEN is set. Returns True when connected."""
    global _client
    token = os.environ.get("SLACK_APP_TOKEN", "")
    if not token.startswith("xapp-") or _client is not None:
        return _client is not None
    _client = SocketModeClient(app_token=token)
    _client.socket_mode_request_listeners.append(handle)
    _client.connect()
    log.warning("Slack Socket Mode connected: listening for button clicks")
    return True


def stop() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
