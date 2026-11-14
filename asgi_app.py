"""Entry point: the Streamlit dashboard plus Slack button handling in the same process.

Run: streamlit run asgi_app.py

Slack clicks arrive one of two ways:
- Socket Mode (default for this Slack app): set SLACK_APP_TOKEN; no public URL needed.
- HTTP: Interactivity Request URL https://<public-domain>/slack/actions with SLACK_SIGNING_SECRET.
"""
from contextlib import asynccontextmanager
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from starlette.routing import Route

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)  # handlers run outside app.py, so load settings here too

import socket_mode  # noqa: E402
from slack_actions import slack_actions  # noqa: E402


@asynccontextmanager
async def lifespan(app):
    try:
        socket_mode.start()
    except Exception as e:  # bad token or no network: the dashboard still runs
        print(f"Slack Socket Mode not started: {e}")
    yield
    socket_mode.stop()


app = st.App(HERE / "app.py", lifespan=lifespan,
             routes=[Route("/slack/actions", slack_actions, methods=["POST"])])
