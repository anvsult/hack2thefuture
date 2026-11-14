# Claims Delay Watch (Millennium Mutual)

Hack2TheFuture 2026, Business Track, Option A.

## Run

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in the values
./.venv/bin/streamlit run asgi_app.py --server.port 8501
```

`asgi_app.py` serves the dashboard (`app.py`) and the Slack button endpoint (`/slack/actions`) on the same port.

Settings live in `.env` (gitignored, never commit it). The app loads it on start.

- `SLACK_WEBHOOK_URL`: Slack incoming webhook for a private demo channel. Without it, alerts are logged only.
- `SLACK_APP_TOKEN`: app-level token (`xapp-...`) for Socket Mode. Button clicks arrive over a connection the app opens to Slack, so no public URL or tunnel is needed.
- `SLACK_SIGNING_SECRET` (only for HTTP delivery): verifies clicks sent to `/slack/actions`.
- `GEMINI_API_KEY`: free Google AI Studio key (https://aistudio.google.com/apikey). Enables AI-written summaries on live Slack posts. Without it, a fixed template is used.
- `GEMINI_MODEL` (optional): defaults to `gemini-3.5-flash-lite`.
- `PUBLIC_URL` (optional): target of the "Open dashboard" button in Slack.
- `GOOGLE_SERVICE_ACCOUNT_FILE` and `GOOGLE_CALENDAR_ID` (optional): create retention-call events directly in Google Calendar. Without them, the button gives a prefilled "Add to Google Calendar" link.

## Slack buttons: one-time setup (Socket Mode)

1. Open https://api.slack.com/apps and select the app that owns the incoming webhook.
2. **Socket Mode**: make sure it is enabled.
3. **Basic Information** > **App-Level Tokens** > **Generate Token and Scopes**: name it, add the scope `connections:write`, generate, and copy the `xapp-...` token into `.env` as `SLACK_APP_TOKEN`.
4. **Interactivity & Shortcuts**: make sure the toggle is on (no Request URL is needed in Socket Mode).
5. Restart the app. The terminal prints "Slack Socket Mode connected" when it is listening.

Button clicks go: Slack > WebSocket > ticket updated in SQLite > Slack message updated, and the dashboard's "Actions from Slack" feed refreshes within 3 seconds.

HTTP alternative (Socket Mode off): set the Interactivity Request URL to `https://<public-domain>/slack/actions`, set `SLACK_SIGNING_SECRET`, and run `ngrok http 8501`.

## Google Calendar API (optional upgrade)

1. In Google Cloud Console, create a project and enable the **Google Calendar API**.
2. Create a **service account**, add a JSON key, and save it outside the repo (or as `service-account.json`, which is gitignored).
3. In Google Calendar, open your calendar's **Settings and sharing** > **Share with specific people**, add the service account's email with **Make changes to events**.
4. In `.env`, set `GOOGLE_SERVICE_ACCOUNT_FILE` to the key's absolute path and `GOOGLE_CALENDAR_ID` to your calendar's email address.

## Files

| File | Role |
|---|---|
| `data.py` | Load and clean the CSVs; every fix is written to a cleaning log |
| `predict.py` | Days-to-close estimators (lookup and gradient boosting), retrained monthly, plus evaluation |
| `detect.py` | Business rules: slow claim, senior-review stall, premium at risk |
| `store.py` | SQLite tickets and the Slack activity log; one ticket per (claim, rule, tier) |
| `notify.py` | Gemini summary with template fallback; Slack post with buttons and dry-run |
| `slack_actions.py` | Button logic (reassign, escalate, fast-track, book call, resolve) and the signed HTTP endpoint |
| `socket_mode.py` | Socket Mode listener: receives button clicks without a public URL |
| `gcal.py` | Retention-call event: Google Calendar API or prefilled link |
| `app.py` | Streamlit screens, simulated clock, automation loop |
| `asgi_app.py` | Entry point: dashboard plus `/slack/actions` on one server |

## Demo checklist

1. Start the app (check the terminal says Socket Mode connected). Click **Reset demo**. The date defaults to 2026-06-26, where one +1 week step fires a Team C reassignment and a $10k review stall. Keep **Dry run** on while rehearsing.
2. Open the Slack channel on a second screen.
3. Live: turn **Dry run** off and press **+1 week**. In Slack, click **Reassign to Team B**, then **Book retention call**. Show the dashboard's "Actions from Slack" feed updating.
