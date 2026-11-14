# Claims Delay Watch (Millennium Mutual)

Hack2TheFuture 2026, Business Track, Option A.

## Run

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in the values
./.venv/bin/streamlit run app.py
```

Settings live in `.env` (gitignored, never commit it). The app loads it on start.

- `SLACK_WEBHOOK_URL`: Slack incoming webhook for a private demo channel. Without it, alerts are logged only.
- `GEMINI_API_KEY`: free Google AI Studio key (https://aistudio.google.com/apikey). Enables AI-written summaries on live Slack posts. Without it, a fixed template is used.
- `GEMINI_MODEL` (optional): defaults to `gemini-3.5-flash-lite`.

## Files

| File | Role |
|---|---|
| `data.py` | Load and clean the CSVs; every fix is written to a cleaning log |
| `predict.py` | Days-to-close estimators (lookup and gradient boosting), retrained monthly, plus evaluation |
| `detect.py` | Business rules: slow claim, senior-review stall, premium at risk |
| `store.py` | SQLite tickets; one ticket per (claim, rule, tier) |
| `notify.py` | Gemini summary with template fallback; Slack post with dry-run |
| `app.py` | Streamlit screens, simulated clock, automation loop |

## Demo checklist

1. Click **Reset demo**. The date defaults to 2026-06-26, where one +1 week step fires a Team C reassignment and a $10k review stall. Keep **Dry run** on while rehearsing.
2. Before presenting: check `.env`, restart the app, and open the Slack channel on a second screen.
3. Live: turn **Dry run** off, press **+1 week** or start **Auto-play**, and show tickets and Slack posts appearing.
