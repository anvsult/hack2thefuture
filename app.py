"""Millennium Mutual: Claims Delay Watch.

Run: streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from pathlib import Path  # noqa: E402

# SLACK_WEBHOOK_URL, GEMINI_API_KEY from the local .env file (gitignored).
load_dotenv(Path(__file__).parent / ".env", override=True)

import notify  # noqa: E402
import store
from data import EXTRACT_DATE, load, open_on, renewal_by_wait
from detect import (Thresholds, crossings, detect, premium_at_risk_total, renewal_uplift,
                    review_sla, risk_since)
from predict import evaluate, feature_importance, train

st.set_page_config(page_title="Claims Delay Watch", page_icon=":material/monitoring:", layout="wide")

# Reference categorical palette, slots 1-3 (validated all-pairs in light and dark).
TEAM_COLOR = {"Team A": "#2a78d6", "Team B": "#eb6834", "Team C": "#1baf7a"}
FOCUS, MUTED = "#2a78d6", "#a8a7a1"
MAX_SENDS_PER_RUN = 5          # Slack messages per rerun; the rest become one digest line
SIM_START, SIM_END = pd.Timestamp("2025-03-01"), EXTRACT_DATE
DEFAULT_DAY = pd.Timestamp("2026-06-26")  # one +1 week step fires a Team C reassign and a review stall


# ---------------------------------------------------------------- cached data
@st.cache_data
def get_data():
    p, c, log = load()
    return p, c, pd.DataFrame(log)


@st.cache_resource
def get_predictor(cutoff: pd.Timestamp, use_model: bool):
    _, c, _ = get_data()
    return train(c, cutoff, use_model=use_model)


@st.cache_data
def get_uplift():
    p, c, _ = get_data()
    return renewal_uplift(p, c)


@st.cache_data
def get_model_check():
    _, c, _ = get_data()
    rows = [evaluate(c, pd.Timestamp(s)) for s in ("2025-10-01", "2026-01-01", "2026-04-01", "2026-07-01")]
    return pd.DataFrame(rows), feature_importance(c, pd.Timestamp("2026-04-01"))


policies, claims, clean_log = get_data()
uplift = get_uplift()

# ---------------------------------------------------------------- clock state
if "day" not in st.session_state:
    st.session_state.day = DEFAULT_DAY.date()


def step(days: int) -> None:
    new = pd.Timestamp(st.session_state.day) + pd.Timedelta(days=days)
    st.session_state.day = min(max(new, SIM_START), SIM_END).date()


# Auto-play ticks are queued here and applied before the date widget is drawn.
if st.session_state.pop("pending_tick", False):
    step(1)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Simulated clock")
    st.date_input("Today", key="day", min_value=SIM_START.date(), max_value=SIM_END.date())
    c1, c2, c3 = st.columns(3)
    c1.button("+1 day", on_click=step, args=(1,), width="stretch")
    c2.button("+1 week", on_click=step, args=(7,), width="stretch")
    c3.button("+30 days", on_click=step, args=(30,), width="stretch")
    playing = st.toggle("Auto-play (1 day every 2 s)", value=False)

    st.header("Rules")
    warn = st.slider("Warn when age or prediction reaches (days)", 5, 40, 20)
    crit = st.slider("Critical when age or prediction reaches (days)", 10, 60, 30)
    estimator = st.radio("Days-to-close estimator", ["Lookup (team x amount)", "Gradient boosting"],
                         help="Both retrain monthly on claims closed before the 1st of the month. "
                              "See 'Data & method' for how they compare.")

    st.header("Actions")
    dry_run = st.toggle("Dry run (log only, do not post to Slack)", value=True)
    st.caption(f"Slack webhook: {'configured' if notify.slack_configured() else 'not set ($SLACK_WEBHOOK_URL)'}")
    use_ai = st.toggle("Gemini-written summaries for Slack posts", value=True,
                       disabled=not notify.ai_configured(),
                       help=None if notify.ai_configured() else "Set GEMINI_API_KEY in .env to enable.")
    if st.button("Reset demo (clear tickets)", type="secondary", width="stretch"):
        with store.connect() as con:
            store.reset(con)
        st.toast("Tickets cleared")

today = pd.Timestamp(st.session_state.day)
cutoff = today.replace(day=1)
predictor = get_predictor(cutoff, estimator.startswith("Gradient"))
th = Thresholds(warn_days=warn, critical_days=max(crit, warn + 1))
alerts = detect(claims, today, predictor, th, uplift["uplift"])


# ---------------------------------------------------------------- automation
def run_actions(alerts: pd.DataFrame) -> list[dict]:
    """Create a ticket for every new (claim, rule, tier); post the top ones to Slack."""
    new = []
    sim = today.date().isoformat()
    with store.connect() as con:
        removed = store.prune_after(con, sim)       # clock moved back: drop tickets from the future
    if removed:
        st.toast(f"Clock moved back: cleared {removed} ticket(s) dated after {sim}")
    if alerts.empty:
        return new
    with store.connect() as con:
        fresh = [a for a in alerts.to_dict("records")
                 if not store.already_sent(con, a["claim_id"], a["rule"], a["tier"])]
        fresh.sort(key=lambda a: (a["tier"] != "critical", -a["premium_at_risk"]))
        for i, a in enumerate(fresh):
            tid = store.reserve(con, a, sim)         # reserve first: reruns never double-send
            if tid is None:
                continue
            if i < MAX_SENDS_PER_RUN:
                live = not dry_run and notify.slack_configured()
                summary, src = (notify.ai_summary(a) if (use_ai and live)
                                else (notify.template_summary(a), "template"))
                slack = notify.send_slack(notify.slack_text(a, summary, sim), dry_run)
            else:
                summary, src, slack = notify.template_summary(a), "template", "in digest"
            store.finish(con, tid, summary, src, slack)
            new.append({**a, "slack": slack})
        extra = len(fresh) - MAX_SENDS_PER_RUN
        if extra > 0:
            notify.send_slack(f":package: {extra} more alerts on {sim}. See the ticket queue.", dry_run)
    return new


new_alerts = run_actions(alerts)
if new_alerts:
    sent = sum(a["slack"] == "sent" for a in new_alerts)
    failed = [a["slack"] for a in new_alerts if a["slack"].startswith("failed")]
    st.toast(f"{len(new_alerts)} new ticket(s) on {today.date()}"
             + (f", {sent} posted to Slack" if sent else ""))
    if failed:
        st.toast(f"Slack post failed: {failed[0]}", icon=":material/error:")
elif st.session_state.get("last_day") != today:
    st.toast(f"No new alerts on {today.date()}. Each claim alerts once per tier.")
st.session_state.last_day = today

# ---------------------------------------------------------------- header
st.title("Claims Delay Watch")
st.caption("Millennium Mutual · slow claims cost renewals · data extracted 2026-10-31")

open_now = open_on(claims, today)
age = (today - claims.loc[open_now, "reported_date"]).dt.days
cross = crossings(claims, uplift["uplift"])
year_start = today.replace(month=1, day=1)
k0, k1, k2, k3, k4 = st.columns(5)
k0.metric(f"Put at risk in {today.year}", f"${risk_since(cross, year_start, today):,.0f}",
          help="Claims that passed 30 days open since Jan 1, with the policy renewal still ahead: "
               f"premium x {uplift['uplift']:.0%}. One count per policy.")
k1.metric("Premium at risk now", f"${premium_at_risk_total(alerts):,.0f}",
          help=f"Critical alerts today with a future renewal: premium x {uplift['uplift']:.0%} "
               f"(renewal {uplift['fast_rate']:.0%} when claims close within 30 days vs "
               f"{uplift['slow_rate']:.0%} after 30 days).")
k2.metric("Open claims", int(open_now.sum()))
k3.metric("Open more than 30 days", int((age > 30).sum()))
k4.metric("Critical alerts", int((alerts["tier"] == "critical").sum()) if not alerts.empty else 0)

tab_exec, tab_alerts, tab_method = st.tabs(["Executive overview", "Live alerts & tickets", "Data & method"])

# ---------------------------------------------------------------- executive overview
with tab_exec:
    left, right = st.columns(2)

    with left:
        st.subheader("Renewal rate falls after a 30-day claim wait")
        rb = renewal_by_wait(policies, claims).reset_index()
        colors = [FOCUS if b == "30+ days" else MUTED for b in rb["bucket"]]
        fig = go.Figure(go.Bar(
            x=rb["bucket"], y=rb["renewal_rate"], marker_color=colors,
            marker_cornerradius=4,
            text=[f"{r:.0%}" for r in rb["renewal_rate"]], textposition="outside",
            customdata=rb[["policies"]],
            hovertemplate="%{x}<br>Renewal rate %{y:.1%}<br>%{customdata[0]} policies<extra></extra>"))
        fig.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1], title="Renewal rate"),
                          xaxis_title="Longest claim wait in the policy year", height=360,
                          margin=dict(t=10, b=10), bargap=0.35)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Policies whose term ended by 2026-10-31, so the renewal decision is known. "
                   f"30+ days: n={uplift['slow_n']}; 30 days or less: n={uplift['fast_n']}.")

    with right:
        st.subheader("Team C slowed down in 2026; the others did not")
        cq = claims.dropna(subset=["closed_date"]).copy()
        cq["q"] = cq["reported_date"].dt.to_period("Q")
        cq = cq[(cq["q"] >= pd.Period("2025Q1")) & (cq["q"] <= pd.Period("2026Q3"))]
        g = cq.groupby(["q", "adjuster_team"]).agg(days=("days_to_close", "mean"),
                                                   n=("claim_id", "count")).reset_index()
        fig = go.Figure()
        for team, d in g.groupby("adjuster_team"):
            fig.add_trace(go.Scatter(
                x=d["q"].astype(str), y=d["days"], name=team, mode="lines+markers",
                line=dict(color=TEAM_COLOR[team], width=2), marker=dict(size=8),
                customdata=d[["n"]],
                hovertemplate=f"{team}<br>%{{x}}<br>%{{y:.1f}} days to close<br>%{{customdata[0]}} claims<extra></extra>"))
        fig.add_hline(y=30, line_dash="dot", line_color=MUTED, annotation_text="30-day renewal cliff",
                      annotation_position="top left")
        fig.update_layout(yaxis=dict(title="Average days to close", rangemode="tozero"),
                          xaxis_title="Quarter reported", height=360, margin=dict(t=10, b=10),
                          hovermode="x unified", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")
        st.caption("Team C handles Montreal and Quebec City. Its claim volume did not rise "
                   "faster than Team A's, so volume alone does not explain the slowdown.")

    left2, right2 = st.columns(2)
    with left2:
        st.subheader("Claims passing 30 days open, by team")
        cq2 = cross[(cross["cross_date"] <= today) & (cross["cross_date"] >= pd.Timestamp("2025-01-01"))].copy()
        cq2["q"] = cq2["cross_date"].dt.to_period("Q").astype(str)
        g3 = cq2.groupby(["q", "adjuster_team"]).size().unstack(fill_value=0)
        fig = go.Figure()
        for team in TEAM_COLOR:
            if team in g3:
                fig.add_trace(go.Bar(x=g3.index, y=g3[team], name=team, marker_color=TEAM_COLOR[team],
                                     marker_cornerradius=4,
                                     hovertemplate=f"{team}<br>%{{x}}: %{{y}} claims<extra></extra>"))
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, height=320,
                          yaxis_title="Claims passing 30 days", margin=dict(t=10, b=10),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Grows with the simulated clock (through {today.date()}). "
                   "Each bar is a renewal put at risk; Team C jumps in 2026 Q2.")
    with right2:
        st.subheader("$10k+ claims wait about twice as long, on every team")
        cs = claims.dropna(subset=["closed_date", "claim_amount"])
        g2 = cs.groupby(["adjuster_team", "large"])["days_to_close"].mean().unstack()
        fig = go.Figure()
        for large, name, col in [(False, "Under $10k", MUTED), (True, "$10k+ (senior review)", FOCUS)]:
            fig.add_trace(go.Bar(x=g2.index, y=g2[large], name=name, marker_color=col,
                                 marker_cornerradius=4,
                                 hovertemplate="%{x}<br>" + name + ": %{y:.1f} days<extra></extra>"))
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, height=320,
                          yaxis_title="Average days to close", margin=dict(t=10, b=10),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

    left3, right3 = st.columns(2)
    with left3:
        st.subheader("Where the claims money goes: broker B07")
        pv = policies[policies["premium_valid"]].copy()
        paid = claims.groupby("policy_id")["paid_amount"].sum()
        pv["paid"] = pv["policy_id"].map(paid).fillna(0)
        br = pv.dropna(subset=["broker_code"]).groupby("broker_code").agg(
            prem=("annual_premium", "sum"), paid=("paid", "sum"))
        br["lr"] = br["paid"] / br["prem"]
        overall = pv["paid"].sum() / pv["annual_premium"].sum()
        fig = go.Figure(go.Bar(
            x=br.index, y=br["lr"], marker_color=[FOCUS if b == "B07" else MUTED for b in br.index],
            marker_cornerradius=4,
            hovertemplate="%{x}<br>Loss ratio %{y:.0%}<extra></extra>"))
        fig.add_hline(y=overall, line_dash="dot", line_color=MUTED,
                      annotation_text=f"Whole book {overall:.0%}", annotation_position="top right")
        fig.update_layout(yaxis=dict(title="Loss ratio (claims paid / premium)", tickformat=".0%"),
                          height=320, margin=dict(t=10, b=10), bargap=0.35)
        st.plotly_chart(fig, width="stretch")
        st.caption("B07 (mostly Toronto): 60 of its 70 claims are theft. "
                   "This is an underwriting question, not a reason to slow every broker.")

    with right3:
      st.subheader("Recommendation")
      st.markdown(
        "1. **Fix Team C, not the whole department.** Rebalance Montreal and Quebec City claims to "
        "Teams A and B now, and find out what changed at Team C in January 2026.\n"
        "2. **Add a fast lane for $10k+ senior review.** It doubles wait time on every team.\n"
        "3. **Keep this watch running.** Alert at 20 days, act before the 30-day renewal cliff.\n"
        "4. **Review broker B07's theft claims.** Do not slow the other brokers.")

# ---------------------------------------------------------------- live alerts
with tab_alerts:
    st.subheader(f"Alerts on {today.date()}")
    if alerts.empty:
        st.info("No open claims trigger a rule on this date.")
    else:
        show = alerts[["tier", "rule", "claim_id", "team", "city", "claim_type", "age", "predicted",
                       "days_to_renewal", "premium_at_risk", "reason", "next_step"]]
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "premium_at_risk": st.column_config.NumberColumn("Premium at risk", format="$%.0f"),
            "age": st.column_config.NumberColumn("Age (days)"),
            "predicted": st.column_config.NumberColumn("Predicted days"),
            "days_to_renewal": st.column_config.NumberColumn("Days to renewal"),
        })
        sla = review_sla(claims, cutoff)
        st.caption(f"Slow claim: warn at {th.warn_days} days, critical at {th.critical_days} days "
                   f"(age or predicted total). Review stall: $10k+ claim open {sla}+ days "
                   f"(75th percentile of past $10k+ close times; the data has no review timestamp).")

    st.subheader("Ticket queue")
    with store.connect() as con:
        tk = store.tickets(con)
    if tk.empty:
        st.info("No tickets yet. Move the clock forward.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Tickets", len(tk))
        c2.metric("Open", int((tk["status"] == "open").sum()))
        c3.metric("Posted to Slack", int((tk["slack_status"] == "sent").sum()))
        edited = st.data_editor(
            tk[["id", "status", "tier", "rule", "claim_id", "team", "owner", "sim_date",
                "next_step", "summary", "slack_status"]],
            hide_index=True, width="stretch", disabled=[c for c in tk.columns if c != "status"],
            column_config={"status": st.column_config.SelectboxColumn(
                "Status", options=["open", "in progress", "resolved"])},
            key=f"tickets_{len(tk)}")  # new tickets reset edit state so edits never shift rows
        changed = edited.merge(tk[["id", "status"]], on="id", suffixes=("", "_old"))
        changed = changed[changed["status"] != changed["status_old"]]
        if not changed.empty:
            with store.connect() as con:
                for r in changed.itertuples():
                    store.set_status(con, int(r.id), r.status)
            st.rerun()

# ---------------------------------------------------------------- data & method
with tab_method:
    st.subheader("Data cleaning log")
    st.dataframe(clean_log, hide_index=True, width="stretch")

    st.subheader("Can we predict slow claims at report time?")
    mc, fi = get_model_check()
    st.dataframe(mc.rename(columns={
        "mae_overall_median": "MAE: overall median", "mae_baseline": "MAE: lookup (team x amount)",
        "mae_model": "MAE: gradient boosting"}).round(1), hide_index=True, width="stretch")
    st.markdown(
        "Mean absolute error in days, testing on claims reported after each split date. "
        "**Gradient boosting does not beat the transparent lookup**, so the lookup is the default. "
        "Prediction at report time catches only about 1 in 4 claims that end up over 30 days, so the "
        "**age rule (20 and 30 days) does most of the detection** and the prediction gives early warning. "
        "Features are only what is known when a claim is reported; status, paid amount and close date are never used. "
        "Training uses closed claims only, which under-represents recent slow claims.")
    if not fi.empty:
        fig = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color=FOCUS,
                               marker_cornerradius=4,
                               hovertemplate="%{y}: %{x:.2f} days<extra></extra>"))
        fig.update_layout(height=300, margin=dict(t=10, b=10), yaxis=dict(autorange="reversed"),
                          xaxis_title="Increase in error when the feature is shuffled (days)")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Premium at risk: how it is computed")
    st.markdown(
        f"For each flagged claim whose policy renewal is still ahead: **annual premium x "
        f"{uplift['uplift']:.1%}**. The {uplift['uplift']:.1%} is the gap between the renewal rate "
        f"when every claim closed within 30 days ({uplift['fast_rate']:.1%}, n={uplift['fast_n']}) and "
        f"when a claim took longer ({uplift['slow_rate']:.1%}, n={uplift['slow_n']}). "
        "It is an estimate of expected lost renewals, not a guarantee. Policies with two alerts count once.")

# ---------------------------------------------------------------- auto-play
if playing:
    import time

    st.session_state.setdefault("last_tick", time.time())

    @st.fragment(run_every=2)
    def _tick():
        # The fragment also runs during every full rerun; only advance after a real 2 s wait.
        if time.time() - st.session_state.last_tick < 1.8:
            return
        st.session_state.last_tick = time.time()
        if pd.Timestamp(st.session_state.day) < SIM_END:
            st.session_state.pending_tick = True
            st.rerun()
    _tick()
else:
    st.session_state.pop("last_tick", None)
